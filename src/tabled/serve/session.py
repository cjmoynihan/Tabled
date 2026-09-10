"""
One person's swipe session, and the log of it.

Two reasons this is a module rather than a pile of Streamlit session state.
It is the thing the UI manipulates, so it should be testable without a
browser; and it is the record that Phase 4 depends on, since every number in
the project so far comes from *simulated* swiping reconstructed from ratings
people gave on a different site years ago.

The distinction that carries the most information is between the four
reactions. Collapsing "skip" into "dislike" would be the easy simplification
and would throw away the most interesting signal available: a skip usually
means unfamiliarity or indifference, not distaste, and a game the user has
already played is not a preference at all. Only likes and dislikes are
evidence about taste; all four are evidence about the session.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from tabled import config
from tabled.models.base import Swipes


class Reaction(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"            # shown, not chosen: indifference or ignorance
    PLAYED = "played"        # knows it already; useless as a suggestion
    DISMISS = "dismiss"      # "move past this", from the results list

    @property
    def is_taste(self) -> bool:
        """Whether this reaction should reach the model."""
        return self in (Reaction.LIKE, Reaction.DISLIKE)

    @property
    def is_settled(self) -> bool:
        """
        Whether the user is done with this game for good.

        Everything except SKIP. A skip on a card means "I don't know this
        one", which leaves the game eligible as a result; the rest are
        positions the user has taken and does not want revisited.
        """
        return self is not Reaction.SKIP


@dataclass
class Event:
    game: int
    reaction: Reaction
    position: int            # 3rd swipe and 30th mean different things
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {"game": self.game, "reaction": self.reaction.value,
                "position": self.position, "at": round(self.at, 3)}


@dataclass
class Session:
    events: list[Event] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def record(self, game: int, reaction: Reaction) -> None:
        self.events.append(Event(game=int(game), reaction=reaction,
                                 position=len(self.events)))

    def undo(self) -> Event | None:
        return self.events.pop() if self.events else None

    @property
    def shown(self) -> np.ndarray:
        """
        Every game already put in front of the user, for the card stream.

        All four reactions count. Re-showing a card someone skipped reads as
        the app not listening, even though a skip says nothing about taste.
        """
        return np.array([e.game for e in self.events], dtype=np.int64)

    @property
    def judged(self) -> np.ndarray:
        """
        Games the user has actually taken a position on, for the results list.

        Skips are excluded here but not from `shown`, and the difference is
        the point. On a card, a skip means "I don't know this one" — which is
        precisely the situation a recommendation exists to address. Hiding
        those games from the results would suppress the ones the user is most
        likely to find useful, purely because they were honest about not
        recognising them.

        So a skip means *ask me later*, not *never again*: the game stops
        interrupting the swipe stream but can still surface as a result, where
        it can be liked, passed on, or dismissed outright.
        """
        return np.array([e.game for e in self.events if e.reaction.is_settled],
                        dtype=np.int64)

    def swipes(self) -> Swipes:
        """Just the taste signal, in the form models consume."""
        taste = [e for e in self.events if e.reaction.is_taste]
        return Swipes.from_reactions(
            liked=[e.game for e in taste if e.reaction is Reaction.LIKE],
            disliked=[e.game for e in taste if e.reaction is Reaction.DISLIKE],
        )

    def counts(self) -> dict[str, int]:
        return {r.value: sum(e.reaction is r for e in self.events)
                for r in Reaction}

    # -- persistence -------------------------------------------------------

    def as_dict(self, games=None) -> dict:
        """
        The session as a loggable record.

        Game *names* are stored next to indices deliberately. Indices are only
        meaningful against one build of the catalogue, and a log that stops
        being readable after the next `tabled prepare` is not much of a log.
        """
        events = [e.as_dict() for e in self.events]
        if games is not None:
            for event in events:
                event["name"] = str(games.loc[event["game"], "Name"])
                event["bgg_id"] = int(games.loc[event["game"], "BGGId"])
        return {"started": round(self.started, 3),
                "events": events, "counts": self.counts()}

    def append_to(self, path: Path | None = None, games=None) -> Path:
        """Append this session to the log as one JSON line."""
        path = path or config.SESSIONS_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.as_dict(games)) + "\n")
        return path

    @classmethod
    def from_dict(cls, payload: dict, games=None) -> "Session":
        """
        Rebuild a session from an exported file.

        Restored by **BGGId**, not by matrix index, whenever the export
        carries one. Indices are positions in one particular build of the
        catalogue: rerun `tabled prepare` with different thresholds and index
        4,102 becomes a different game, so an index-based restore would
        silently return someone else's taste rather than failing.
        """
        by_bgg = None
        if games is not None:
            by_bgg = {int(b): int(i) for i, b in
                      zip(games["game_index"], games["BGGId"])}

        session = cls(started=float(payload.get("started") or time.time()))

        for raw in payload.get("events", []):
            game = int(raw["game"])
            if by_bgg is not None and "bgg_id" in raw:
                resolved = by_bgg.get(int(raw["bgg_id"]))
                if resolved is None:
                    continue          # game is no longer in the catalogue
                game = resolved
            session.events.append(Event(
                game=game,
                reaction=Reaction(raw["reaction"]),
                position=int(raw.get("position", len(session.events))),
                at=float(raw.get("at", session.started)),
            ))
        return session

    @classmethod
    def from_json(cls, text: str | bytes, games=None) -> "Session":
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return cls.from_dict(json.loads(text), games)

    def to_json(self, games=None) -> str:
        return json.dumps(self.as_dict(games), indent=2)
