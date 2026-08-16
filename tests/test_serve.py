"""
Tests for the card-selection layer.

None of this needs Streamlit. That is the point of keeping the logic in
`tabled.serve` — the interesting behaviour is "what does this app show next
and why", and it should be checkable without a browser.

The last test drives a whole session headlessly against a synthetic user with
a fixed taste, which is the closest thing to actually using the app.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tabled.models.base import Swipes
from tabled.models.item_item import ItemItem
from tabled.serve import policy
from tabled.serve.session import Reaction, Session

CORE = 20
BLOCK = 40


def cluster_of(game: int) -> int:
    return -1 if game < CORE else (game - CORE) // BLOCK


@pytest.fixture
def model(clustered):
    return ItemItem(k=20, shrinkage=1.0).fit(clustered)


# -- session ---------------------------------------------------------------

def test_only_likes_and_dislikes_reach_the_model():
    s = Session()
    s.record(1, Reaction.LIKE)
    s.record(2, Reaction.DISLIKE)
    s.record(3, Reaction.SKIP)
    s.record(4, Reaction.PLAYED)

    swipes = s.swipes()
    assert set(swipes.items.tolist()) == {1, 2}
    assert set(swipes.liked.tolist()) == {1}
    assert set(swipes.disliked.tolist()) == {2}


def test_every_reaction_counts_as_shown():
    """A skipped card must not come back; it just says nothing about taste."""
    s = Session()
    for game, reaction in enumerate(Reaction):
        s.record(game, reaction)

    assert sorted(s.shown.tolist()) == [0, 1, 2, 3]


def test_skip_and_dislike_are_not_the_same_signal():
    skipped, disliked = Session(), Session()
    skipped.record(5, Reaction.SKIP)
    disliked.record(5, Reaction.DISLIKE)

    assert len(skipped.swipes()) == 0
    assert len(disliked.swipes()) == 1


def test_position_is_recorded():
    s = Session()
    for game in range(3):
        s.record(game, Reaction.LIKE)

    assert [e.position for e in s.events] == [0, 1, 2]


def test_undo_removes_the_last_reaction():
    s = Session()
    s.record(1, Reaction.LIKE)
    s.record(2, Reaction.DISLIKE)

    assert s.undo().game == 2
    assert len(s.swipes()) == 1
    s.undo()
    assert s.undo() is None


def test_session_log_round_trips(tmp_path, clustered):
    s = Session()
    s.record(3, Reaction.LIKE)
    s.record(7, Reaction.SKIP)
    path = s.append_to(tmp_path / "swipes.jsonl", games=clustered.games)

    line = json.loads(path.read_text(encoding="utf-8").strip())
    assert line["counts"]["like"] == 1
    assert line["counts"]["skip"] == 1
    assert [e["game"] for e in line["events"]] == [3, 7]
    # Names travel with the log, so it survives a rebuild of the catalogue.
    assert line["events"][0]["name"] == clustered.games.loc[3, "Name"]


def test_log_appends_rather_than_overwrites(tmp_path):
    path = tmp_path / "swipes.jsonl"
    for game in (1, 2):
        s = Session()
        s.record(game, Reaction.LIKE)
        s.append_to(path)

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


# -- policy ----------------------------------------------------------------

def test_the_gate_relaxes_as_the_session_goes_on(clustered):
    popularity = clustered.popularity()
    gates = [policy.popularity_gate(n, popularity) for n in range(0, 20)]

    assert gates == sorted(gates, reverse=True)
    assert gates[-1] < gates[0]
    # It stops relaxing once the session is well under way.
    assert (policy.popularity_gate(policy.GATE_SWIPES, popularity)
            == policy.popularity_gate(100, popularity))


def test_the_gate_scales_to_the_catalogue_it_is_given(clustered):
    """
    An absolute rating floor means something different on every catalogue.
    The gate should always admit roughly the same *fraction* of games,
    whatever the popularity numbers happen to look like.
    """
    popularity = clustered.popularity()
    for scale in (1, 100, 10_000):
        gate = policy.popularity_gate(0, popularity * scale)
        admitted = (popularity * scale >= gate).mean()
        assert 0.005 < admitted < 0.05, f"scale {scale} admitted {admitted:.1%}"


def test_cold_start_picks_recognisable_games(clustered):
    """Seeds must come from the well-known end of the catalogue, or the
    opening question is wasted on a stranger."""
    seeds = policy.cold_start_seeds(clustered, n=8, pool=40)
    popularity = clustered.popularity()

    assert (popularity[seeds] >= np.median(popularity)).all()


def test_cold_start_spans_the_weight_range(clustered):
    seeds = policy.cold_start_seeds(clustered, n=12, pool=60)
    weights = clustered.games["GameWeight"].to_numpy()[seeds]

    assert weights.max() - weights.min() > 1.0, "opening hand is all one thing"


def test_first_card_comes_from_the_seeds(clustered, model):
    seeds = policy.cold_start_seeds(clustered, n=8, pool=40)
    card = policy.next_card(model, Session().swipes(), clustered,
                            np.array([], dtype=np.int64),
                            np.random.default_rng(0), seeds=seeds)

    assert card in set(seeds.tolist())


def test_a_card_is_never_shown_twice(clustered, model):
    session = Session()
    rng = np.random.default_rng(0)
    seeds = policy.cold_start_seeds(clustered, n=8, pool=40)

    for _ in range(40):
        card = policy.next_card(model, session.swipes(), clustered,
                                session.shown, rng, seeds=seeds)
        assert card is not None
        assert card not in session.shown.tolist()
        session.record(card, Reaction.LIKE)


def test_exhausting_the_catalogue_returns_none(clustered, model):
    everything = np.arange(clustered.shape[1])
    assert policy.next_card(model, Swipes.from_reactions(liked=[0]), clustered,
                            everything, np.random.default_rng(0)) is None


def test_gate_is_a_preference_not_a_rule(clustered, model):
    """If the gate would leave nothing showable, we show something anyway
    rather than claiming to have run out of board games."""
    popular = np.argsort(-clustered.popularity())[:60]
    card = policy.next_card(model, Swipes.from_reactions(liked=[CORE + 1]),
                            clustered, popular, np.random.default_rng(0))

    assert card is not None


def test_zero_temperature_takes_the_top_card(clustered, model):
    swipes = Swipes.from_reactions(liked=[CORE + 1, CORE + 2])
    shown = np.array([], dtype=np.int64)
    picks = {policy.next_card(model, swipes, clustered, shown,
                              np.random.default_rng(s), temperature=0.0)
             for s in range(5)}

    assert len(picks) == 1, "temperature=0 should be deterministic"


def test_sampling_varies_between_sessions(clustered, model):
    """Two users with identical taste should not get identical sessions."""
    swipes = Swipes.from_reactions(liked=[CORE + 1, CORE + 2])
    shown = np.array([], dtype=np.int64)
    picks = {policy.next_card(model, swipes, clustered, shown,
                              np.random.default_rng(s), temperature=1.0)
             for s in range(25)}

    assert len(picks) > 1


def test_explanation_names_a_game_the_user_liked(clustered, model):
    swipes = Swipes.from_reactions(liked=[CORE + 1])
    ranked = model.recommend(swipes, n=3)
    why = policy.explain(model, swipes, int(ranked[0]), clustered.games)

    assert why == clustered.games.loc[CORE + 1, "Name"]


def test_explanation_is_absent_when_the_model_cannot_give_one(clustered):
    from tabled.models.baselines import Popularity

    model = Popularity().fit(clustered)
    assert policy.explain(model, Swipes.from_reactions(liked=[1]), 2,
                          clustered.games) is None


# -- the whole loop --------------------------------------------------------

def test_a_full_session_converges_on_the_users_taste(clustered, model):
    """
    Drive the app headlessly. A synthetic user likes cluster 0 and dislikes
    everything else; by the end of the session the cards being served should
    be overwhelmingly from cluster 0.
    """
    session = Session()
    rng = np.random.default_rng(0)
    seeds = policy.cold_start_seeds(clustered, n=8, pool=40)
    served = []

    for _ in range(30):
        card = policy.next_card(model, session.swipes(), clustered,
                                session.shown, rng, seeds=seeds)
        if card is None:
            break
        served.append(card)
        session.record(card, Reaction.LIKE if cluster_of(card) == 0
                       else Reaction.DISLIKE)

    early = [c for c in served[:10] if cluster_of(c) == 0]
    late = [c for c in served[-10:] if cluster_of(c) == 0]

    assert len(late) > len(early), (
        f"session did not converge: {len(early)} on-taste early, "
        f"{len(late)} late")
    assert len(late) >= 7, f"only {len(late)}/10 late cards were on-taste"
