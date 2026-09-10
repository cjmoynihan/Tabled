"""
Describing someone's taste back to them in words.

Two purposes. It tells the user the app is actually listening, which a list of
recommendations does not quite do on its own. And it makes the model
falsifiable to the person best placed to judge it: "you lean towards heavy
strategy games" is a claim they can immediately agree or disagree with, in a
way that a ranked list of ten titles is not.

Everything here is descriptive statistics over the games the user liked. There
is no model involved, deliberately — this should describe what they *said*,
not what the recommender inferred, or it would only ever agree with itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# BGG's category flags, in the order they read best in a sentence.
CATEGORIES = {
    "Cat:Strategy": "strategy",
    "Cat:Thematic": "thematic",
    "Cat:War": "wargames",
    "Cat:Family": "family",
    "Cat:Party": "party",
    "Cat:Abstract": "abstract",
    "Cat:CGS": "card",
    "Cat:Childrens": "children's",
}

WEIGHT_BANDS = [
    (1.8, "light, quick to teach"),
    (2.5, "fairly light"),
    (3.2, "middleweight"),
    (4.0, "meaty"),
    (5.1, "heavy, long evening"),
]


def weight_phrase(weight: float) -> str:
    for ceiling, phrase in WEIGHT_BANDS:
        if weight < ceiling:
            return phrase
    return WEIGHT_BANDS[-1][1]


def summarise(games: pd.DataFrame, liked: np.ndarray,
              min_liked: int = 3) -> dict | None:
    """
    A description of what this person likes, or None if it is too early.

    The `min_liked` floor matters. Two liked games will produce a confident
    sentence about someone's taste from almost no evidence, and being told
    something wrong about yourself is worse than being told nothing.
    """
    liked = np.asarray(liked, dtype=np.int64)
    if len(liked) < min_liked:
        return None

    rows = games.iloc[liked]
    weight = float(rows["GameWeight"].mean())

    categories = []
    for column, label in CATEGORIES.items():
        if column in rows.columns:
            share = float(rows[column].mean())
            if share >= 0.4:                    # most of what they liked
                categories.append((share, label))
    categories.sort(reverse=True)

    players = None
    if {"MinPlayers", "MaxPlayers"} <= set(rows.columns):
        low = int(np.median(rows["MinPlayers"]))
        high = int(np.median(rows["MaxPlayers"]))
        players = f"{low} to {high}" if high > low else f"{low}"

    playtime = None
    if "ComMaxPlaytime" in rows.columns:
        minutes = float(np.median(rows["ComMaxPlaytime"]))
        if minutes > 0:
            playtime = int(round(minutes / 15.0) * 15)

    return {
        "n_liked": int(len(liked)),
        "weight": round(weight, 1),
        "weight_phrase": weight_phrase(weight),
        "categories": [label for _, label in categories[:3]],
        "players": players,
        "playtime": playtime,
    }


def sentence(profile: dict | None) -> str:
    """The profile as one readable line."""
    if profile is None:
        return ""

    parts = [f"You lean towards **{profile['weight_phrase']}** games"]
    if profile["categories"]:
        kinds = profile["categories"]
        joined = kinds[0] if len(kinds) == 1 else (
            " and ".join(kinds) if len(kinds) == 2
            else f"{', '.join(kinds[:-1])} and {kinds[-1]}")
        parts.append(f"mostly **{joined}**")
    if profile["players"]:
        parts.append(f"usually for **{profile['players']} players**")
    if profile["playtime"]:
        parts.append(f"around **{profile['playtime']} minutes**")

    return ", ".join(parts) + "."
