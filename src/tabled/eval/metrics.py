"""
Ranking metrics, plus the two that stop accuracy being read in isolation.

Everything here takes an already-ranked array of game indices and a set of
relevant ones. Deliberately no rating prediction and no RMSE: the app never
shows a predicted score, it shows one game, so what matters is whether good
games appear near the top of a list.
"""

from __future__ import annotations

import numpy as np


def hits(ranked: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """Boolean mask over `ranked`: was each recommendation a good one?"""
    return np.isin(ranked, relevant, assume_unique=False)


def recall_at_n(ranked: np.ndarray, relevant: np.ndarray, n: int) -> float:
    """
    Fraction of the user's liked games that appear in the top `n`.

    Capped at `n / len(relevant)` when a user likes more games than the list
    is long, which is the usual criticism of recall here — a user with 200
    liked games cannot score above 0.1 at n=20 no matter how good the model
    is. It is kept because it stays comparable across models on the same
    cohort, and `hit_rate` covers the "did we find anything" question without
    that ceiling.
    """
    if len(relevant) == 0:
        return float("nan")
    return float(hits(ranked[:n], relevant).sum() / len(relevant))


def hit_rate_at_n(ranked: np.ndarray, relevant: np.ndarray, n: int) -> float:
    """Did the top `n` contain at least one game the user liked?"""
    if len(relevant) == 0:
        return float("nan")
    return float(hits(ranked[:n], relevant).any())


def ndcg_at_n(ranked: np.ndarray, relevant: np.ndarray, n: int) -> float:
    """
    Normalised discounted cumulative gain, binary relevance.

    Unlike recall this rewards *where* in the list a hit lands, which is the
    thing a swipe interface actually cares about — the user sees the top card,
    not the whole list. Normalising by the best achievable DCG for this user
    keeps it comparable across users with different numbers of liked games.
    """
    if len(relevant) == 0:
        return float("nan")

    gains = hits(ranked[:n], relevant) / np.log2(np.arange(2, len(ranked[:n]) + 2))
    ideal = (1 / np.log2(np.arange(2, min(len(relevant), n) + 2))).sum()
    return float(gains.sum() / ideal) if ideal > 0 else float("nan")


def coverage(recommended: set[int], n_games: int) -> float:
    """
    Fraction of the catalogue that ever got recommended to anyone.

    A recommender that only ever serves the top 200 games can score well on
    every accuracy metric above and still be useless as a discovery tool,
    which is the entire premise of this app. This is the check for that.
    """
    return len(recommended) / n_games if n_games else float("nan")


def popularity_percentile(ranked: np.ndarray,
                          popularity_rank: np.ndarray) -> float:
    """
    Where recommendations sit in the catalogue's popularity order, 0-100.

    0 means "only ever suggests the single most-rated game", 50 means the
    median game. Read next to coverage: together they say whether a model is
    finding things the user could not have found by browsing the front page.
    """
    if len(ranked) == 0:
        return float("nan")
    return float(np.median(popularity_rank[ranked]) * 100
                 / max(len(popularity_rank) - 1, 1))
