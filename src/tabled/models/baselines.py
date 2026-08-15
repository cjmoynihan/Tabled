"""
The controls. None of these look at what the user swiped.

That is exactly why they matter. Ratings on BGG are extremely concentrated —
the most-rated game has 108,000 ratings against a median of 125 — so simply
naming popular games hits a user's liked list surprisingly often. Any
personalised model has to beat that to have earned its complexity, and
measuring it first stops a mediocre recommender from looking impressive.

`Random` is the floor and does double duty: it is the only arm that gets
coverage and popularity-bias numbers right by construction, which makes it a
useful reference for how skewed the others are.
"""

from __future__ import annotations

import numpy as np

from tabled.data.prepare import Dataset
from tabled.models.base import Recommender, Swipes


class Popularity(Recommender):
    """Rank by how many people rated the game, ignoring the user entirely."""

    name = "popularity"

    def fit(self, ds: Dataset) -> "Popularity":
        self._remember_shape(ds)
        self._scores = ds.popularity().astype(np.float64)
        return self

    def score(self, swipes: Swipes) -> np.ndarray:
        return self._scores


class BayesianAverage(Recommender):
    """
    Rank by BGG's own shrunk average rating.

    Distinct from `Popularity` in an important way: it ranks by *quality*
    rather than volume, so it surfaces well-regarded niche games. Shrinking
    toward the global mean is what stops a game with nine 10s from topping
    the list, and it is the same trick item-item similarity will need against
    thinly co-rated pairs.
    """

    name = "bayes"

    def fit(self, ds: Dataset) -> "BayesianAverage":
        self._remember_shape(ds)
        scores = ds.games["BayesAvgRating"].to_numpy(dtype=np.float64)
        # A handful of games carry no Bayesian average. Rank them last with a
        # finite sentinel rather than -inf: -inf is how `recommend` marks a
        # game as ineligible, and these are merely unranked, not excluded.
        self._scores = np.nan_to_num(scores, nan=-1e9)
        return self

    def score(self, swipes: Swipes) -> np.ndarray:
        return self._scores


class Random(Recommender):
    """Uniformly random ranking — the floor any model must clear."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def fit(self, ds: Dataset) -> "Random":
        self._remember_shape(ds)
        return self

    def score(self, swipes: Swipes) -> np.ndarray:
        # A fresh draw per call. Reusing one permutation would make every user
        # get the same list, which quietly turns catalogue coverage from the
        # best score in the table into the worst.
        return self._rng.random(self.n_games)


ALL = {m.name: m for m in (Popularity, BayesianAverage, Random)}
