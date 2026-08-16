"""
The contract every recommender implements, and the type that represents what
a user has told us so far.

Both the evaluation harness and the app talk to models only through this, so
adding a model never means touching either. The asymmetry worth noticing is
that `fit` sees a matrix of *existing* users while `score` is called for a
user who is not in it — that gap is the whole problem this project is about,
and putting it in the interface keeps it visible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from tabled import config
from tabled.data.prepare import Dataset


@dataclass(frozen=True)
class Swipes:
    """
    One user's reactions, as game column indices and 1-10 ratings.

    Ratings rather than likes, even though the UI only collects thumbs
    up/down, because the training data is on the 1-10 scale and a model may
    reasonably want the magnitude. `from_reactions` maps the UI's binary
    signal onto the scale in one clearly marked place, so the assumption is
    auditable instead of scattered through each model.
    """

    items: np.ndarray            # int, column indices into the matrix
    ratings: np.ndarray          # float32, on the 1-10 scale

    def __post_init__(self) -> None:
        if len(self.items) != len(self.ratings):
            raise ValueError(
                f"{len(self.items)} items but {len(self.ratings)} ratings")

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def from_reactions(cls, liked=(), disliked=()) -> "Swipes":
        """Build from the UI's binary signal."""
        liked = np.asarray(liked, dtype=np.int64)
        disliked = np.asarray(disliked, dtype=np.int64)
        overlap = np.intersect1d(liked, disliked)
        if len(overlap):
            raise ValueError(f"games both liked and disliked: {overlap[:5]}")

        return cls(
            items=np.concatenate([liked, disliked]),
            ratings=np.concatenate([
                np.full(len(liked), config.LIKE_VALUE, dtype=np.float32),
                np.full(len(disliked), config.DISLIKE_VALUE, dtype=np.float32),
            ]),
        )

    @property
    def liked(self) -> np.ndarray:
        return self.items[self.ratings >= config.LIKE_THRESHOLD]

    @property
    def disliked(self) -> np.ndarray:
        return self.items[self.ratings < config.LIKE_THRESHOLD]

    def centred(self, prior_mean: float | None = None,
                prior_weight: float = 0.0) -> np.ndarray:
        """
        Ratings with this user's mean removed, optionally shrunk to a prior.

        Same reasoning as centring the training matrix: a user who rates
        everything 8-10 is not telling us they love everything, only that
        their scale starts high.

        The prior matters more than it looks. Centring on the raw mean is
        degenerate at one swipe — the mean *is* that rating, so it centres to
        exactly zero and the model receives no signal at all. Shrinking toward
        the training mean fixes that: with `prior_weight` pseudo-observations,
        a single 9 still reads as "above average" instead of "no opinion", and
        the user's own mean takes over as swipes accumulate.
        """
        if len(self) == 0:
            return self.ratings
        if prior_mean is None or prior_weight <= 0:
            return self.ratings - self.ratings.mean()

        n = len(self)
        centre = (n * self.ratings.mean() + prior_weight * prior_mean) / (n + prior_weight)
        return self.ratings - centre


class Recommender(ABC):
    """Fit on the training matrix, then score games for an unseen user."""

    name: str = "recommender"

    @abstractmethod
    def fit(self, ds: Dataset) -> "Recommender":
        """Learn whatever is needed. Must return self."""

    @abstractmethod
    def score(self, swipes: Swipes) -> np.ndarray:
        """
        A score per game, length `n_games`, higher is better.

        Scores are compared only within one call, so they need no particular
        scale — but they must be finite, since `recommend` uses -inf to mask.
        """

    def recommend(self, swipes: Swipes, n: int = 10,
                  exclude: np.ndarray | None = None) -> np.ndarray:
        """
        Top `n` game indices, best first, excluding anything already swiped.

        Excluding swiped games is not a detail: a recommender that returns a
        game the user just reacted to looks broken regardless of how good its
        ranking is, and in evaluation it would score free points for
        rediscovering the input.
        """
        scores = np.asarray(self.score(swipes), dtype=np.float64).copy()
        if scores.shape != (self.n_games,):
            raise ValueError(
                f"{self.name}.score returned {scores.shape}, "
                f"expected ({self.n_games},)")

        scores[swipes.items] = -np.inf
        if exclude is not None and len(exclude):
            scores[exclude] = -np.inf

        n = min(n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return np.empty(0, dtype=np.int64)

        top = np.argpartition(-scores, n - 1)[:n]
        return top[np.argsort(-scores[top])]

    @property
    def n_games(self) -> int:
        return self._n_games

    def _remember_shape(self, ds: Dataset) -> None:
        self._n_games = ds.shape[1]
