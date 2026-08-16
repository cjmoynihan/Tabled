"""
Matrix factorisation on implicit feedback (Hu, Koren & Volinsky ALS).

This is the arm that tests the sparsity argument. At 0.47% density most pairs
of games share no raters at all, and item-item can only ever relate games
through users who rated both. Factorisation routes everything through a
shared latent space, so two games can end up close because they sit near the
same kinds of players, with no co-rater in common.

**Why implicit feedback rather than rating prediction.** The app shows a card,
not a number, so ranking is the objective and reconstructing someone's 7.5 is
beside the point. Binarising to likes also matches what the UI collects.

**Fold-in is where cold start actually happens.** A new user has no row in the
factorisation, so their vector is solved for directly against fixed item
factors — the same least-squares step ALS already runs internally, applied
once. That makes the cost of a new swipe a small solve rather than a refit,
which is what makes this usable in a live app at all.

**Dislikes are used, and that is not standard.** Implicit-feedback models
normally only see positives, because a missing click means nothing. A swipe
app is different: a thumbs-down is observed evidence. It enters as a
high-confidence observation of *zero* preference, which is exactly the case
the Hu-Koren-Volinsky confidence weighting was built to express.
"""

from __future__ import annotations

import logging

import numpy as np

from tabled import config
from tabled.data.prepare import Dataset, to_implicit
from tabled.models.base import Recommender, Swipes

log = logging.getLogger(__name__)


class ImplicitALS(Recommender):
    name = "als"

    def __init__(self, factors: int = 64, iterations: int = 15,
                 regularization: float = 0.05, alpha: float = 40.0,
                 threshold: float | None = None, seed: int = 0) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        # Confidence weight on an observation. Hu et al.'s c = 1 + alpha*r;
        # with binary r this is simply how much more an observed reaction
        # counts than the unobserved majority.
        self.alpha = alpha
        self.threshold = (config.LIKE_THRESHOLD if threshold is None
                          else threshold)
        self.seed = seed

    def fit(self, ds: Dataset) -> "ImplicitALS":
        import implicit

        self._remember_shape(ds)
        liked = to_implicit(ds.matrix, self.threshold)
        log.info("ALS on %s likes, %d factors, %d iterations",
                 f"{liked.nnz:,}", self.factors, self.iterations)

        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors, iterations=self.iterations,
            regularization=self.regularization, alpha=self.alpha,
            random_state=self.seed, calculate_training_loss=False)
        model.fit(liked, show_progress=False)

        return self.set_factors(np.asarray(model.item_factors,
                                           dtype=np.float64))

    def set_factors(self, item_factors: np.ndarray) -> "ImplicitALS":
        """Attach precomputed item factors, bypassing the expensive fit."""
        self._items = item_factors
        self._n_games = item_factors.shape[0]
        # Y^T Y is the same for every user and reappears in every fold-in, so
        # it is computed once here rather than per recommendation.
        self._YtY = item_factors.T @ item_factors
        return self

    @property
    def item_factors(self) -> np.ndarray:
        return self._items

    def fold_in(self, swipes: Swipes) -> np.ndarray:
        """
        Solve for a user vector from their swipes alone.

        Standard ALS user step: with preference p (1 for liked, 0 for
        disliked) and confidence c = 1 + alpha on everything observed,

            x = (YᵀY + Yₛᵀ(Cₛ - I)Yₛ + λI)⁻¹ Yₛᵀ Cₛ pₛ

        Only swiped rows appear in the correction term, which is what keeps
        this a small solve rather than a pass over the catalogue.
        """
        f = self._items.shape[1]
        if len(swipes) == 0:
            return np.zeros(f)

        seen = self._items[swipes.items]                     # (s, f)
        is_liked = swipes.ratings >= self.threshold

        # (C - I) = alpha for every observation, liked or not: we are equally
        # confident about both, and they differ only in preference.
        A = (self._YtY + seen.T @ (self.alpha * seen)
             + self.regularization * np.eye(f))
        b = (self.alpha + 1.0) * seen[is_liked].sum(axis=0)

        return np.linalg.solve(A, b)

    def score(self, swipes: Swipes) -> np.ndarray:
        return self._items @ self.fold_in(swipes)
