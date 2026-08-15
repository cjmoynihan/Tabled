"""
Holding out whole users, not random cells.

This is the decision that makes the harness measure the right thing. The
usual approach — hide a random 20% of every user's ratings — asks "given
everything else this user told us, can you predict the rest". That is not the
task. The app meets people who have told it nothing, and answers by taking a
handful of swipes from someone the model has never seen. Splitting by user
reproduces that; splitting by cell quietly lets every test user's taste sit
inside the training data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from tabled.data.prepare import Dataset

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Split:
    train_rows: np.ndarray
    test_rows: np.ndarray

    def __repr__(self) -> str:
        return (f"Split(train={len(self.train_rows):,}, "
                f"test={len(self.test_rows):,})")


def by_user(ds: Dataset, n_test: int = 5000, min_ratings: int = 25,
            seed: int = 0) -> Split:
    """
    Reserve `n_test` sufficiently active users for testing.

    `min_ratings` exists because a test user has to supply both the swipes and
    the answers: with k=20 revealed, a user holding 20 ratings has nothing
    left to be scored against. Requiring the same floor for every arm of the
    sweep keeps the k values comparable — but it does mean the low-k results
    are measured on unusually active users, who are not quite the people the
    app will actually meet. Worth remembering before reading too much into
    absolute numbers; the comparison between models is the point.
    """
    per_user = np.diff(ds.matrix.indptr)
    eligible = np.flatnonzero(per_user >= min_ratings)
    if len(eligible) < n_test:
        raise ValueError(
            f"only {len(eligible):,} users have >= {min_ratings} ratings, "
            f"need {n_test:,} for the test set")

    rng = np.random.default_rng(seed)
    test_rows = np.sort(rng.choice(eligible, size=n_test, replace=False))

    is_test = np.zeros(ds.shape[0], dtype=bool)
    is_test[test_rows] = True
    train_rows = np.flatnonzero(~is_test)

    log.info("split: %s train users, %s test users (>= %d ratings)",
             f"{len(train_rows):,}", f"{len(test_rows):,}", min_ratings)
    return Split(train_rows=train_rows, test_rows=test_rows)
