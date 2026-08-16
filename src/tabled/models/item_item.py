"""
Item-item collaborative filtering: score a game by its similarity to what the
user already liked.

The appeal for a swipe app is structural rather than statistical. There is no
user model to estimate, so three swipes produce a real recommendation instead
of a noisy projection, and every suggestion carries its own explanation —
*because you liked X* — which a factorisation cannot give you.

Two implementation points carry most of the quality:

**Shrinkage.** Raw cosine treats two games sharing four enthusiastic raters
the same as two sharing four thousand. On a catalogue with a median of 125
raters per game, that fills the neighbour lists with coincidences. Every
similarity is multiplied by `n / (n + shrinkage)`, so thin evidence is pulled
toward zero and volume has to earn its place.

**Truncation.** The full similarity matrix is 17,140 x 17,140 — 1.2 GB dense,
and about 89% non-zero, so sparsity does not save us. Only the top `k`
neighbours of each game are kept, which is all scoring ever reads and turns
the artifact into a few megabytes.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp

from tabled.data.prepare import Dataset, mean_center
from tabled.models.base import Recommender, Swipes

log = logging.getLogger(__name__)


def _normalise_columns(matrix: sp.csr_matrix) -> sp.csc_matrix:
    """L2-normalise each game's column so a dot product is a cosine."""
    csc = matrix.tocsc()
    norms = np.sqrt(np.asarray(csc.multiply(csc).sum(axis=0))).ravel()
    norms[norms == 0] = 1.0
    return csc.multiply(sp.csr_matrix(1.0 / norms)).tocsc().astype(np.float32)


def neighbours(matrix: sp.csr_matrix, k: int = 100, shrinkage: float = 50.0,
               block: int = 256, block_range: tuple[int, int] | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Top-`k` neighbours of every game, as (indices, similarities).

    Computed a block of games at a time. The product of one block against the
    whole catalogue is ~89% dense, so the intermediate is materialised and
    immediately reduced to its top `k` — never assembling the full matrix.

    `block_range` restricts the work to games `[start, stop)`, which exists so
    a long fit can be split across processes and merged. Rows outside the
    range come back as -1 / 0 so partial results are obvious rather than
    silently looking like games with no neighbours.
    """
    centred, _ = mean_center(matrix)
    normed = _normalise_columns(centred)
    # Binary copy, for counting how many users rated both games in a pair.
    indicator = matrix.copy().tocsc()
    indicator.data = np.ones_like(indicator.data, dtype=np.float32)

    n_games = matrix.shape[1]
    start, stop = block_range or (0, n_games)
    idx = np.full((n_games, k), -1, dtype=np.int32)
    sim = np.zeros((n_games, k), dtype=np.float32)

    started = time.perf_counter()
    for lo in range(start, stop, block):
        hi = min(lo + block, stop)

        # Cosine, and the co-rater count for the same pairs. Two products of
        # equal cost; the count is what makes shrinkage possible at all.
        cos = np.asarray((normed[:, lo:hi].T @ normed).todense())
        common = np.asarray((indicator[:, lo:hi].T @ indicator).todense())

        cos *= common / (common + shrinkage)
        # A game is trivially its own best neighbour and must never be
        # recommended off the back of itself.
        cos[np.arange(hi - lo), np.arange(lo, hi)] = -np.inf

        top = np.argpartition(-cos, k - 1, axis=1)[:, :k]
        rows = np.arange(hi - lo)[:, None]
        order = np.argsort(-cos[rows, top], axis=1)
        top = top[rows, order]

        idx[lo:hi] = top
        sim[lo:hi] = np.maximum(cos[rows, top], 0.0)  # negatives are noise here
        log.info("  games %d-%d of %d (%.0fs)", lo, hi, stop,
                 time.perf_counter() - started)

    return idx, sim


class ItemItem(Recommender):
    """Score by summing similarities to the games the user reacted to."""

    name = "item-item"

    def __init__(self, k: int = 100, shrinkage: float = 50.0,
                 block: int = 256, prior_weight: float = 5.0) -> None:
        self.k = k
        self.shrinkage = shrinkage
        self.block = block
        self.prior_weight = prior_weight

    def fit(self, ds: Dataset) -> "ItemItem":
        self._remember_shape(ds)
        self._global_mean = float(ds.matrix.data.mean())
        self._idx, self._sim = neighbours(ds.matrix, self.k, self.shrinkage,
                                          self.block)
        return self

    def set_neighbours(self, idx: np.ndarray, sim: np.ndarray,
                       n_games: int, global_mean: float) -> "ItemItem":
        """Attach precomputed neighbours, bypassing the expensive fit."""
        self._idx, self._sim = idx, sim
        self._n_games = n_games
        self._global_mean = global_mean
        return self

    def score(self, swipes: Swipes) -> np.ndarray:
        if len(swipes) == 0:
            return np.zeros(self.n_games)

        # Centring on a shrunk mean is what lets a dislike push its
        # neighbourhood down rather than merely failing to push it up.
        weights = swipes.centred(self._global_mean, self.prior_weight)

        idx = self._idx[swipes.items]
        contribution = (weights[:, None] * self._sim[swipes.items]).ravel()
        flat = idx.ravel()

        valid = flat >= 0
        return np.bincount(flat[valid], weights=contribution[valid],
                           minlength=self.n_games)

    def similarity_to(self, game: int, other: int) -> float:
        """
        Similarity between two games, or 0 if `other` is not a near neighbour.

        Only the top-k list is stored, so this answers "are these two close"
        rather than "how close exactly" — which is all the callers need, and
        the reason the full 1.2 GB matrix never has to exist.
        """
        hit = np.flatnonzero(self._idx[game] == other)
        return float(self._sim[game, hit[0]]) if len(hit) else 0.0

    def because_of(self, swipes: Swipes, game: int) -> tuple[int, float] | None:
        """
        Which swiped game contributed most to `game`'s score.

        Free here, and not available from a factorisation at all: the score
        *is* a sum of per-neighbour terms, so the largest one is the reason.
        A swipe interface benefits from saying so out loud — it turns a
        recommendation into a claim the user can agree or disagree with,
        and it makes a bad neighbour list visible instead of merely felt.
        """
        if len(swipes) == 0:
            return None

        weights = swipes.centred(self._global_mean, self.prior_weight)
        best, best_value = None, 0.0
        for position, item in enumerate(swipes.items):
            hit = np.flatnonzero(self._idx[item] == game)
            if not len(hit):
                continue
            value = weights[position] * self._sim[item, hit[0]]
            if value > best_value:
                best, best_value = int(item), float(value)

        return (best, best_value) if best is not None else None
