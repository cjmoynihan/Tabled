"""
Replay the swipe flow against held-out users.

For each test user: reveal `k` of their ratings as if they had just swiped
them, ask the model for a top-N list, and check it against the games they
liked but were never shown. Sweeping `k` is the experiment, not a detail —
the interesting question is not which model is better overall but which is
better *at the number of swipes the app actually has*, which early on is
three or four.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from tabled import config
from tabled.data.prepare import Dataset
from tabled.eval import metrics
from tabled.models.base import Recommender, Swipes

log = logging.getLogger(__name__)

# How the k revealed swipes are chosen from a test user's ratings.
#   random     — unbiased, the standard choice, and the one to quote
#   popular    — the games the app would actually show first, since a card for
#                a game nobody recognises earns a shrug rather than a signal
#   favourites — the user's own highest-rated games, standing in for an
#                "add some games you love" picker. Note that this reveals only
#                positives: someone naming favourites volunteers no dislikes,
#                which is a real cost the comparison has to include.
SEED_POLICIES = ("random", "popular", "favourites")


@dataclass
class Result:
    model: str
    k: int
    n: int
    policy: str
    users: int
    recall: float
    ndcg: float
    hit_rate: float
    coverage: float
    popularity_pct: float
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Accumulator:
    recall: list = field(default_factory=list)
    ndcg: list = field(default_factory=list)
    hit: list = field(default_factory=list)
    pop: list = field(default_factory=list)
    seen: set = field(default_factory=set)


def _reveal(items: np.ndarray, ratings: np.ndarray, k: int, policy: str,
            rng: np.random.Generator, popularity: np.ndarray) -> np.ndarray:
    """Positions within `items` to expose as swipes."""
    if policy == "random":
        return rng.choice(len(items), size=k, replace=False)
    if policy == "popular":
        return np.argsort(-popularity[items])[:k]
    if policy == "favourites":
        # Ties broken by popularity: asked to name games they love, people
        # reach for ones they can actually remember.
        order = np.lexsort((-popularity[items], -ratings))
        return order[:k]
    raise ValueError(f"unknown seed policy {policy!r}")


def evaluate(model: Recommender, ds: Dataset, test_rows: np.ndarray,
             ks: tuple[int, ...] = (3, 5, 10, 20), n: int = 20,
             policy: str = "random", seed: int = 0,
             like_threshold: float | None = None) -> list[Result]:
    """
    Score one already-fitted model across the k sweep.

    `ds` must be the full dataset — test users' rows are read from it to build
    their swipes. The model itself must have been fitted on the training rows
    only; nothing here checks that, because `fit` has already happened by the
    time we get here, so it is the caller's job to keep the split honest.
    """
    threshold = config.LIKE_THRESHOLD if like_threshold is None else like_threshold
    popularity = ds.popularity()

    # Rank 0 = most-rated game. Used to describe how mainstream a model's
    # suggestions are, not to score them.
    popularity_rank = np.empty(ds.shape[1], dtype=np.int64)
    popularity_rank[np.argsort(-popularity)] = np.arange(ds.shape[1])

    matrix = ds.matrix
    results = []

    for k in ks:
        rng = np.random.default_rng(seed)
        acc = _Accumulator()
        started = time.perf_counter()

        for row in test_rows:
            lo, hi = matrix.indptr[row], matrix.indptr[row + 1]
            items, ratings = matrix.indices[lo:hi], matrix.data[lo:hi]
            if len(items) <= k:
                continue

            shown = _reveal(items, ratings, k, policy, rng, popularity)
            held = np.setdiff1d(np.arange(len(items)), shown,
                                assume_unique=False)
            relevant = items[held][ratings[held] >= threshold]
            if len(relevant) == 0:
                # Nothing to find. Keeping these users would reward every
                # model equally with a zero and dilute the comparison.
                continue

            ranked = model.recommend(
                Swipes(items=items[shown], ratings=ratings[shown]), n=n)

            acc.recall.append(metrics.recall_at_n(ranked, relevant, n))
            acc.ndcg.append(metrics.ndcg_at_n(ranked, relevant, n))
            acc.hit.append(metrics.hit_rate_at_n(ranked, relevant, n))
            acc.pop.append(metrics.popularity_percentile(ranked, popularity_rank))
            acc.seen.update(ranked.tolist())

        results.append(Result(
            model=model.name, k=k, n=n, policy=policy, users=len(acc.recall),
            recall=float(np.mean(acc.recall)) if acc.recall else float("nan"),
            ndcg=float(np.mean(acc.ndcg)) if acc.ndcg else float("nan"),
            hit_rate=float(np.mean(acc.hit)) if acc.hit else float("nan"),
            coverage=metrics.coverage(acc.seen, ds.shape[1]),
            popularity_pct=float(np.mean(acc.pop)) if acc.pop else float("nan"),
            seconds=round(time.perf_counter() - started, 1),
        ))
        log.info("  %-12s k=%-3d recall %.4f  ndcg %.4f  hit %.3f  (%ss)",
                 model.name, k, results[-1].recall, results[-1].ndcg,
                 results[-1].hit_rate, results[-1].seconds)

    return results


def format_table(results: list[Result]) -> str:
    """The scoreboard, grouped by k so models are read against each other."""
    lines = [
        f"{'model':<14}{'k':>4}{'users':>8}{'recall@N':>11}{'nDCG@N':>10}"
        f"{'hit@N':>8}{'coverage':>10}{'pop %ile':>10}",
        "-" * 75,
    ]
    for k in sorted({r.k for r in results}):
        for r in sorted((r for r in results if r.k == k),
                        key=lambda r: -r.ndcg):
            lines.append(
                f"{r.model:<14}{r.k:>4}{r.users:>8,}{r.recall:>11.4f}"
                f"{r.ndcg:>10.4f}{r.hit_rate:>8.3f}"
                f"{r.coverage:>9.1%}{r.popularity_pct:>10.1f}")
        lines.append("")
    return "\n".join(lines)
