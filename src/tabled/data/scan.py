"""
One streaming pass over user_ratings.csv, producing support counts.

Both `profile` and `prepare` need to know how many ratings each game and each
user has before they can do anything useful — profile to describe the data,
prepare to decide what to throw away. The pass costs about a minute on the
full 400 MB file, so it lives in one place and both callers share it.

Nothing here loads the whole file into memory: the counters are keyed by the
number of *distinct* games (~22k) and users (~400k), not by the 19M rows.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tabled import config

log = logging.getLogger(__name__)

RATING_COLUMNS = ["BGGId", "Rating", "Username"]
RATING_DTYPES = {"BGGId": "int32", "Rating": "float32", "Username": "str"}

# Bump whenever a change to `clean` or `count` would produce different counts
# for the same input file. The cache is keyed on the source file, so without
# this a code change would keep silently serving counts computed by the old
# rules.
CACHE_VERSION = 2


@dataclass(frozen=True)
class Counts:
    """Support counts for one ratings file, plus the rating value histogram."""

    game_ids: np.ndarray        # int32, ascending
    game_counts: np.ndarray     # int64, aligned to game_ids
    usernames: np.ndarray       # object (str), sorted
    user_counts: np.ndarray     # int64, aligned to usernames
    rating_hist: dict[float, int]
    n_rows: int

    @property
    def n_games(self) -> int:
        return len(self.game_ids)

    @property
    def n_users(self) -> int:
        return len(self.usernames)

    @property
    def density(self) -> float:
        """Fraction of the user x game grid that is actually filled."""
        cells = self.n_users * self.n_games
        return self.n_rows / cells if cells else 0.0

    def games_with_at_least(self, n: int) -> np.ndarray:
        return self.game_ids[self.game_counts >= n]

    def users_with_at_least(self, n: int) -> np.ndarray:
        return self.usernames[self.user_counts >= n]


def chunks(path: Path, chunk_rows: int | None = None):
    """Yield the ratings file in memory-bounded pieces."""
    return pd.read_csv(
        path,
        chunksize=chunk_rows or config.CHUNK_ROWS,
        usecols=RATING_COLUMNS,
        dtype=RATING_DTYPES,
    )


def clean(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that cannot legitimately become a matrix cell.

    Two kinds: rows missing an id, username, or rating, which have nowhere to
    go; and ratings outside the 1-10 scale. The second is rare — eight rows in
    the whole export — but three of them are zero, and a zero written into a
    sparse matrix is the same thing as an absent rating. That would turn a
    data error into a silently wrong observation rather than a loud one.

    Both `count` and the matrix build use this, so the support counts always
    describe exactly the rows that can end up in the matrix.
    """
    chunk = chunk.dropna(subset=RATING_COLUMNS)
    on_scale = chunk["Rating"].between(config.RATING_MIN, config.RATING_MAX)
    return chunk[on_scale]


def count(path: Path, chunk_rows: int | None = None) -> Counts:
    """Single pass over the ratings CSV, accumulating support counts."""
    per_game: Counter = Counter()
    per_user: Counter = Counter()
    per_value: Counter = Counter()
    n_rows = 0

    for i, chunk in enumerate(chunks(path, chunk_rows)):
        chunk = clean(chunk)
        n_rows += len(chunk)
        per_game.update(chunk["BGGId"].value_counts().to_dict())
        per_user.update(chunk["Username"].value_counts().to_dict())
        per_value.update(chunk["Rating"].round(1).value_counts().to_dict())
        log.info("  scanned %s rows", f"{n_rows:,}")

    game_ids = np.array(sorted(per_game), dtype=np.int32)
    usernames = np.array(sorted(per_user), dtype=object)

    return Counts(
        game_ids=game_ids,
        game_counts=np.array([per_game[g] for g in game_ids], dtype=np.int64),
        usernames=usernames,
        user_counts=np.array([per_user[u] for u in usernames], dtype=np.int64),
        rating_hist={float(k): int(v) for k, v in sorted(per_value.items())},
        n_rows=n_rows,
    )


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------

def _save(counts: Counts, cache: Path, source: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        game_ids=counts.game_ids,
        game_counts=counts.game_counts,
        usernames=counts.usernames,
        user_counts=counts.user_counts,
        hist_values=np.array(list(counts.rating_hist), dtype=np.float64),
        hist_freqs=np.array(list(counts.rating_hist.values()), dtype=np.int64),
        n_rows=np.int64(counts.n_rows),
        source_size=np.int64(source.stat().st_size),
        version=np.int64(CACHE_VERSION),
    )


def count_cached(path: Path, cache: Path, chunk_rows: int | None = None,
                 refresh: bool = False) -> Counts:
    """
    `count`, but remembering the answer between runs.

    The counts depend only on the file, not on any threshold, so re-scanning
    400 MB every time someone wants to try a different cutoff is pure waste —
    and choosing thresholds well means trying several. The cache is keyed on
    both the source file's size and `CACHE_VERSION`, so replacing the export
    *or* changing the counting rules invalidates it rather than silently
    returning counts computed under different assumptions.
    """
    if cache.exists() and not refresh:
        stored = np.load(cache, allow_pickle=True)
        version = int(stored["version"]) if "version" in stored.files else -1
        if (int(stored["source_size"]) == path.stat().st_size
                and version == CACHE_VERSION):
            log.info("reusing counts from %s", cache.name)
            return Counts(
                game_ids=stored["game_ids"],
                game_counts=stored["game_counts"],
                usernames=stored["usernames"],
                user_counts=stored["user_counts"],
                rating_hist=dict(zip(stored["hist_values"].tolist(),
                                     stored["hist_freqs"].tolist())),
                n_rows=int(stored["n_rows"]),
            )
        log.info("counts cache does not match the current file; rescanning")

    counts = count(path, chunk_rows)
    _save(counts, cache, path)
    return counts
