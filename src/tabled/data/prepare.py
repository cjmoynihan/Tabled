"""
Turn the two CSVs into the artifacts every model reads.

Output is a users x games CSR matrix of explicit 1-10 ratings, plus the game
metadata and the index maps needed to get back to real names. Deliberately
*explicit* ratings rather than pre-binarised likes: the swipe UI collects
thumbs up/down, but throwing away the 1-10 scale here would make that choice
for the models permanently. `to_implicit()` does the binarising at model time,
where it can be tuned.

Two passes over the file. The first counts support so we know what to keep;
the second builds the matrix. Both are chunked, so peak memory is set by the
surviving interactions (~200 MB) rather than the 400 MB of CSV.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tabled import config
from tabled.data import scan

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    """The prepared matrix and everything needed to interpret its indices."""

    matrix: sp.csr_matrix          # users x games, float32, explicit ratings
    games: pd.DataFrame            # one row per column of `matrix`, in order
    usernames: np.ndarray          # one entry per row of `matrix`, in order
    manifest: dict

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def game_index(self) -> pd.Series:
        """BGGId -> column index."""
        return pd.Series(np.arange(len(self.games)),
                         index=self.games["BGGId"].to_numpy())

    def subset_users(self, rows: np.ndarray) -> "Dataset":
        """
        The same catalogue, restricted to some users.

        Columns are deliberately left alone even if a subset leaves a game
        with no ratings at all. Game indices have to mean the same thing in
        the training split, the test split and the app, or nothing downstream
        can be compared; a column of zeros is a much smaller problem than a
        silently shifted index.
        """
        return Dataset(
            matrix=self.matrix[rows].tocsr(),
            games=self.games,
            usernames=self.usernames[rows],
            manifest={**self.manifest, "subset_users": int(len(self.usernames[rows]))},
        )

    def popularity(self) -> np.ndarray:
        """Number of ratings per game, as a column-aligned array."""
        return np.bincount(self.matrix.indices, minlength=self.shape[1])


@dataclass(frozen=True)
class Catalogue:
    """
    Everything the app needs to serve, without the ratings matrix.

    The matrix exists to *fit* models. At serving time it was being loaded for
    exactly one purpose — counting ratings per game — which is now a column in
    `games.parquet`. Dropping it takes the running app from 352 MB to 191 MB
    and the deployable artifacts from 157 MB to about 17 MB, which matters
    rather more, since a 146 MB file in a git repo is its own problem.

    Deliberately duck-types `Dataset` on the three things `serve` touches:
    `games`, `popularity()` and `shape[1]`. That is why the policy and
    diversification code works unchanged against either.
    """

    games: pd.DataFrame

    @property
    def shape(self) -> tuple[int, int]:
        """(users, games). Users is zero: there are none here, by design."""
        return (0, len(self.games))

    def popularity(self) -> np.ndarray:
        if "n_ratings" not in self.games.columns:
            raise KeyError(
                "games.parquet has no 'n_ratings' column — it predates "
                "matrix-free serving. Re-run `tabled prepare`.")
        return self.games["n_ratings"].to_numpy()


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def _index_map(keys: np.ndarray) -> dict:
    return {k: i for i, k in enumerate(keys.tolist())}


def _collect(ratings_csv: Path, game_pos: dict, user_pos: dict,
             chunk_rows: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second pass: keep only rows whose game *and* user survived filtering."""
    rows, cols, vals = [], [], []
    seen = 0

    for chunk in scan.chunks(ratings_csv, chunk_rows):
        chunk = scan.clean(chunk)
        seen += len(chunk)

        u = chunk["Username"].map(user_pos)
        g = chunk["BGGId"].map(game_pos)
        keep = u.notna() & g.notna()
        if not keep.any():
            continue

        rows.append(u[keep].to_numpy(dtype=np.int32))
        cols.append(g[keep].to_numpy(dtype=np.int32))
        vals.append(chunk.loc[keep, "Rating"].to_numpy(dtype=np.float32))
        log.info("  read %s rows, kept %s", f"{seen:,}",
                 f"{sum(len(r) for r in rows):,}")

    if not rows:
        empty_i = np.empty(0, dtype=np.int32)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)

    # Concatenate and release the per-chunk pieces as we go: holding the list
    # and the joined array at once doubles peak memory for no reason.
    out = []
    for parts in (rows, cols, vals):
        out.append(np.concatenate(parts))
        parts.clear()
    return out[0], out[1], out[2]


def _to_csr(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
            shape: tuple[int, int]) -> tuple[sp.csr_matrix, int]:
    """
    Assemble the matrix, averaging any repeated (user, game) pairs.

    This has to be handled explicitly. SciPy *sums* duplicate coordinates when
    it builds a CSR, so a user who appears twice for one game would silently
    end up with a rating of 16 rather than 8 — a value outside the scale, which
    would then quietly poison every similarity that game takes part in.

    Averaging is done by dividing the summed matrix by a second matrix built
    the same way with all-ones data, so the duplicate positions divide by their
    own multiplicity. The alternative — deduplicating the triples first with a
    composite `user * n_games + game` key — needs an int64 array and a sort
    over 18M elements, and that is what pushes peak memory past a few GB.
    """
    matrix = sp.coo_matrix((vals, (rows, cols)), shape=shape,
                           dtype=np.float32).tocsr()
    duplicates = len(vals) - matrix.nnz
    if duplicates:
        log.warning("averaging %s duplicate (user, game) pairs",
                    f"{duplicates:,}")
        multiplicity = sp.coo_matrix(
            (np.ones(len(vals), dtype=np.float32), (rows, cols)),
            shape=shape, dtype=np.float32).tocsr()
        matrix.data /= multiplicity.data
    matrix.sort_indices()
    return matrix, duplicates


def _converge(matrix: sp.csr_matrix, min_user: int, min_game: int,
              max_rounds: int = 50) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Find the largest submatrix where every user and game clears its threshold.

    The two filters interact: dropping rare games costs some users ratings,
    which can push those users under `min_user`, which can in turn push a
    borderline game under `min_game`. Counting the raw CSV once is therefore
    not enough — the filters have to be applied until they stop changing.

    Only the two keep-masks are computed here; the matrix is sliced once at
    the end. Each round is two `bincount` calls over the nonzeros, so the whole
    loop costs about as much as building the matrix once. Slicing the matrix
    every round instead means a CSC conversion and a full reindex per round,
    which is orders of magnitude slower at 18M nonzeros.

    Returns the surviving user and game masks and the number of rounds taken.
    """
    n_users, n_games = matrix.shape
    rows = np.repeat(np.arange(n_users, dtype=np.int32),
                     np.diff(matrix.indptr))
    cols = matrix.indices

    u_ok = np.ones(n_users, dtype=bool)
    g_ok = np.ones(n_games, dtype=bool)
    active = np.ones(matrix.nnz, dtype=bool)

    for round_no in range(1, max_rounds + 1):
        new_g = np.bincount(cols[active], minlength=n_games) >= min_game
        new_u = np.bincount(rows[active], minlength=n_users) >= min_user

        if np.array_equal(new_g, g_ok) and np.array_equal(new_u, u_ok):
            return u_ok, g_ok, round_no - 1

        u_ok, g_ok = new_u, new_g
        active = u_ok[rows] & g_ok[cols]
        log.info("  prune round %d: %s users x %s games, %s interactions",
                 round_no, f"{u_ok.sum():,}", f"{g_ok.sum():,}",
                 f"{active.sum():,}")

    raise RuntimeError("pruning did not converge; thresholds may be too high")


def build(ratings_csv: Path | None = None, games_csv: Path | None = None,
          min_game_ratings: int | None = None,
          min_user_ratings: int | None = None,
          chunk_rows: int | None = None,
          counts_cache: Path | None = None,
          rescan: bool = False) -> Dataset:
    ratings_csv = ratings_csv or config.RATINGS_CSV
    games_csv = games_csv or config.GAMES_CSV
    min_game = (config.MIN_GAME_RATINGS if min_game_ratings is None
                else min_game_ratings)
    min_user = (config.MIN_USER_RATINGS if min_user_ratings is None
                else min_user_ratings)
    started = time.perf_counter()

    log.info("pass 1/2: counting support in %s", config.display(ratings_csv))
    if counts_cache is None:
        counts = scan.count(ratings_csv, chunk_rows)
    else:
        counts = scan.count_cached(ratings_csv, counts_cache, chunk_rows,
                                   refresh=rescan)

    # The matrix may only contain games we have metadata for; a recommendation
    # we cannot name or picture is useless to the UI.
    meta = pd.read_csv(games_csv, low_memory=False)
    meta = meta[[c for c in config.GAME_COLUMNS if c in meta.columns]]
    describable = set(meta["BGGId"].astype("int64"))

    candidate_games = np.array(
        [g for g in counts.games_with_at_least(min_game).tolist()
         if g in describable], dtype=np.int64)
    candidate_users = counts.users_with_at_least(min_user)
    log.info("after count filter: %s games, %s users",
             f"{len(candidate_games):,}", f"{len(candidate_users):,}")

    log.info("pass 2/2: building the matrix")
    rows, cols, vals = _collect(ratings_csv, _index_map(candidate_games),
                                _index_map(candidate_users), chunk_rows)
    matrix, duplicates = _to_csr(
        rows, cols, vals,
        shape=(len(candidate_users), len(candidate_games)))
    del rows, cols, vals          # ~220 MB, and nothing below needs them

    u_ok, g_ok, prune_rounds = _converge(matrix, min_user, min_game)
    if not (u_ok.all() and g_ok.all()):
        matrix = matrix[u_ok][:, g_ok].tocsr()
        matrix.sort_indices()

    usernames = candidate_users[u_ok]
    kept_ids = candidate_games[g_ok]
    games = (meta.set_index("BGGId").loc[kept_ids].reset_index()
             .assign(game_index=np.arange(len(kept_ids))))

    # Popularity is stored as a column, not left to be derived. It is the only
    # thing the app ever needed the ratings matrix for, and a 17,140-element
    # array is not worth loading 146 MB to compute — writing it here is what
    # lets serving skip the matrix entirely.
    games["n_ratings"] = np.bincount(matrix.indices, minlength=matrix.shape[1])

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {
            "ratings_csv": config.display(ratings_csv),
            "games_csv": config.display(games_csv),
            "ratings_bytes": ratings_csv.stat().st_size,
        },
        "thresholds": {"min_game_ratings": min_game,
                       "min_user_ratings": min_user},
        "raw": {"rows": counts.n_rows, "users": counts.n_users,
                "games": counts.n_games,
                "density_pct": round(100 * counts.density, 4)},
        "kept": {
            "users": int(matrix.shape[0]),
            "games": int(matrix.shape[1]),
            "interactions": int(matrix.nnz),
            "pct_of_interactions": round(100 * matrix.nnz / counts.n_rows, 2),
            "density_pct": round(
                100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]), 4),
            "mean_per_user": round(matrix.nnz / matrix.shape[0], 1),
            "mean_per_game": round(matrix.nnz / matrix.shape[1], 1),
        },
        "duplicates_dropped": duplicates,
        "prune_rounds": prune_rounds,
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    log.info("built %s x %s, %s interactions in %.0fs",
             f"{matrix.shape[0]:,}", f"{matrix.shape[1]:,}",
             f"{matrix.nnz:,}", manifest["build_seconds"])

    return Dataset(matrix=matrix, games=games, usernames=usernames,
                   manifest=manifest)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save(ds: Dataset, out_dir: Path | None = None) -> None:
    """
    Write the artifacts, building them locally first and copying them in.

    Writing straight to the destination means an interrupted run leaves a
    truncated `.npz` sitting exactly where the next `load()` expects a good
    one — and a truncated matrix is not obviously broken, it is just quietly
    missing the tail of the data. Building in scratch and copying in makes the
    failure mode "no artifact" rather than "a plausible wrong one".

    Deliberately uncompressed. zlib on a 145 MB matrix costs far more time
    than writing the extra bytes, and this file is read on every app start, so
    load speed matters more than disk.
    """
    out_dir = out_dir or config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [config.RATINGS_NPZ.name, config.GAMES_PARQUET.name,
             config.USERS_NPY.name, config.MANIFEST_JSON.name]

    with tempfile.TemporaryDirectory(prefix="tabled-") as scratch:
        tmp = Path(scratch)
        sp.save_npz(tmp / config.RATINGS_NPZ.name, ds.matrix, compressed=False)
        ds.games.to_parquet(tmp / config.GAMES_PARQUET.name, index=False)
        np.save(tmp / config.USERS_NPY.name, ds.usernames, allow_pickle=True)
        (tmp / config.MANIFEST_JSON.name).write_text(
            json.dumps(ds.manifest, indent=2), encoding="utf-8")

        for name in names:
            shutil.copyfile(tmp / name, out_dir / name)
            log.info("  wrote %s (%.1f MB)", name,
                     (out_dir / name).stat().st_size / 1e6)

    log.info("wrote artifacts to %s", config.display(out_dir))


def load_catalogue(out_dir: Path | None = None) -> Catalogue:
    """Load just the game metadata — what a running app needs."""
    out_dir = out_dir or config.PROCESSED_DIR
    path = out_dir / config.GAMES_PARQUET.name
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path.name} in {config.display(out_dir)} — "
            f"run `tabled prepare` first")
    return Catalogue(games=pd.read_parquet(path))


def load(out_dir: Path | None = None) -> Dataset:
    out_dir = out_dir or config.PROCESSED_DIR
    missing = [p.name for p in (config.RATINGS_NPZ, config.GAMES_PARQUET,
                                config.USERS_NPY, config.MANIFEST_JSON)
               if not (out_dir / p.name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {', '.join(missing)} in {config.display(out_dir)} — "
            f"run `tabled prepare` first")

    return Dataset(
        matrix=sp.load_npz(out_dir / config.RATINGS_NPZ.name).tocsr(),
        games=pd.read_parquet(out_dir / config.GAMES_PARQUET.name),
        usernames=np.load(out_dir / config.USERS_NPY.name, allow_pickle=True),
        manifest=json.loads(
            (out_dir / config.MANIFEST_JSON.name).read_text(encoding="utf-8")),
    )


# --------------------------------------------------------------------------
# views the models will want
# --------------------------------------------------------------------------

def to_implicit(matrix: sp.csr_matrix,
                threshold: float | None = None) -> sp.csr_matrix:
    """
    Explicit ratings -> binary likes, dropping everything below `threshold`.

    This is what an ALS/implicit-feedback model consumes, and it is also the
    honest representation of what the swipe UI collects.
    """
    threshold = config.LIKE_THRESHOLD if threshold is None else threshold
    liked = matrix.copy()
    liked.data = (liked.data >= threshold).astype(np.float32)
    liked.eliminate_zeros()
    return liked


def mean_center(matrix: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Subtract each user's mean from their ratings.

    BGG raters differ more in generosity than in taste — some never go below
    6, others treat 5 as average. Centring removes that offset so similarity
    reflects relative preference instead of rating style.
    """
    centred = matrix.copy().astype(np.float32)
    counts = np.diff(centred.indptr)
    # bincount rather than add.reduceat: reduceat returns the wrong thing for
    # empty rows, and an empty row is legal here even if pruning avoids it.
    row_of = np.repeat(np.arange(matrix.shape[0]), counts)
    sums = np.bincount(row_of, weights=centred.data,
                       minlength=matrix.shape[0]).astype(np.float32)
    means = np.zeros(matrix.shape[0], dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    centred.data -= means[row_of]
    return centred, means
