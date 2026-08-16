"""
A miniature stand-in for the Kaggle export.

Small enough to build in milliseconds, but shaped like the real thing: a
long-tailed popularity distribution, a duplicated rating, a null, and a rated
game with no metadata row. Every awkward case the real file contains has a
representative here, so the tests exercise the same branches the 400 MB file
does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Tuned so the support filters actually cascade: with 250 games drawn on a
# Zipf-ish curve, the tail games have single-digit rater counts, so dropping
# them genuinely pushes users under the user threshold. An earlier fixture had
# 80 games and no game rarer than 60 raters, which meant no threshold could
# ever trigger the interaction the pruning loop exists to handle.
N_GAMES = 250
N_USERS = 600
MAX_PICKS = 25
ORPHAN_ID = 9999           # rated, but absent from games.csv
DUPLICATED = (1, "user0")  # rated twice by the same user

# Off-scale ratings go to dedicated users who rate nothing else on that game,
# so "this cell must be empty" is an unambiguous assertion. Reusing a general
# user would not work: they may legitimately rate the same game too.
OFF_SCALE_GAME = 2
OFF_SCALE_USERS = ("offscale0", "offscale1", "offscale2")
OFF_SCALE_VALUES = (0.0, 0.1, 11.0)

# Chosen with the same shape as the real run, which converged in 4 rounds.
MIN_GAME_RATINGS = 15
MIN_USER_RATINGS = 8


@pytest.fixture
def raw_dir(tmp_path):
    """Write a games.csv / user_ratings.csv pair and return their directory."""
    rng = np.random.default_rng(20260730)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)

    # Long-tailed popularity: low-numbered games get many more raters, which
    # is what makes the support filters meaningful.
    weights = 1.0 / np.arange(1, N_GAMES + 1)
    weights /= weights.sum()

    rows = []
    # user0 rates the 20 most popular games outright, so it is guaranteed to
    # survive filtering and the duplicate-handling test can rely on it.
    for g in range(20):
        rows.append((g + 1, float(rng.integers(2, 21)) / 2, "user0"))

    for u in range(1, N_USERS):
        n = int(rng.integers(1, MAX_PICKS))
        picks = rng.choice(N_GAMES, size=n, replace=False, p=weights)
        for g in picks:
            rows.append((int(g) + 1, float(rng.integers(2, 21)) / 2, f"user{u}"))

    rows.append((DUPLICATED[0], 2.0, DUPLICATED[1]))   # duplicate pair
    rows.append((ORPHAN_ID, 7.0, "user0"))             # no metadata
    rows.append((3, np.nan, "user1"))                  # null rating

    # Off-scale ratings, as found in the real export. The zero is the
    # dangerous one: written into a sparse matrix it is indistinguishable
    # from never having rated the game at all. Each of these users rates a
    # block of other popular games so they survive filtering, and pointedly
    # does not rate OFF_SCALE_GAME anywhere else.
    for username, value in zip(OFF_SCALE_USERS, OFF_SCALE_VALUES):
        rows.append((OFF_SCALE_GAME, value, username))
        for g in range(3, 23):
            rows.append((g, 7.0, username))

    pd.DataFrame(rows, columns=["BGGId", "Rating", "Username"]).to_csv(
        raw / "user_ratings.csv", index=False)

    ids = np.arange(1, N_GAMES + 1)
    pd.DataFrame({
        "BGGId": ids,
        "Name": [f"Game {i}" for i in ids],
        "YearPublished": rng.integers(1990, 2022, N_GAMES),
        "GameWeight": rng.random(N_GAMES) * 5,
        "AvgRating": rng.random(N_GAMES) * 10,
        "BayesAvgRating": rng.random(N_GAMES) * 10,
        "StdDev": rng.random(N_GAMES),
        "MinPlayers": 2, "MaxPlayers": 4, "ComAgeRec": 10.0,
        "MfgPlaytime": 60, "ComMinPlaytime": 30, "ComMaxPlaytime": 90,
        "NumUserRatings": rng.integers(10, 9000, N_GAMES),
        "NumOwned": 100, "NumWant": 5, "NumWish": 9,
        "Family": "", "Kickstarted": 0, "ImagePath": "",
        "Rank:boardgame": ids,
        "Cat:Thematic": 0, "Cat:Strategy": 1, "Cat:War": 0, "Cat:Family": 0,
        "Cat:CGS": 0, "Cat:Abstract": 0, "Cat:Party": 0, "Cat:Childrens": 0,
    }).to_csv(raw / "games.csv", index=False)

    return raw


@pytest.fixture
def clustered():
    """
    A synthetic Dataset with planted taste clusters and a popular core.

    Built in memory rather than through the CSV pipeline: these tests are
    about the harness, not about parsing. The structure is deliberate —
    a block of universally-rated games so the popularity baseline has real
    signal to work with, and disjoint per-cluster blocks so a model that uses
    the swipes has something popularity cannot see.
    """
    import pandas as pd
    import scipy.sparse as sp

    from tabled.data.prepare import Dataset

    rng = np.random.default_rng(7)
    n_clusters, per_cluster, block = 4, 150, 40
    core = 20                                   # games everybody rates
    n_games = core + n_clusters * block
    n_users = n_clusters * per_cluster

    rows, cols, vals = [], [], []
    for u in range(n_users):
        c = u // per_cluster
        # Everyone rates the core, with middling enthusiasm.
        for g in rng.choice(core, size=12, replace=False):
            rows.append(u); cols.append(int(g))
            vals.append(float(rng.integers(5, 9)))
        # Their own cluster's block, enthusiastically.
        start = core + c * block
        for g in rng.choice(block, size=18, replace=False):
            rows.append(u); cols.append(start + int(g))
            vals.append(float(rng.integers(8, 11)))

    matrix = sp.coo_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows), np.array(cols))),
        shape=(n_users, n_games), dtype=np.float32).tocsr()
    matrix.sort_indices()

    popularity = np.bincount(matrix.indices, minlength=n_games)
    games = pd.DataFrame({
        "BGGId": np.arange(1, n_games + 1),
        "Name": [f"Game {i}" for i in range(n_games)],
        "BayesAvgRating": rng.random(n_games) * 3 + 5.5,
        "AvgRating": rng.random(n_games) * 3 + 5.5,
        "NumUserRatings": popularity,
        # Spread across the full 1-5 weight range and a realistic spread of
        # disagreement, so the cold-start policy has something to balance.
        "GameWeight": rng.uniform(1.0, 5.0, n_games),
        "StdDev": rng.uniform(0.8, 2.2, n_games),
        "MinPlayers": 2, "MaxPlayers": 4,
        "YearPublished": rng.integers(1990, 2022, n_games),
        "ComMinPlaytime": 30, "ComMaxPlaytime": 90,
        "game_index": np.arange(n_games),
    })

    return Dataset(
        matrix=matrix, games=games,
        usernames=np.array([f"u{i}" for i in range(n_users)], dtype=object),
        manifest={"thresholds": {"min_game_ratings": 1, "min_user_ratings": 1}},
    )


@pytest.fixture
def dataset(raw_dir):
    """A built Dataset over the fixture, with deliberately low thresholds."""
    from tabled.data import prepare

    return prepare.build(
        ratings_csv=raw_dir / "user_ratings.csv",
        games_csv=raw_dir / "games.csv",
        min_game_ratings=MIN_GAME_RATINGS,
        min_user_ratings=MIN_USER_RATINGS,
        chunk_rows=250,          # forces several chunks, so cross-chunk
    )                            # accumulation is actually exercised
