"""
Paths and tunable constants. Bottom of the dependency graph: this module
imports nothing else from `tabled`, and everything else may import it.

Layout under `data/`:

    data/raw/         the Kaggle export, exactly as downloaded
    data/processed/   everything this project generates

Every path is env-overridable so the pipeline can run against a small
fixture set in tests without touching the real 400 MB file.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Nearest ancestor containing pyproject.toml, else the CWD."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


ROOT = project_root()

DATA_DIR = Path(os.environ.get("TABLED_DATA") or ROOT / "data")
RAW_DIR = Path(os.environ.get("TABLED_RAW") or DATA_DIR / "raw")
PROCESSED_DIR = Path(os.environ.get("TABLED_PROCESSED") or DATA_DIR / "processed")

# -- inputs (filenames as they appear in the Kaggle archive) ----------------
GAMES_CSV = RAW_DIR / "games.csv"
RATINGS_CSV = RAW_DIR / "user_ratings.csv"

# -- outputs ---------------------------------------------------------------
PROFILE_JSON = PROCESSED_DIR / "profile.json"
COUNTS_NPZ = PROCESSED_DIR / "counts.npz"          # cached support counts
RATINGS_NPZ = PROCESSED_DIR / "ratings.npz"        # CSR, users x games
GAMES_PARQUET = PROCESSED_DIR / "games.parquet"    # one row per kept game
USERS_NPY = PROCESSED_DIR / "users.npy"            # usernames by user index
MANIFEST_JSON = PROCESSED_DIR / "manifest.json"    # what was built, and how

# Fitted models. Both are expensive to build and cheap to load, so they are
# artifacts rather than something the app or the harness recomputes.
ITEM_ITEM_NPZ = PROCESSED_DIR / "item_item.npz"    # top-k neighbour lists
ALS_NPZ = PROCESSED_DIR / "als.npz"                # item latent factors

# -- ingest ----------------------------------------------------------------
# 19M rows will not fit comfortably alongside a similarity matrix, so every
# pass over user_ratings.csv is chunked. 2M rows of (int32, float32, str)
# costs roughly 250 MB while it is live.
CHUNK_ROWS = int(os.environ.get("TABLED_CHUNK", "2000000"))

# -- filtering -------------------------------------------------------------
# Collaborative filtering learns nothing from an item three people rated, and
# a user with two ratings contributes almost no co-occurrence signal while
# still costing a row. These are the defaults; `tabled prepare` overrides them.
MIN_GAME_RATINGS = int(os.environ.get("TABLED_MIN_GAME", "50"))
MIN_USER_RATINGS = int(os.environ.get("TABLED_MIN_USER", "10"))

# BGG stores "unranked" as a sentinel rank rather than a null.
RANK_SENTINEL = 21926

# The valid rating scale. The export contains a handful of values outside it
# (three at ~0, a couple below 1) out of 18.9M — statistically nothing, but a
# rating of 0 is indistinguishable from "not rated" once it is in a sparse
# matrix, so they are dropped rather than tolerated.
RATING_MIN = 1.0
RATING_MAX = 10.0

# Columns worth carrying into the app. The full games.csv has 48, most of
# which are either free text or per-family ranks the UI will never show.
GAME_COLUMNS = [
    "BGGId", "Name", "YearPublished", "GameWeight", "AvgRating",
    "BayesAvgRating", "StdDev", "MinPlayers", "MaxPlayers", "ComAgeRec",
    "MfgPlaytime", "ComMinPlaytime", "ComMaxPlaytime", "NumUserRatings",
    "NumOwned", "NumWant", "NumWish", "Family", "Kickstarted", "ImagePath",
    "Rank:boardgame",
    "Cat:Thematic", "Cat:Strategy", "Cat:War", "Cat:Family", "Cat:CGS",
    "Cat:Abstract", "Cat:Party", "Cat:Childrens",
]

# -- swipe semantics -------------------------------------------------------
# The UI collects likes and dislikes; the training data is a 1-10 scale.
# These two constants are the bridge, and they are deliberately visible
# rather than buried in the model: changing them changes what the model
# thinks a swipe means.
LIKE_THRESHOLD = float(os.environ.get("TABLED_LIKE", "7.5"))
LIKE_VALUE = 9.0
DISLIKE_VALUE = 4.0


def display(path: Path) -> str:
    """Path relative to the project root when inside it, else absolute."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
