"""
Describe the raw data before modelling it.

The one number that drives every later decision is density. Collaborative
filtering on a matrix this sparse behaves very differently from the dense
MovieLens-style examples most tutorials use, and the retention table below is
what turns "pick a threshold" into an informed choice: it shows, for each
candidate cutoff, how many games and interactions survive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tabled import config
from tabled.data import scan
from tabled.data.scan import Counts

log = logging.getLogger(__name__)

PERCENTILES = (50, 75, 90, 95, 99)
GAME_CUTOFFS = (0, 25, 50, 100, 250, 500, 1000)
USER_CUTOFFS = (0, 5, 10, 20, 50, 100)


def _spread(values: np.ndarray) -> dict:
    return {
        "min": int(values.min()),
        "mean": round(float(values.mean()), 1),
        "median": float(np.median(values)),
        **{f"p{q}": float(np.percentile(values, q)) for q in PERCENTILES},
        "max": int(values.max()),
    }


def retention_table(counts: Counts) -> list[dict]:
    """
    For each (game cutoff, user cutoff) pair, how much data survives.

    Filtering games and users interacts: dropping low-support games removes
    ratings from users, which can push those users under their own cutoff.
    This estimates the joint effect in one shot rather than pretending the
    two filters are independent.
    """
    rows = []
    for g_min in GAME_CUTOFFS:
        kept_games = counts.game_counts >= g_min
        # Ratings that survive the game filter, as an upper bound on what any
        # user filter can then keep.
        surviving = int(counts.game_counts[kept_games].sum())
        for u_min in USER_CUTOFFS:
            kept_users = int((counts.user_counts >= u_min).sum())
            rows.append({
                "min_game_ratings": g_min,
                "min_user_ratings": u_min,
                "games": int(kept_games.sum()),
                "users_before_regrade": kept_users,
                "interactions_upper_bound": surviving,
                "pct_of_all_interactions": round(100 * surviving / counts.n_rows, 2),
            })
    return rows


def summarise(counts: Counts, games_csv: Path) -> dict:
    games = pd.read_csv(games_csv, low_memory=False)
    known_ids = set(games["BGGId"].astype("int64"))
    rated_ids = set(counts.game_ids.astype("int64").tolist())

    ranked = games["Rank:boardgame"] != config.RANK_SENTINEL

    return {
        "ratings": {
            "rows": counts.n_rows,
            "unique_users": counts.n_users,
            "unique_games": counts.n_games,
            "density_pct": round(100 * counts.density, 4),
            "mean_ratings_per_user": round(counts.n_rows / counts.n_users, 1),
            "value_histogram": counts.rating_hist,
        },
        "games_csv": {
            "rows": int(len(games)),
            "columns": int(games.shape[1]),
            "ranked_games": int(ranked.sum()),
            "unranked_games": int((~ranked).sum()),
            "year_range": [int(games["YearPublished"].min()),
                           int(games["YearPublished"].max())],
        },
        "join": {
            "rated_games_missing_from_games_csv": len(rated_ids - known_ids),
            "games_with_no_ratings_at_all": len(known_ids - rated_ids),
        },
        "ratings_per_user": _spread(counts.user_counts),
        "ratings_per_game": _spread(counts.game_counts),
        "users_at_least": {n: int((counts.user_counts >= n).sum())
                           for n in USER_CUTOFFS},
        "games_at_least": {n: int((counts.game_counts >= n).sum())
                           for n in GAME_CUTOFFS},
        "retention": retention_table(counts),
    }


def run(ratings_csv: Path | None = None, games_csv: Path | None = None,
        out: Path | None = None) -> dict:
    ratings_csv = ratings_csv or config.RATINGS_CSV
    games_csv = games_csv or config.GAMES_CSV
    out = out or config.PROFILE_JSON

    log.info("scanning %s", config.display(ratings_csv))
    report = summarise(scan.count(ratings_csv), games_csv)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s", config.display(out))
    return report


def print_report(report: dict) -> None:
    r = report["ratings"]
    print(f"\n{'ratings':<26}{r['rows']:>14,}")
    print(f"{'users':<26}{r['unique_users']:>14,}")
    print(f"{'games':<26}{r['unique_games']:>14,}")
    print(f"{'density':<26}{r['density_pct']:>13.4f}%")
    print(f"{'ratings per user (median)':<26}"
          f"{report['ratings_per_user']['median']:>14,.0f}")
    print(f"{'ratings per game (median)':<26}"
          f"{report['ratings_per_game']['median']:>14,.0f}")

    print(f"\n{'min/game':>9}{'min/user':>10}{'games':>9}{'users':>10}"
          f"{'interactions':>15}{'kept':>8}")
    for row in report["retention"]:
        if row["min_user_ratings"] not in (0, 10, 50):
            continue
        print(f"{row['min_game_ratings']:>9}{row['min_user_ratings']:>10}"
              f"{row['games']:>9,}{row['users_before_regrade']:>10,}"
              f"{row['interactions_upper_bound']:>15,}"
              f"{row['pct_of_all_interactions']:>7.1f}%")
    print()
