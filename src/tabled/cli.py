"""
Single entry point, installed as the `tabled` command.

    tabled profile              describe the raw CSVs
    tabled prepare              build the model-ready artifacts
    tabled info                 show what is currently built

This module is the only one allowed to import from every subpackage; the
subpackages do not import each other sideways.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from tabled import config

log = logging.getLogger("tabled")


def cmd_profile(args) -> int:
    from tabled.data import profile

    report = profile.run(args.ratings, args.games, args.out)
    profile.print_report(report)
    print(f"full report: {config.display(args.out or config.PROFILE_JSON)}\n")
    return 0


def cmd_prepare(args) -> int:
    from tabled.data import prepare

    ds = prepare.build(
        ratings_csv=args.ratings,
        games_csv=args.games,
        min_game_ratings=args.min_game,
        min_user_ratings=args.min_user,
        counts_cache=args.counts_cache or config.COUNTS_NPZ,
        rescan=args.rescan,
    )
    prepare.save(ds, args.out)
    _print_manifest(ds.manifest)
    return 0


def cmd_info(args) -> int:
    from tabled.data import prepare

    try:
        ds = prepare.load(args.out)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    _print_manifest(ds.manifest)
    print("most-rated games in the matrix:")
    top = ds.games.nlargest(5, "NumUserRatings")
    for _, row in top.iterrows():
        print(f"  {row['Name'][:44]:<46}{row['NumUserRatings']:>9,} ratings")
    print()
    return 0


def _print_manifest(manifest: dict) -> None:
    raw, kept = manifest["raw"], manifest["kept"]
    t = manifest["thresholds"]
    print(f"\nthresholds   >= {t['min_game_ratings']} ratings/game, "
          f">= {t['min_user_ratings']} ratings/user\n")
    print(f"{'':<16}{'raw':>14}{'kept':>14}")
    print(f"{'users':<16}{raw['users']:>14,}{kept['users']:>14,}")
    print(f"{'games':<16}{raw['games']:>14,}{kept['games']:>14,}")
    print(f"{'interactions':<16}{raw['rows']:>14,}{kept['interactions']:>14,}")
    print(f"{'density':<16}{raw['density_pct']:>13.4f}%"
          f"{kept['density_pct']:>13.4f}%")
    print(f"\nkept {kept['pct_of_interactions']}% of all ratings; "
          f"{kept['mean_per_user']} per user, {kept['mean_per_game']} per game")
    print(f"built in {manifest['build_seconds']}s\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tabled", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="per-chunk progress while scanning")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("profile", help="describe the raw CSVs")
    s.add_argument("--ratings", type=Path, default=None)
    s.add_argument("--games", type=Path, default=None)
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(func=cmd_profile)

    s = sub.add_parser("prepare", help="build the model-ready artifacts")
    s.add_argument("--ratings", type=Path, default=None)
    s.add_argument("--games", type=Path, default=None)
    s.add_argument("--out", type=Path, default=None, help="output directory")
    s.add_argument("--min-game", type=int, default=None,
                   help=f"minimum ratings per game "
                        f"(default {config.MIN_GAME_RATINGS})")
    s.add_argument("--min-user", type=int, default=None,
                   help=f"minimum ratings per user "
                        f"(default {config.MIN_USER_RATINGS})")
    s.add_argument("--counts-cache", type=Path, default=None,
                   help="where to keep the cached support counts")
    s.add_argument("--rescan", action="store_true",
                   help="ignore the counts cache and re-read the CSV")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("info", help="show what is currently built")
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(func=cmd_info)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
