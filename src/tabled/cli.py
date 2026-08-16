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


def cmd_fit(args) -> int:
    """
    Fit an expensive model once and cache it.

    Deliberately separate from `evaluate`. Item-item takes minutes, and
    rebuilding it for every sweep of k or every change of policy would make
    experimenting painful enough that it stops happening.
    """
    from tabled.data import prepare
    from tabled.eval import split
    from tabled.models import store
    from tabled.models.item_item import ItemItem
    from tabled.models.mf import ImplicitALS

    ds = prepare.load(args.out)
    spec = None
    if args.holdout:
        spec = {"test_users": args.test_users,
                "min_ratings": args.min_ratings, "seed": args.seed}
        parts = split.by_user(ds, **{"n_test": args.test_users,
                                     "min_ratings": args.min_ratings,
                                     "seed": args.seed})
        ds_fit = ds.subset_users(parts.train_rows)
        print(f"fitting on {ds_fit.shape[0]:,} training users "
              f"({args.test_users:,} held out)")
    else:
        ds_fit = ds
        print(f"fitting on all {ds_fit.shape[0]:,} users")

    if args.model == "item-item":
        model = ItemItem(k=args.k, shrinkage=args.shrinkage).fit(ds_fit)
        store.save_item_item(model, args.path or config.ITEM_ITEM_NPZ, spec)
    else:
        model = ImplicitALS(factors=args.factors, iterations=args.iterations,
                            regularization=args.regularization,
                            alpha=args.alpha, seed=args.seed).fit(ds_fit)
        store.save_als(model, args.path or config.ALS_NPZ, spec)
    return 0


def _check_split(path, args) -> None:
    """
    Refuse to score a cached model against a split it did not hold out.

    Silent leakage is the failure this guards. A model fitted with 5,000 users
    held out, then evaluated against a differently-seeded 5,000, is being
    tested largely on users it trained on — and the only symptom is that the
    numbers look good.
    """
    from tabled.models import store

    spec = store.split_spec(path)
    if spec is None:
        print(f"warning: {path.name} records no holdout — it was fitted on "
              f"every user, so these scores are optimistic and not "
              f"comparable to the baselines")
        return

    current = {"test_users": args.test_users,
               "min_ratings": args.min_ratings, "seed": args.seed}
    if spec != current:
        raise SystemExit(
            f"{path.name} held out {spec}, but this run is evaluating against "
            f"{current}. Those test users overlap its training data — refit "
            f"with matching --test-users/--min-ratings/--seed, or evaluate "
            f"with the values it was fitted for.")


def _load_model(name: str, train, args):
    """Baselines are cheap to fit here; the real models come off disk."""
    from tabled.models import baselines, store

    if name in baselines.ALL:
        return baselines.ALL[name]().fit(train)
    if name == "item-item":
        _check_split(config.ITEM_ITEM_NPZ, args)
        return store.load_item_item(config.ITEM_ITEM_NPZ, train.shape[1])
    if name == "als":
        _check_split(config.ALS_NPZ, args)
        return store.load_als(config.ALS_NPZ, train.shape[1])
    raise KeyError(name)


def cmd_evaluate(args) -> int:
    import json

    from tabled.data import prepare
    from tabled.eval import simulate, split
    from tabled.models import baselines

    ds = prepare.load(args.out)
    parts = split.by_user(ds, n_test=args.test_users,
                          min_ratings=args.min_ratings, seed=args.seed)
    train = ds.subset_users(parts.train_rows)

    known = set(baselines.ALL) | {"item-item", "als"}
    wanted = [m.strip() for m in args.models.split(",")]
    unknown = [m for m in wanted if m not in known]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}")
        print(f"available: {', '.join(sorted(known))}")
        return 1

    ks = tuple(int(k) for k in args.ks.split(","))
    results = []
    for name in wanted:
        try:
            model = _load_model(name, train, args)
        except FileNotFoundError:
            print(f"{name} has not been fitted yet — run `tabled fit "
                  f"--model {name} --holdout` first")
            return 1
        results += simulate.evaluate(
            model, ds, parts.test_rows, ks=ks, n=args.n,
            policy=args.policy, seed=args.seed)

    print(f"\n{len(parts.test_rows):,} held-out users, top-{args.n}, "
          f"'{args.policy}' swipe seeding\n")
    print(simulate.format_table(results))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps([r.as_dict() for r in results],
                                        indent=2), encoding="utf-8")
        print(f"wrote {config.display(args.save)}\n")
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

    s = sub.add_parser("fit", help="fit an expensive model and cache it")
    s.add_argument("--model", required=True, choices=("item-item", "als"))
    s.add_argument("--out", type=Path, default=None, help="artifact directory")
    s.add_argument("--path", type=Path, default=None, help="where to write it")
    s.add_argument("--holdout", action="store_true",
                   help="fit on the training split only, for honest evaluation")
    s.add_argument("--test-users", type=int, default=5000)
    s.add_argument("--min-ratings", type=int, default=25)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--k", type=int, default=100, help="neighbours per game")
    s.add_argument("--shrinkage", type=float, default=50.0)
    s.add_argument("--factors", type=int, default=64)
    s.add_argument("--iterations", type=int, default=15)
    s.add_argument("--regularization", type=float, default=0.05)
    s.add_argument("--alpha", type=float, default=40.0)
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("evaluate", help="score models on held-out users")
    s.add_argument("--out", type=Path, default=None, help="artifact directory")
    s.add_argument("--models", default="popularity,bayes,random",
                   help="comma-separated model names")
    s.add_argument("--ks", default="3,5,10,20",
                   help="comma-separated swipe counts to sweep")
    s.add_argument("--n", type=int, default=20, help="length of the top-N list")
    s.add_argument("--policy", default="random",
                   choices=simulate_policies(),
                   help="how the revealed swipes are chosen")
    s.add_argument("--test-users", type=int, default=5000)
    s.add_argument("--min-ratings", type=int, default=25,
                   help="minimum ratings for a user to be eligible for testing")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--save", type=Path, default=None,
                   help="also write the results as JSON")
    s.set_defaults(func=cmd_evaluate)

    return p


def simulate_policies() -> tuple[str, ...]:
    from tabled.eval.simulate import SEED_POLICIES

    return SEED_POLICIES


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
