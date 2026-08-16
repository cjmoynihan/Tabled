"""
Persisting fitted models.

Fitting item-item takes minutes and ALS tens of seconds, so neither can happen
inside a web request or at the top of an evaluation run. Both reduce to a
couple of dense arrays, which is the whole reason the expensive step can be
done once offline: the app loads a few megabytes and does arithmetic.

Each artifact carries the shape it was fitted at. Loading neighbours built for
a different catalogue would not error — it would just silently recommend the
wrong games — so the check is explicit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tabled.models.item_item import ItemItem
from tabled.models.mf import ImplicitALS

log = logging.getLogger(__name__)


def split_spec(path: Path) -> dict | None:
    """
    Which users were held out when this model was fitted, if any.

    The nastiest failure mode in this project is evaluating a model against a
    split it was partly trained on: nothing errors, the scores just come out
    flattering. Recording the split alongside the weights lets the harness
    refuse rather than quietly report a wrong number.
    """
    stored = np.load(path)
    if "split_test_users" not in stored.files:
        return None
    return {
        "test_users": int(stored["split_test_users"]),
        "min_ratings": int(stored["split_min_ratings"]),
        "seed": int(stored["split_seed"]),
    }


def _spec_arrays(spec: dict | None) -> dict:
    if spec is None:
        return {}
    return {
        "split_test_users": np.int64(spec["test_users"]),
        "split_min_ratings": np.int64(spec["min_ratings"]),
        "split_seed": np.int64(spec["seed"]),
    }


def save_item_item(model: ItemItem, path: Path,
                   spec: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, idx=model._idx, sim=model._sim,
             n_games=np.int64(model.n_games),
             global_mean=np.float64(model._global_mean),
             k=np.int64(model.k), shrinkage=np.float64(model.shrinkage),
             **_spec_arrays(spec))
    log.info("wrote %s (%.1f MB)", path.name, path.stat().st_size / 1e6)


def load_item_item(path: Path, n_games: int | None = None,
                   prior_weight: float = 5.0) -> ItemItem:
    stored = np.load(path)
    fitted = int(stored["n_games"])
    if n_games is not None and fitted != n_games:
        raise ValueError(
            f"{path.name} was fitted on {fitted:,} games but the current "
            f"dataset has {n_games:,}; refit it")

    model = ItemItem(k=int(stored["k"]), shrinkage=float(stored["shrinkage"]),
                     prior_weight=prior_weight)
    return model.set_neighbours(stored["idx"], stored["sim"], fitted,
                                float(stored["global_mean"]))


def save_als(model: ImplicitALS, path: Path, spec: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, items=model.item_factors,
             alpha=np.float64(model.alpha),
             regularization=np.float64(model.regularization),
             threshold=np.float64(model.threshold),
             **_spec_arrays(spec))
    log.info("wrote %s (%.1f MB)", path.name, path.stat().st_size / 1e6)


def load_als(path: Path, n_games: int | None = None) -> ImplicitALS:
    stored = np.load(path)
    items = stored["items"]
    if n_games is not None and items.shape[0] != n_games:
        raise ValueError(
            f"{path.name} was fitted on {items.shape[0]:,} games but the "
            f"current dataset has {n_games:,}; refit it")

    model = ImplicitALS(factors=items.shape[1],
                        regularization=float(stored["regularization"]),
                        alpha=float(stored["alpha"]),
                        threshold=float(stored["threshold"]))
    return model.set_factors(items)
