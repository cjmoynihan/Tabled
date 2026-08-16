"""
Tests for the two real recommenders.

The bar is behavioural rather than numerical: on data with planted taste
clusters, each model must recover the cluster structure and beat the
popularity control. Asserting exact similarity values would only pin down the
current implementation, and these are the properties that actually have to
hold for the app to work.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabled import config
from tabled.eval import simulate, split
from tabled.models import baselines, store
from tabled.models.base import Swipes
from tabled.models.item_item import ItemItem, neighbours
from tabled.models.mf import ImplicitALS

implicit_installed = pytest.importorskip


CORE = 20        # matches the `clustered` fixture: games everybody rates
BLOCK = 40       # per-cluster block size


def cluster_of(game: int) -> int:
    """Which planted cluster a game index belongs to (-1 for the core)."""
    return -1 if game < CORE else (game - CORE) // BLOCK


# -- item-item -------------------------------------------------------------

def test_neighbours_recover_the_planted_clusters(clustered):
    idx, sim = neighbours(clustered.matrix, k=10, shrinkage=1.0, block=64)

    # For games inside a cluster block, most neighbours should be from the
    # same block — that is the only structure in the data.
    same, total = 0, 0
    for game in range(CORE, clustered.shape[1]):
        for neighbour in idx[game]:
            if neighbour >= CORE:
                same += cluster_of(neighbour) == cluster_of(game)
                total += 1

    assert same / total > 0.9, f"only {same / total:.0%} of neighbours match"


def test_a_game_is_never_its_own_neighbour(clustered):
    idx, _ = neighbours(clustered.matrix, k=5, shrinkage=1.0, block=64)
    for game in range(clustered.shape[1]):
        assert game not in idx[game]


def test_similarities_are_ordered_and_non_negative(clustered):
    _, sim = neighbours(clustered.matrix, k=10, shrinkage=1.0, block=64)

    assert (sim >= 0).all()
    assert (np.diff(sim, axis=1) <= 1e-6).all()


def test_shrinkage_penalises_thinly_co_rated_pairs(clustered):
    """The whole point of shrinkage: with a heavier prior, similarities built
    on few shared raters must fall further than well-evidenced ones."""
    _, light = neighbours(clustered.matrix, k=10, shrinkage=1.0, block=64)
    _, heavy = neighbours(clustered.matrix, k=10, shrinkage=500.0, block=64)

    assert heavy.mean() < light.mean()


def test_block_range_matches_a_full_pass(clustered):
    """Sharding a fit across processes must not change the answer."""
    whole_idx, whole_sim = neighbours(clustered.matrix, k=8, block=64)

    part_idx, part_sim = neighbours(clustered.matrix, k=8, block=64,
                                    block_range=(64, 128))
    assert np.array_equal(part_idx[64:128], whole_idx[64:128])
    assert np.allclose(part_sim[64:128], whole_sim[64:128])
    # Everything outside the range is left obviously unfilled.
    assert (part_idx[:64] == -1).all()


def test_item_item_scores_the_users_cluster_highest(clustered):
    model = ItemItem(k=20, shrinkage=1.0).fit(clustered)
    seeds = np.arange(CORE, CORE + 5)              # cluster 0
    ranked = model.recommend(Swipes(seeds, np.full(5, 9.0)), n=10)

    assert (np.array([cluster_of(g) for g in ranked]) == 0).mean() > 0.7


def test_dislikes_push_their_neighbourhood_down(clustered):
    """A thumbs-down has to be usable evidence, not just a missing like."""
    model = ItemItem(k=20, shrinkage=1.0).fit(clustered)
    target = CORE + 7                              # a cluster-0 game

    liked = model.score(Swipes.from_reactions(liked=[CORE + 1]))[target]
    disliked = model.score(Swipes.from_reactions(disliked=[CORE + 1]))[target]

    assert disliked < liked


def test_no_swipes_gives_no_opinion(clustered):
    model = ItemItem(k=10).fit(clustered)
    empty = Swipes(np.array([], dtype=int), np.array([]))

    assert not model.score(empty).any()


def test_single_swipe_still_carries_signal(clustered):
    """
    Centring on the user's raw mean is degenerate at one swipe — it zeroes the
    only rating there is. The shrunk prior is what keeps this working.
    """
    model = ItemItem(k=20, shrinkage=1.0, prior_weight=5.0).fit(clustered)
    scores = model.score(Swipes(np.array([CORE + 1]), np.array([10.0])))

    assert scores.any(), "one swipe produced no signal at all"


# -- ALS -------------------------------------------------------------------

@pytest.fixture
def als(clustered):
    pytest.importorskip("implicit")
    return ImplicitALS(factors=16, iterations=12, seed=0).fit(clustered)


def test_als_recovers_the_planted_clusters(als):
    seeds = np.arange(CORE, CORE + 5)
    ranked = als.recommend(Swipes(seeds, np.full(5, 9.0)), n=10)

    assert (np.array([cluster_of(g) for g in ranked]) == 0).mean() > 0.7


def test_fold_in_needs_no_refit(als):
    """The property the whole app depends on: a brand-new user gets a vector
    from a small solve, not from retraining."""
    before = als.item_factors.copy()
    als.fold_in(Swipes.from_reactions(liked=[CORE + 1, CORE + 2]))

    assert np.array_equal(als.item_factors, before)


def test_fold_in_separates_the_clusters(als):
    a = als.fold_in(Swipes.from_reactions(liked=list(range(CORE, CORE + 6))))
    b = als.fold_in(Swipes.from_reactions(
        liked=list(range(CORE + BLOCK, CORE + BLOCK + 6))))

    assert a @ b < min(a @ a, b @ b), "different tastes gave similar vectors"


def test_fold_in_of_nothing_is_the_zero_vector(als):
    assert not als.fold_in(Swipes(np.array([], dtype=int), np.array([]))).any()


def test_als_uses_dislikes(als):
    """Implicit-feedback models normally ignore negatives; this one should
    not, because a swipe-down is observed evidence rather than absence."""
    target = CORE + 7
    liked = als.score(Swipes.from_reactions(liked=[CORE + 1]))[target]
    both = als.score(Swipes.from_reactions(
        liked=[CORE + 1], disliked=[CORE + 2, CORE + 3]))[target]

    assert both != liked


# -- persistence -----------------------------------------------------------

def test_item_item_round_trip(clustered, tmp_path):
    model = ItemItem(k=10, shrinkage=3.0).fit(clustered)
    path = tmp_path / "ii.npz"
    store.save_item_item(model, path, {"test_users": 5, "min_ratings": 2,
                                       "seed": 1})
    loaded = store.load_item_item(path, clustered.shape[1])

    swipes = Swipes.from_reactions(liked=[CORE + 1, CORE + 2])
    assert np.allclose(model.score(swipes), loaded.score(swipes))
    assert store.split_spec(path) == {"test_users": 5, "min_ratings": 2,
                                      "seed": 1}


def test_als_round_trip(als, tmp_path):
    path = tmp_path / "als.npz"
    store.save_als(als, path)
    loaded = store.load_als(path, als.n_games)

    swipes = Swipes.from_reactions(liked=[CORE + 1, CORE + 2])
    assert np.allclose(als.score(swipes), loaded.score(swipes))
    assert store.split_spec(path) is None


def test_loading_a_model_for_the_wrong_catalogue_is_refused(clustered, tmp_path):
    """Neighbour indices are meaningless against a different catalogue, and
    would misrecommend rather than crash."""
    path = tmp_path / "ii.npz"
    store.save_item_item(ItemItem(k=5).fit(clustered), path)

    with pytest.raises(ValueError, match="refit"):
        store.load_item_item(path, n_games=clustered.shape[1] + 1)


# -- against the harness ---------------------------------------------------

@pytest.mark.parametrize("build", [
    lambda: ItemItem(k=20, shrinkage=1.0),
    lambda: ImplicitALS(factors=16, iterations=12, seed=0),
], ids=["item-item", "als"])
def test_both_models_beat_popularity_on_clustered_data(clustered, build):
    pytest.importorskip("implicit")
    parts = split.by_user(clustered, n_test=120, min_ratings=20, seed=0)
    train = clustered.subset_users(parts.train_rows)

    def ndcg(model):
        return simulate.evaluate(model.fit(train), clustered, parts.test_rows,
                                 ks=(5,), n=10, seed=0)[0].ndcg

    assert ndcg(build()) > ndcg(baselines.Popularity()) * 1.5


def test_personalised_models_cover_more_of_the_catalogue(clustered):
    """Accuracy alone would hide the failure mode this app cares about."""
    parts = split.by_user(clustered, n_test=120, min_ratings=20, seed=0)
    train = clustered.subset_users(parts.train_rows)

    def coverage(model):
        return simulate.evaluate(model.fit(train), clustered, parts.test_rows,
                                 ks=(5,), n=10, seed=0)[0].coverage

    assert coverage(ItemItem(k=20, shrinkage=1.0)) > coverage(baselines.Popularity())
