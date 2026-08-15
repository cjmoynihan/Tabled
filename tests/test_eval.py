"""
Tests for the evaluation harness.

A scoreboard that cannot tell a good model from a bad one is worse than no
scoreboard, because it looks like evidence. So alongside the unit checks
there is a planted-signal test: a synthetic dataset with known taste clusters
and a model that exploits them, which must beat popularity by a wide margin.
If that ever stops holding, the harness has gone blind and every number it
produces afterwards is unsafe to act on.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tabled import config
from tabled.eval import metrics, simulate, split
from tabled.models import baselines
from tabled.models.base import Recommender, Swipes


# -- metrics, against hand-computed answers --------------------------------

def test_recall_counts_only_the_top_n():
    ranked = np.array([5, 1, 9, 2, 7])
    relevant = np.array([1, 2, 42])          # 42 is never recommended
    # top-3 is [5, 1, 9]; one of three relevant found
    assert metrics.recall_at_n(ranked, relevant, 3) == pytest.approx(1 / 3)
    # top-5 finds 1 and 2
    assert metrics.recall_at_n(ranked, relevant, 5) == pytest.approx(2 / 3)


def test_hit_rate_is_all_or_nothing():
    ranked = np.array([5, 1, 9])
    assert metrics.hit_rate_at_n(ranked, np.array([1]), 3) == 1.0
    assert metrics.hit_rate_at_n(ranked, np.array([99]), 3) == 0.0


def test_ndcg_rewards_hits_nearer_the_top():
    relevant = np.array([7])
    first = metrics.ndcg_at_n(np.array([7, 0, 1]), relevant, 3)
    third = metrics.ndcg_at_n(np.array([0, 1, 7]), relevant, 3)

    assert first == pytest.approx(1.0)       # perfect ranking
    assert third == pytest.approx(0.5)       # 1/log2(4) = 0.5
    assert first > third


def test_ndcg_is_one_when_every_relevant_game_is_on_top():
    ranked = np.array([3, 8, 1, 4])
    assert metrics.ndcg_at_n(ranked, np.array([3, 8]), 4) == pytest.approx(1.0)


def test_metrics_are_nan_when_there_is_nothing_to_find():
    empty = np.array([], dtype=np.int64)
    assert np.isnan(metrics.recall_at_n(np.array([1]), empty, 1))
    assert np.isnan(metrics.ndcg_at_n(np.array([1]), empty, 1))


def test_popularity_percentile_spans_the_catalogue():
    # rank 0 = most popular, rank 9 = least
    ranks = np.arange(10)
    assert metrics.popularity_percentile(np.array([0]), ranks) == pytest.approx(0)
    assert metrics.popularity_percentile(np.array([9]), ranks) == pytest.approx(100)


def test_coverage_is_a_fraction_of_the_catalogue():
    assert metrics.coverage({1, 2, 3}, 10) == pytest.approx(0.3)


# -- Swipes ----------------------------------------------------------------

def test_from_reactions_maps_the_ui_signal_onto_the_scale():
    s = Swipes.from_reactions(liked=[1, 2], disliked=[3])

    assert set(s.liked.tolist()) == {1, 2}
    assert set(s.disliked.tolist()) == {3}
    assert len(s) == 3


def test_from_reactions_rejects_contradictions():
    with pytest.raises(ValueError, match="both liked and disliked"):
        Swipes.from_reactions(liked=[1, 2], disliked=[2])


def test_swipes_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="ratings"):
        Swipes(items=np.array([1, 2]), ratings=np.array([9.0]))


def test_centred_swipes_remove_the_users_own_offset():
    generous = Swipes(items=np.array([0, 1]), ratings=np.array([9.0, 10.0]))
    harsh = Swipes(items=np.array([0, 1]), ratings=np.array([4.0, 5.0]))
    # Both users prefer game 1 to game 0 by the same margin.
    assert np.allclose(generous.centred(), harsh.centred())


# -- recommend contract ----------------------------------------------------

def test_recommend_never_returns_an_already_swiped_game(clustered):
    model = baselines.Popularity().fit(clustered)
    swipes = Swipes(items=np.arange(10), ratings=np.full(10, 8.0))

    assert not set(model.recommend(swipes, n=20)) & set(range(10))


def test_recommend_respects_exclude(clustered):
    model = baselines.Popularity().fit(clustered)
    swipes = Swipes(items=np.array([0]), ratings=np.array([8.0]))
    banned = np.arange(1, 30)

    assert not set(model.recommend(swipes, n=10, exclude=banned)) & set(banned)


def test_recommend_is_ordered_best_first(clustered):
    model = baselines.Popularity().fit(clustered)
    swipes = Swipes(items=np.array([0]), ratings=np.array([8.0]))
    ranked = model.recommend(swipes, n=15)

    scores = model.score(swipes)[ranked]
    assert (np.diff(scores) <= 0).all()


def test_recommend_rejects_a_wrongly_shaped_score(clustered):
    class Broken(Recommender):
        name = "broken"

        def fit(self, ds):
            self._remember_shape(ds)
            return self

        def score(self, swipes):
            return np.zeros(3)

    model = Broken().fit(clustered)
    with pytest.raises(ValueError, match="expected"):
        model.recommend(Swipes(np.array([0]), np.array([8.0])), n=2)


def test_random_gives_different_users_different_lists(clustered):
    model = baselines.Random(seed=1).fit(clustered)
    swipes = Swipes(items=np.array([0]), ratings=np.array([8.0]))

    assert model.recommend(swipes, 10).tolist() != model.recommend(swipes, 10).tolist()


# -- split -----------------------------------------------------------------

def test_split_is_disjoint_and_complete(clustered):
    parts = split.by_user(clustered, n_test=50, min_ratings=10, seed=0)

    assert not set(parts.train_rows) & set(parts.test_rows)
    assert len(parts.train_rows) + len(parts.test_rows) == clustered.shape[0]


def test_test_users_all_clear_the_minimum(clustered):
    parts = split.by_user(clustered, n_test=50, min_ratings=20, seed=0)
    per_user = np.diff(clustered.matrix.indptr)

    assert (per_user[parts.test_rows] >= 20).all()


def test_split_refuses_to_invent_users(clustered):
    with pytest.raises(ValueError, match="need"):
        split.by_user(clustered, n_test=10 ** 6, min_ratings=10)


def test_subset_keeps_game_indices_stable(clustered):
    """The whole comparison collapses if a column means something different
    in train than it does in test."""
    train = clustered.subset_users(np.arange(100))

    assert train.shape[1] == clustered.shape[1]
    assert train.games["BGGId"].tolist() == clustered.games["BGGId"].tolist()


# -- the harness end to end ------------------------------------------------

class CoOccurrence(Recommender):
    """
    Minimal collaborative signal: score a game by how often it was rated by
    the same people who rated the swiped games.

    Exists to prove the harness can see a real model beating a popularity
    control. It is not a serious recommender — no normalisation at all, so it
    is heavily popularity-biased itself — which makes it a conservative test:
    if even this clears popularity by a wide margin, the harness is measuring
    something.
    """

    name = "cooccurrence"

    def fit(self, ds):
        self._remember_shape(ds)
        liked = ds.matrix.copy()
        liked.data = (liked.data >= config.LIKE_THRESHOLD).astype(np.float32)
        liked.eliminate_zeros()
        self._liked = liked.tocsc()
        return self

    def score(self, swipes):
        seeds = swipes.liked
        if len(seeds) == 0:
            return np.zeros(self.n_games)
        # users who liked any seeded game, then what else those users liked
        users = self._liked[:, seeds].sum(axis=1).A.ravel() > 0
        return np.asarray(self._liked.T @ users.astype(np.float32)).ravel()


def test_harness_detects_a_model_that_beats_popularity(clustered):
    """
    The load-bearing test. On data with planted taste clusters a model that
    uses the swipes must beat one that ignores them; if this fails, every
    number the harness produces is untrustworthy.
    """
    parts = split.by_user(clustered, n_test=120, min_ratings=20, seed=0)
    train = clustered.subset_users(parts.train_rows)

    scores = {}
    for model in (baselines.Popularity(), CoOccurrence()):
        results = simulate.evaluate(model.fit(train), clustered,
                                    parts.test_rows, ks=(5,), n=10, seed=0)
        scores[model.name] = results[0].ndcg

    assert scores["cooccurrence"] > scores["popularity"] * 1.5, scores


def test_evaluation_excludes_the_revealed_swipes(clustered):
    """
    A model must not be able to score points by returning the games it was
    just handed. Popularity would do exactly that on the most-rated games if
    `recommend` were not masking them.
    """
    parts = split.by_user(clustered, n_test=60, min_ratings=20, seed=0)
    train = clustered.subset_users(parts.train_rows)
    model = baselines.Popularity().fit(train)

    matrix = clustered.matrix
    for row in parts.test_rows[:20]:
        lo, hi = matrix.indptr[row], matrix.indptr[row + 1]
        items = matrix.indices[lo:hi]
        swipes = Swipes(items=items[:5], ratings=matrix.data[lo:hi][:5])
        assert not set(model.recommend(swipes, n=10)) & set(items[:5].tolist())


def test_results_are_reproducible(clustered):
    parts = split.by_user(clustered, n_test=60, min_ratings=20, seed=0)
    train = clustered.subset_users(parts.train_rows)

    def run():
        return simulate.evaluate(baselines.Popularity().fit(train), clustered,
                                 parts.test_rows, ks=(5,), n=10, seed=7)[0]

    assert run().ndcg == run().ndcg


def test_sweeping_k_reports_one_row_per_k(clustered):
    parts = split.by_user(clustered, n_test=40, min_ratings=20, seed=0)
    train = clustered.subset_users(parts.train_rows)
    results = simulate.evaluate(baselines.Popularity().fit(train), clustered,
                                parts.test_rows, ks=(3, 5, 10), n=10)

    assert [r.k for r in results] == [3, 5, 10]
    assert all(r.users > 0 for r in results)


def test_popular_seeding_reveals_the_best_known_games(clustered):
    """The 'popular' policy exists to mimic the app showing recognisable
    cards first, so it must actually pick popular games."""
    parts = split.by_user(clustered, n_test=40, min_ratings=20, seed=0)
    train = clustered.subset_users(parts.train_rows)
    model = baselines.Popularity().fit(train)

    by_policy = {
        policy: simulate.evaluate(model, clustered, parts.test_rows,
                                  ks=(5,), n=10, policy=policy)[0]
        for policy in simulate.SEED_POLICIES
    }
    # Revealing the most popular games leaves a less popular remainder to
    # find, so a popularity model should do no better under that policy.
    assert by_policy["popular"].ndcg <= by_policy["random"].ndcg


def test_unknown_seed_policy_is_rejected(clustered):
    parts = split.by_user(clustered, n_test=20, min_ratings=20, seed=0)
    model = baselines.Popularity().fit(clustered.subset_users(parts.train_rows))

    with pytest.raises(ValueError, match="unknown seed policy"):
        simulate.evaluate(model, clustered, parts.test_rows, ks=(5,),
                          policy="nonsense")
