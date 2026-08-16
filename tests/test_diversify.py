"""
Tests for near-duplicate filtering.

The motivating case is concrete: liking The Red Dragon Inn returned Red Dragon
Inn 2, 3, 4 and 5. Every one is a correct prediction and the list is still
useless. The tests below pin the distinction that makes the fix work — series
entries are capped, genuinely different games in the same franchise are not —
plus the real-data check that a similarity threshold could not have done it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabled import config
from tabled.serve import diversify


@pytest.fixture
def catalogue():
    """A small catalogue with one troublesome series in it."""
    return pd.DataFrame({
        "game_index": range(8),
        "BGGId": range(100, 108),
        "Name": ["Red Dragon Inn", "Red Dragon Inn 2", "Red Dragon Inn 3",
                 "Codenames", "Codenames: Duet", "Catan", "Azul", "Gloom"],
        "Family": ["Red Dragon Inn", "Red Dragon Inn", "Red Dragon Inn",
                   "Codenames", "Codenames", "Catan", None, None],
    })


def test_a_series_gets_one_slot(catalogue):
    picks = diversify.diversify(np.arange(8), catalogue, n=5)
    names = catalogue.loc[picks, "Name"].tolist()

    assert names.count("Red Dragon Inn") == 1
    assert "Red Dragon Inn 2" not in names
    assert "Red Dragon Inn 3" not in names


def test_the_best_member_of_a_series_survives(catalogue):
    """Thinning, not reordering: whichever ranked highest is the one kept."""
    ranked = np.array([2, 1, 0, 3, 5, 6, 7, 4])   # RDI 3 ranked first
    picks = diversify.diversify(ranked, catalogue, n=5)

    assert catalogue.loc[picks[0], "Name"] == "Red Dragon Inn 3"


def test_ranking_order_is_otherwise_preserved(catalogue):
    picks = diversify.diversify(np.arange(8), catalogue, n=6)
    assert picks.tolist() == sorted(picks.tolist())


def test_relaxing_the_cap_lets_more_of_a_series_through(catalogue):
    picks = diversify.diversify(np.arange(8), catalogue, n=6,
                                max_per_family=2)
    names = catalogue.loc[picks, "Name"].tolist()

    assert names.count("Red Dragon Inn") + names.count("Red Dragon Inn 2") == 2


def test_unfamilied_games_do_not_all_count_as_one_series(catalogue):
    """Azul and Gloom share no family, and must not compete for one slot."""
    picks = diversify.diversify(np.array([6, 7]), catalogue, n=5)
    assert len(picks) == 2


def test_asking_for_more_than_survives_returns_what_there_is(catalogue):
    picks = diversify.diversify(np.arange(8), catalogue, n=99)
    assert len(picks) == 5      # 3 families collapse to 1 each, plus 2 loners


def test_similarity_cap_catches_twins_the_family_column_missed(catalogue):
    class Twins:
        """Azul and Gloom have no family, but claim to be near-identical."""

        def similarity_to(self, a, b):
            return 0.9 if {a, b} == {6, 7} else 0.0

    picks = diversify.diversify(np.array([6, 7]), catalogue, n=5, model=Twins())
    assert len(picks) == 1


def test_a_model_without_similarity_is_fine(catalogue):
    """Baselines expose no similarity; diversification should still run."""
    picks = diversify.diversify(np.arange(8), catalogue, n=5, model=object())
    assert len(picks) == 5


def test_family_reaction_counts(catalogue):
    counts = diversify.family_reaction_counts(catalogue, np.array([0, 1, 3]))
    assert counts["Red Dragon Inn"] == 2
    assert counts["Codenames"] == 1


# -- the reason a threshold was not used -----------------------------------

@pytest.mark.skipif(not config.ITEM_ITEM_NPZ.exists(),
                    reason="needs a fitted item-item model")
def test_no_similarity_threshold_separates_these_cases():
    """
    The empirical justification for using `Family` at all.

    On the real data, near-duplicates and genuinely different games in the
    same franchise interleave by similarity, so no cutoff can split them. If
    this ever stops being true the simpler approach becomes available, so the
    assertion is worth keeping honest rather than assuming.
    """
    from tabled.data import prepare
    from tabled.models import store

    ds = prepare.load()
    model = store.load_item_item(config.ITEM_ITEM_NPZ, ds.shape[1])
    pos = {n: i for i, n in enumerate(ds.games["Name"].to_numpy())}

    duplicate = model.similarity_to(pos["The Red Dragon Inn"],
                                    pos["The Red Dragon Inn 2"])
    distinct = model.similarity_to(pos["Codenames"], pos["Codenames: Duet"])

    assert duplicate > 0 and distinct > 0
    # Both are "similar"; the one we want to cut is not reliably far enough
    # above the one we want to keep for a threshold to be safe.
    assert duplicate / distinct < 2.0, (
        f"duplicate {duplicate:.3f} vs distinct {distinct:.3f} — these have "
        f"separated, so a similarity cutoff may now be viable")
