"""
What `prepare` must guarantee for the models downstream.

The theme is that an index is only useful if it maps back to something real:
almost every bug in a pipeline like this is a silent misalignment between a
matrix column and the game it is supposed to mean, which produces plausible
recommendations that are simply wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabled import config
from tabled.data import prepare
from tests.conftest import (DUPLICATED, OFF_SCALE_GAME, OFF_SCALE_USERS,
                            ORPHAN_ID)


# -- shape and alignment ---------------------------------------------------

def test_artifacts_agree_on_shape(dataset):
    users, games = dataset.shape
    assert len(dataset.usernames) == users
    assert len(dataset.games) == games


def test_game_rows_are_in_column_order(dataset):
    """games.parquet row i must describe matrix column i, or every
    recommendation is mislabelled."""
    assert dataset.games["game_index"].tolist() == list(range(dataset.shape[1]))


def test_column_maps_back_to_the_right_game(dataset, raw_dir):
    """
    Spot-check the whole chain: pick a game, recount its ratings straight from
    the CSV, and confirm the matching matrix column has the same nonzeros.
    """
    raw = pd.read_csv(raw_dir / "user_ratings.csv").dropna(
        subset=["BGGId", "Rating", "Username"])
    kept_users = set(dataset.usernames.tolist())

    for col in (0, dataset.shape[1] // 2, dataset.shape[1] - 1):
        bgg_id = int(dataset.games.loc[col, "BGGId"])
        expected = raw[(raw["BGGId"] == bgg_id)
                       & (raw["Username"].isin(kept_users))]
        # Duplicated pairs collapse to one cell.
        assert dataset.matrix[:, col].nnz == expected["Username"].nunique()


def test_usernames_are_unique(dataset):
    assert len(set(dataset.usernames.tolist())) == len(dataset.usernames)


# -- filtering -------------------------------------------------------------

def test_result_satisfies_its_own_thresholds(dataset):
    """
    The point of converging the filters. Applying each cutoff once leaves rows
    and columns that no longer clear it, because the two filters interact.
    """
    t = dataset.manifest["thresholds"]
    per_user = np.diff(dataset.matrix.indptr)
    per_game = np.bincount(dataset.matrix.indices,
                           minlength=dataset.shape[1])

    assert per_user.min() >= t["min_user_ratings"]
    assert per_game.min() >= t["min_game_ratings"]


def test_convergence_actually_needed_more_than_one_round(dataset):
    """Guards the fixture, not the code: if this drops to zero the fixture has
    stopped exercising the interaction between the two filters."""
    assert dataset.manifest["prune_rounds"] >= 1


def test_games_without_metadata_are_excluded(dataset):
    assert ORPHAN_ID not in set(dataset.games["BGGId"].tolist())


# -- data integrity --------------------------------------------------------

def test_ratings_stay_on_the_scale(dataset):
    """
    Catches the duplicate-summing trap. SciPy adds duplicate coordinates, so
    a pair rated twice would land at 16 rather than 8 if it were not handled.
    """
    assert dataset.matrix.data.min() >= 1.0
    assert dataset.matrix.data.max() <= 10.0


def test_duplicate_pair_is_averaged_not_summed(dataset, raw_dir):
    bgg_id, username = DUPLICATED
    if username not in set(dataset.usernames.tolist()):
        pytest.skip("fixture user was filtered out")

    raw = pd.read_csv(raw_dir / "user_ratings.csv")
    both = raw[(raw["BGGId"] == bgg_id) & (raw["Username"] == username)]
    assert len(both) == 2, "fixture no longer contains the duplicate"

    row = int(np.where(dataset.usernames == username)[0][0])
    col = int(dataset.games.index[dataset.games["BGGId"] == bgg_id][0])
    assert dataset.matrix[row, col] == pytest.approx(both["Rating"].mean())


def test_duplicates_are_reported(dataset):
    assert dataset.manifest["duplicates_dropped"] >= 1


def test_null_ratings_never_enter_the_matrix(dataset):
    assert not np.isnan(dataset.matrix.data).any()


def test_off_scale_ratings_are_dropped(dataset, raw_dir):
    """
    Ratings of 0, 0.1 and 11 exist in the real export. The zero matters most:
    stored in a sparse matrix it reads as 'never rated', so it has to be
    excluded at the door rather than filtered by anything downstream.
    """
    assert dataset.matrix.data.min() >= config.RATING_MIN
    assert dataset.matrix.data.max() <= config.RATING_MAX

    col = dataset.games.index[dataset.games["BGGId"] == OFF_SCALE_GAME]
    assert len(col) == 1, "fixture game should have survived filtering"
    column = dataset.matrix[:, int(col[0])]

    for username in OFF_SCALE_USERS:
        where = np.where(dataset.usernames == username)[0]
        assert len(where) == 1, f"{username} should have survived filtering"
        assert column[int(where[0]), 0] == 0, (
            f"{username}'s off-scale rating reached the matrix")


def test_scan_and_build_agree_on_which_rows_are_valid(raw_dir):
    """
    The support counts must describe exactly the rows that can become cells.
    If `count` and the matrix build disagreed about validity, thresholds would
    be applied against a row set that never actually gets built.
    """
    from tabled.data import scan

    counts = scan.count(raw_dir / "user_ratings.csv", chunk_rows=250)
    raw = pd.read_csv(raw_dir / "user_ratings.csv")
    valid = raw.dropna(subset=["BGGId", "Rating", "Username"])
    valid = valid[valid["Rating"].between(config.RATING_MIN,
                                          config.RATING_MAX)]
    assert counts.n_rows == len(valid)


def test_no_stored_zeros(dataset):
    """A stored zero is indistinguishable from 'not rated' to every model that
    reads this, so it must not exist."""
    assert (dataset.matrix.data != 0).all()


# -- persistence -----------------------------------------------------------

def test_save_load_round_trip(dataset, tmp_path):
    out = tmp_path / "processed"
    prepare.save(dataset, out)
    again = prepare.load(out)

    assert again.shape == dataset.shape
    assert (again.usernames == dataset.usernames).all()
    assert np.array_equal(again.matrix.indptr, dataset.matrix.indptr)
    assert np.allclose(again.matrix.data, dataset.matrix.data)
    pd.testing.assert_frame_equal(again.games, dataset.games)


def test_load_without_artifacts_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="tabled prepare"):
        prepare.load(tmp_path / "nothing")


# -- model-facing views ----------------------------------------------------

def test_to_implicit_is_binary_and_drops_low_ratings(dataset):
    liked = prepare.to_implicit(dataset.matrix, threshold=7.5)

    assert set(np.unique(liked.data).tolist()) <= {1.0}
    assert liked.nnz == int((dataset.matrix.data >= 7.5).sum())
    assert liked.shape == dataset.matrix.shape


def test_mean_center_zeroes_each_users_average(dataset):
    centred, means = prepare.mean_center(dataset.matrix)
    counts = np.diff(centred.indptr)
    row_of = np.repeat(np.arange(centred.shape[0]), counts)
    per_user = np.bincount(row_of, weights=centred.data) / counts

    assert np.allclose(per_user, 0, atol=1e-4)
    assert means.shape == (dataset.shape[0],)
    assert means.min() >= 1.0 and means.max() <= 10.0


def test_mean_center_leaves_sparsity_alone(dataset):
    """Centring must not densify: an unrated game stays unrated, it does not
    become 'rated at minus the user's mean'."""
    centred, _ = prepare.mean_center(dataset.matrix)
    assert np.array_equal(centred.indptr, dataset.matrix.indptr)
    assert np.array_equal(centred.indices, dataset.matrix.indices)


# -- config ----------------------------------------------------------------

def test_game_columns_all_exist_in_the_real_export(dataset):
    """Every column config asks for must have survived into the artifact."""
    for column in config.GAME_COLUMNS:
        assert column in dataset.games.columns
