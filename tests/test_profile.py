"""
Tests for the written taste description.

The risk with a feature like this is not that it crashes, it is that it says
something confidently wrong about the user from two data points. So the tests
are mostly about restraint: stay quiet early, describe only what was actually
said, and never claim more than the evidence supports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabled.serve import profile


@pytest.fixture
def games():
    """Twelve heavy strategy games and twelve light party games."""
    heavy = pd.DataFrame({
        "Name": [f"Heavy {i}" for i in range(12)],
        "GameWeight": 4.2, "MinPlayers": 2, "MaxPlayers": 4,
        "ComMaxPlaytime": 150,
        "Cat:Strategy": 1, "Cat:Party": 0, "Cat:Thematic": 0,
    })
    light = pd.DataFrame({
        "Name": [f"Light {i}" for i in range(12)],
        "GameWeight": 1.2, "MinPlayers": 4, "MaxPlayers": 8,
        "ComMaxPlaytime": 30,
        "Cat:Strategy": 0, "Cat:Party": 1, "Cat:Thematic": 0,
    })
    return pd.concat([heavy, light], ignore_index=True)


def test_it_says_nothing_from_too_little_evidence(games):
    """Being told something wrong about yourself is worse than being told
    nothing, and two likes is not a taste."""
    assert profile.summarise(games, np.array([0, 1])) is None
    assert profile.sentence(None) == ""


def test_it_speaks_up_once_there_is_enough(games):
    assert profile.summarise(games, np.array([0, 1, 2])) is not None


def test_it_describes_a_heavy_strategy_player(games):
    said = profile.sentence(profile.summarise(games, np.arange(0, 6)))

    assert "meaty" in said or "heavy" in said
    assert "strategy" in said
    assert "party" not in said


def test_it_describes_a_party_player(games):
    said = profile.sentence(profile.summarise(games, np.arange(12, 18)))

    assert "light" in said
    assert "party" in said
    assert "strategy" not in said


def test_a_category_needs_to_be_most_of_what_they_liked(games):
    """One wargame among ten euros is not a taste for wargames."""
    mixed = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12])   # 9 heavy, 1 light
    summary = profile.summarise(games, mixed)

    assert "party" not in summary["categories"]
    assert "strategy" in summary["categories"]


def test_weight_bands_are_ordered():
    phrases = [profile.weight_phrase(w) for w in (1.0, 2.0, 3.0, 3.5, 4.5)]
    assert len(set(phrases)) == len(phrases), f"bands collapsed: {phrases}"


def test_the_sentence_reads_as_a_sentence(games):
    said = profile.sentence(profile.summarise(games, np.arange(0, 6)))

    assert said.startswith("You lean towards")
    assert said.endswith(".")
    assert ", ," not in said and "  " not in said


def test_it_survives_a_catalogue_without_category_columns():
    bare = pd.DataFrame({"GameWeight": [3.0] * 5, "MinPlayers": [2] * 5,
                         "MaxPlayers": [4] * 5, "ComMaxPlaytime": [60] * 5})
    said = profile.sentence(profile.summarise(bare, np.arange(5)))

    assert said and "You lean towards" in said
