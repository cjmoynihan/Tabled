"""
Names to show people.

249 titles in the catalogue are shared by more than one game, covering 528
games in total. Four separate games are called Cosmic Encounter; five are
called Robin Hood. In a search box they are indistinguishable, so picking your
favourite is guesswork and the seed you give the model may not be the game you
meant.

Disambiguation is applied only where it is needed. Appending a year to all
17,140 games would make every list noisier to fix a problem affecting 3% of
it, so unique names are left exactly as they are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def display_names(games: pd.DataFrame) -> np.ndarray:
    """
    One label per game, with a year added only to break ties.

    Where a name and year still collide, the rating count settles it: the
    Cosmic Encounter with 28,643 ratings is the one most people mean, and
    saying so is more use than an internal id would be.
    """
    names = games["Name"].astype(str).to_numpy()
    labels = names.copy()

    counts = pd.Series(names).value_counts()
    ambiguous = set(counts[counts > 1].index)
    if not ambiguous:
        return labels

    years = games["YearPublished"].to_numpy()
    for position, name in enumerate(names):
        if name in ambiguous:
            year = int(years[position])
            labels[position] = f"{name} ({year})" if year > 0 else name

    # A year is usually enough. When it is not, fall back to how well known
    # the game is, which is the thing a person can actually recognise.
    repeated = pd.Series(labels).value_counts()
    still_ambiguous = set(repeated[repeated > 1].index)
    if still_ambiguous and "NumUserRatings" in games.columns:
        ratings = games["NumUserRatings"].to_numpy()
        for position, label in enumerate(labels):
            if label in still_ambiguous:
                labels[position] = f"{label}, {int(ratings[position]):,} ratings"

    return labels
