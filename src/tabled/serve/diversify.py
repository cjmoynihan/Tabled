"""
Stopping a recommendation list from being the same game five times.

Liking The Red Dragon Inn returns Red Dragon Inn 2, 3, 4 and 5. Every one is a
defensible *prediction* — people who like one do like the others — and the
list is still useless, because it answers a question the user did not ask. Ten
slots spent on one game is a ranking problem the accuracy metrics cannot see:
nDCG rewards each of those hits.

**Similarity thresholds do not work here**, which is worth stating because it
is the obvious first idea. Measured on the real data:

    Red Dragon Inn 2       0.339      near-duplicate, should be cut
    Ticket to Ride: Europe 0.293      arguably the same
    Codenames: Duet        0.255      genuinely different game, should stay
    Codenames: Pictures    0.193      genuinely different game, should stay

Any cutoff that removes Red Dragon Inn 2 also removes Codenames: Duet. The
orderings interleave, so no threshold separates them.

**The `Family` column does**, because it encodes publisher intent rather than
statistical closeness. 5,694 of 17,140 games carry one, and it groups exactly
the series that cause the problem. Capping a family at one entry keeps the
best Red Dragon Inn and drops the rest, while Codenames: Duet — same family as
Codenames, but only competing with it if Codenames itself is in the list —
survives on its own merit.

For the 11,446 games with no family, a similarity cap is the fallback. It is
imprecise for the reasons above, so it is set loosely: it exists to catch
outright twins, not to make fine judgements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Deliberately high. Given the overlap shown above, a tight cap would cut
# legitimately different games; this only catches near-identical pairs that
# the Family column missed.
SIMILARITY_CAP = 0.45


def family_keys(games: pd.DataFrame) -> np.ndarray:
    """
    A grouping key per game: its family, or a unique key when it has none.

    Games without a family must each get their own key rather than sharing an
    empty one, or every unfamilied game in the catalogue would count as a
    single series and the cap would allow just one of them.
    """
    if "Family" not in games.columns:
        return np.array([f"__{i}" for i in range(len(games))], dtype=object)

    family = games["Family"].to_numpy()
    return np.array([
        f.strip() if isinstance(f, str) and f.strip() else f"__{i}"
        for i, f in enumerate(family)
    ], dtype=object)


def _too_similar(model, candidate: int, accepted: list[int],
                 cap: float) -> bool:
    """Whether `candidate` is a near-twin of something already chosen."""
    lookup = getattr(model, "similarity_to", None)
    if lookup is None:
        return False
    return any(lookup(chosen, candidate) > cap for chosen in accepted)


def diversify(ranked: np.ndarray, games: pd.DataFrame, n: int = 10,
              max_per_family: int = 1, model=None,
              similarity_cap: float = SIMILARITY_CAP) -> np.ndarray:
    """
    Take the best `n` of `ranked`, allowing each family only so many slots.

    Order is preserved, so the *best* member of a family is the one kept —
    this thins the list rather than reordering it.

    `ranked` should be longer than `n`; anything filtered out needs a
    replacement to promote, and a list of exactly `n` cannot provide one.
    """
    keys = family_keys(games)
    seen: dict[str, int] = {}
    chosen: list[int] = []

    for candidate in ranked.tolist():
        key = keys[candidate]
        if seen.get(key, 0) >= max_per_family:
            continue
        if _too_similar(model, candidate, chosen, similarity_cap):
            continue
        seen[key] = seen.get(key, 0) + 1
        chosen.append(candidate)
        if len(chosen) >= n:
            break

    return np.array(chosen, dtype=np.int64)


def family_reaction_counts(games: pd.DataFrame,
                           reacted: np.ndarray) -> dict[str, int]:
    """How many games from each family the user has already been shown."""
    keys = family_keys(games)
    counts: dict[str, int] = {}
    for game in np.asarray(reacted, dtype=np.int64).tolist():
        key = keys[game]
        counts[key] = counts.get(key, 0) + 1
    return counts
