"""
Choosing the next card.

Ranking and choosing are different problems, and the gap between them is
where a swipe app is won or lost. The model answers "what would this person
most likely enjoy"; the card has to also be *worth asking about*.

Three forces, in tension:

**Recognisability.** A card for a game nobody has heard of earns a shrug, and
a shrug is not data. Early cards are gated to well-known games and the gate
relaxes as the session goes on — by which point the user has told us enough
that obscure suggestions are both better targeted and more interesting.

**Information.** The first few swipes are worth more as questions than as
hits. Games everyone rates highly separate nobody; the ones that split opinion
sort people fastest, which is why cold start seeds on divisive games rather
than beloved ones.

**Exploitation.** Eventually the point is to be right. Later cards are drawn
from the top of the ranking, with enough randomness that two people with
similar taste do not get identical sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabled.data.prepare import Dataset
from tabled.models.base import Recommender, Swipes
from tabled.serve import diversify

# How well-known a game must be to be shown, as a percentile of the
# catalogue's own popularity distribution. The first card must be something a
# stranger plausibly recognises — the top 1% — and by fifteen swipes anything
# above the median is fair game.
#
# Expressed as percentiles rather than rating counts on purpose. An absolute
# floor ("at least 200 ratings") silently means something different on every
# catalogue: change the `prepare` thresholds, or run against a subset, and a
# number tuned for 17,000 games can exclude everything the user would have
# liked while still looking like a reasonable constant.
GATE_START_PCT = 99.0
GATE_END_PCT = 50.0
GATE_SWIPES = 15


def popularity_gate(n_swipes: int, popularity: np.ndarray) -> float:
    """The minimum rating count a game needs to be worth showing right now."""
    ratio = min(n_swipes / GATE_SWIPES, 1.0)
    percentile = GATE_START_PCT + (GATE_END_PCT - GATE_START_PCT) * ratio
    return float(np.percentile(popularity, percentile))


def divisiveness(ds: Dataset) -> np.ndarray:
    """
    How much people disagree about each game.

    BGG's own StdDev when available. A game everyone rates 8 tells us nothing
    about a new user; a game half the world rates 9 and half rates 4 splits
    the audience in one card, which is exactly what a cold start needs.
    """
    if "StdDev" in ds.games.columns:
        return ds.games["StdDev"].to_numpy(dtype=np.float64)

    # Fall back to computing it from the matrix.
    matrix = ds.matrix.tocsc()
    out = np.zeros(ds.shape[1])
    for game in range(ds.shape[1]):
        lo, hi = matrix.indptr[game], matrix.indptr[game + 1]
        out[game] = matrix.data[lo:hi].std() if hi > lo else 0.0
    return out


def cold_start_seeds(ds: Dataset, n: int = 12, pool: int = 400,
                     seed: int = 0) -> np.ndarray:
    """
    The opening hand: recognisable games that people disagree about.

    Drawn from the `pool` most-rated games, ranked by disagreement, then
    spread across the weight range. The spread matters — the most divisive
    popular games skew heavy, and opening with six sprawling strategy titles
    would ask the same question six times.
    """
    popularity = ds.popularity()
    candidates = np.argsort(-popularity)[:pool]

    spread = divisiveness(ds)[candidates]
    weight = ds.games["GameWeight"].to_numpy(dtype=np.float64)[candidates]

    # Four weight bands, filled round-robin by disagreement, so the opening
    # hand asks about complexity as well as taste.
    bands = np.clip(np.digitize(weight, [1.5, 2.5, 3.5]), 0, 3)
    ranked = candidates[np.argsort(-spread)]
    ranked_bands = bands[np.argsort(-spread)]

    picked: list[int] = []
    for band in range(4):
        picked.extend(ranked[ranked_bands == band][:max(1, n // 4)].tolist())

    # Top up from the most divisive overall if the bands came up short.
    for game in ranked.tolist():
        if len(picked) >= n:
            break
        if game not in picked:
            picked.append(game)

    rng = np.random.default_rng(seed)
    return rng.permutation(np.array(picked[:n], dtype=np.int64))


def next_card(model: Recommender, swipes: Swipes, ds: Dataset,
              shown: np.ndarray, rng: np.random.Generator,
              seeds: np.ndarray | None = None, top_m: int = 25,
              temperature: float = 1.0,
              max_per_family: int = 2) -> int | None:
    """
    The single game to show next, or None when the catalogue is exhausted.

    While the user has told us nothing, this serves the cold-start hand. After
    that it samples from the model's top `top_m` eligible games rather than
    taking the argmax — always showing the single best card makes every
    session with similar taste identical, and gives the model no chance to
    discover it was wrong.
    """
    already = set(shown.tolist())

    if seeds is not None and len(swipes) == 0:
        remaining = [int(g) for g in seeds if int(g) not in already]
        if remaining:
            return remaining[0]

    scores = np.asarray(model.score(swipes), dtype=np.float64).copy()
    if len(already):
        scores[np.fromiter(already, dtype=np.int64, count=len(already))] = -np.inf

    popularity = ds.popularity()
    eligible = popularity >= popularity_gate(len(shown), popularity)
    # The gate is a preference, not a rule: if it would leave nothing, ignore
    # it rather than telling the user we have run out of board games.
    if np.isfinite(scores[eligible]).any():
        scores[~eligible] = -np.inf

    # Having already seen two Red Dragon Inns, a third is not a question worth
    # asking — the answer is known and the card is wasted.
    if max_per_family and len(shown):
        counts = diversify.family_reaction_counts(ds.games, shown)
        exhausted = {key for key, seen in counts.items()
                     if seen >= max_per_family}
        if exhausted:
            keys = diversify.family_keys(ds.games)
            blocked = np.array([k in exhausted for k in keys])
            if np.isfinite(scores[~blocked]).any():
                scores[blocked] = -np.inf

    finite = int(np.isfinite(scores).sum())
    if finite == 0:
        return None

    m = min(top_m, finite)
    top = np.argpartition(-scores, m - 1)[:m]
    top = top[np.isfinite(scores[top])]
    if len(top) == 0:
        return None
    if temperature <= 0:
        return int(top[np.argmax(scores[top])])

    # Softmax over the shortlist, normalised so the spread of scores rather
    # than their absolute size decides how adventurous the pick is.
    values = scores[top]
    values = (values - values.max()) / (values.std() + 1e-9) / temperature
    weights = np.exp(values)
    return int(rng.choice(top, p=weights / weights.sum()))


def top_picks(model: Recommender, swipes: Swipes, ds: Dataset,
              judged: np.ndarray, n: int = 10,
              max_per_family: int = 1) -> np.ndarray:
    """
    The user's current best `n` games, thinned to one per series.

    Over-fetches before thinning: filtering a list of exactly `n` leaves gaps,
    so the candidate pool has to be several times longer than the answer.

    `judged` should be the games the user has taken a position on — liked,
    passed on, or already played — and *not* the ones they merely skipped. A
    skip means they did not recognise the game, which is the best possible
    reason to put it in front of them as a result.
    """
    if len(swipes) == 0:
        return np.array([], dtype=np.int64)

    candidates = model.recommend(swipes, n=n * 12, exclude=judged)
    return diversify.diversify(candidates, ds.games, n=n,
                               max_per_family=max_per_family, model=model)


def explain(model: Recommender, swipes: Swipes, game: int,
            games: pd.DataFrame) -> str | None:
    """A human-readable reason, when the model can give one."""
    reason = getattr(model, "because_of", None)
    if reason is None:
        return None
    found = reason(swipes, game)
    if found is None:
        return None
    return str(games.loc[found[0], "Name"])
