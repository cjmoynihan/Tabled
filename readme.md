# Tabled

A board game recommender over the [BoardGameGeek ratings dump][kaggle], with a
swipe interface: you react to games one at a time and each reaction sharpens
the next suggestion.

The data pipeline and the evaluation harness are built; the recommenders are
next. See [docs/PLAN.md](docs/PLAN.md) for the full roadmap and the reasoning
behind each decision.

## Setup

Put `games.csv` and `user_ratings.csv` from the Kaggle archive in `data/raw/`,
then:

```bash
pip install -e ".[dev]"
```

## Usage

```bash
tabled profile     # describe the raw CSVs; writes data/processed/profile.json
tabled prepare     # build the model-ready artifacts
tabled info        # show what is currently built
tabled evaluate    # score models against held-out users
```

`profile` is worth running first — it prints a retention table showing how many
games and interactions survive each candidate filter, which is how `prepare`'s
thresholds were chosen:

```bash
tabled prepare --min-game 50 --min-user 10
```

Support counts are cached after the first run, so trying different thresholds
does not re-read the 400 MB file. `--rescan` forces a fresh pass.

## What `prepare` produces

Four artifacts in `data/processed/`:

| file | contents |
|---|---|
| `ratings.npz` | CSR matrix, users × games, float32, explicit 1-10 ratings |
| `games.parquet` | one row per matrix column, in column order |
| `users.npy` | one username per matrix row, in row order |
| `manifest.json` | thresholds, shapes, density, what was dropped |

At the default thresholds that is **223,798 users × 17,140 games** with
**18.2M ratings** — 95.8% of the export, at 0.47% density. The build takes
about 40 seconds and peaks around 1.1 GB of memory.

```python
from tabled.data import prepare

ds = prepare.load()
ds.matrix                              # scipy CSR, explicit ratings
ds.games.loc[0, "Name"]                # what column 0 means

liked = prepare.to_implicit(ds.matrix) # binary likes, for ALS
centred, means = prepare.mean_center(ds.matrix)
```

Ratings are stored on the original 1-10 scale rather than pre-binarised. The
swipe UI collects thumbs up/down, but collapsing the scale during `prepare`
would freeze a modelling choice into the data; `to_implicit(threshold=)` does
it at model time instead, where it can be tuned.

## Evaluating

```bash
tabled evaluate --test-users 3000 --ks 3,5,10,20 --policy random
tabled evaluate --test-users 3000 --policy popular    # what the app will do
```

Entire users are held out, never random ratings — the app meets people it has
never seen, so hiding a slice of a known user's ratings would measure a
different task. Each test user has `k` ratings revealed as simulated swipes,
and the model's top-N is scored against the games they liked but were never
shown.

Quote both seeding policies. Under random seeding the popularity control
reaches nDCG@20 of 0.24; under popular-first seeding it falls to 0.026 by
k=20, because once the app has shown someone the most popular games, those
games can no longer be recommended. Which policy you use decides how hard the
baseline is, so a model beating "popularity" means nothing until you say
which.

Read accuracy next to coverage. Both non-random baselines score well while
drawing from **0.1% of the catalogue** — about 20 games, for everybody.

## Tests

```bash
pytest
```

The fixture is a miniature export shaped like the real thing — long-tailed
popularity, a duplicated pair, a null, an off-scale rating, and a rated game
with no metadata — so the tests exercise the same branches the 400 MB file
does.

## Layout

```
src/tabled/
  config.py          paths and tunable constants; imports nothing internal
  cli.py             the `tabled` command
  data/
    scan.py          chunked support counting, with an on-disk cache
    profile.py       describe the raw CSVs
    prepare.py       build, save and load the model-ready artifacts
  models/
    base.py          the Recommender contract and the Swipes type
    baselines.py     popularity, Bayesian average, random
  eval/
    split.py         hold out whole users
    metrics.py       recall@N, nDCG@N, coverage, popularity bias
    simulate.py      replay the swipe flow across a sweep of k
```

Layering runs `config → data → models → eval → cli`; nothing imports upwards
or sideways.

[kaggle]: https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek
