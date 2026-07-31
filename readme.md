# Tabled

A board game recommender over the [BoardGameGeek ratings dump][kaggle], with a
swipe interface: you react to games one at a time and each reaction sharpens
the next suggestion.

Currently the data pipeline is built and verified. See [docs/PLAN.md](docs/PLAN.md)
for the full roadmap and the reasoning behind each decision.

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
```

[kaggle]: https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek
