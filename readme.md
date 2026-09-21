# Tabled

A board game recommender over the [BoardGameGeek ratings dump][kaggle], with a
swipe interface: you react to games one at a time and each reaction sharpens
the next suggestion.

The pipeline, the evaluation harness, both recommenders and the swipe app are
built. See [docs/PLAN.md](docs/PLAN.md) for the full roadmap and the reasoning
behind each decision.

```bash
pip install -e ".[model,app]"
tabled prepare                     # build the matrix   (~40 s)
tabled fit --model item-item       # build the model    (~2 min)
streamlit run app.py               # swipe
```

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
tabled fit         # fit and cache a recommender
tabled evaluate    # score models against held-out users
```

Run `profile` to see model construction details. These are used for tabled prepare:

```bash
tabled prepare --min-game 50 --min-user 10
```

Support counts are cached after the first run, so using different thresholds
does not re-read the 400 MB file. Use `--rescan` forces a fresh pass.

## What `prepare` does

Four artifacts in `data/processed/`:

| file | contents |
|---|---|
| `ratings.npz` | CSR matrix, users × games, float32, explicit 1-10 ratings |
| `games.parquet` | one row per matrix column, in column order |
| `users.npy` | one username per matrix row, in row order |
| `manifest.json` | thresholds, shapes, density, what was dropped |

At the default thresholds that is **223,798 users × 17,140 games** with
**18.2M ratings**, using 95.8% of the export at 0.47% density. Expect the build to take
about 40 seconds and around 1.1 GB of memory.

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
would modify the ground truth's data; `to_implicit(threshold=)` does
it at model time instead, where it can be tuned.

## Recommending

Two models, both cached because fitting is far too slow to happen
each request:

```bash
tabled fit --model item-item --holdout   # ~2 min, writes item_item.npz (14 MB)
tabled fit --model als --holdout         # ~30 s, writes als.npz (9 MB)
```

`--holdout` fits on the training split only and records which users were held
out, so `evaluate` can refuse to score a model against a split it trained on.
Drop it to fit on everyone for serving.

```python
from tabled import config
from tabled.data import prepare
from tabled.models import store
from tabled.models.base import Swipes

ds = prepare.load()
model = store.load_item_item(config.ITEM_ITEM_NPZ, ds.shape[1])

swipes = Swipes.from_reactions(liked=[1234, 5678], disliked=[910])
for game in model.recommend(swipes, n=10):
    print(ds.games.loc[game, "Name"])
```

**Item-item** ranks by similarity to what you reacted to, with similarities
shrunk by co-rater count so sparse pairs (eg: pair sharing four raters) cannot outrank a well-defined pair (eg: four thousand). Only the top 100 neighbours per game are kept.

**ALS** factorises binarised likes and folds a new user in with a single
least-squares solve against fixed item factors. Unusually for an
implicit-feedback model it uses dislikes, entered as high-confidence
observations of zero preference — in a swipe app a thumbs-down is evidence,
not absence.

## The app

```bash
streamlit run app.py
```

Fit without `--holdout` before serving real users — the held-out artifact is
for honest evaluation and ignores 5,000 people's ratings.

It opens with an optional picker: name a few games you love and skip ahead.
Worth doing — the harness says three named favourites get further than ten
swipes, against a harder target set. Skip it and you get twelve recognisable,
divisive games instead (Monopoly, Chess, Go, Cards Against Humanity…) spread
across the weight range. Games everyone likes separate nobody; the ones that
split opinion sort people fastest.

Your **top 10** sits in its own tab, with covers, and each pick can be liked,
passed or marked as played — which drops it and refills the list.

Skipping is deliberately *not* a rejection. A skip means "I don't know this
one", so the game stops interrupting the swipe stream but stays eligible as a
result — which is the point, since an unfamiliar game is exactly what a
recommendation is for. Only Like, Nope and Played remove a game from your
top 10.

Lists allow **one game per series**. Liking The Red Dragon Inn used to return
Red Dragon Inn 2 through 7; now it returns RDI 2 and then Munchkin, Fluxx,
Guillotine, BANG!. Codenames still yields Codenames: Duet, because that is a
genuinely different game — the distinction comes from the `Family` column, not
a similarity threshold, since no threshold separates those two cases.

249 titles are shared by more than one game: four are called Cosmic Encounter,
five Robin Hood. Those get a year appended (`Cosmic Encounter (2008)`) so you
can tell which you are picking. The other 96.9% of names are left alone, since
disambiguating everything would make every list noisier to fix a problem
affecting 3% of it.

Sessions **export and import** as JSON, so you can come back without an
account. Restore matches on BGGId rather than matrix index, so a rebuilt
catalogue cannot silently return someone else's taste.

Four reactions, deliberately distinct:

| | shapes recommendations | shown as a card again | can appear in your top 10 |
|---|---|---|---|
| 👍 Like / 👎 Nope | yes | no | no |
| ✓ Played, ⊘ Dismiss | no | no | no |
| 🤷 Skip | no | no | **yes** |

A skip usually means unfamiliarity or indifference, not distaste, so folding
it into "dislike" would poison the signal. It is the one reaction that leaves
the card stream without removing the game from the results.

Skipping repeatedly is also a signal in itself. `skip_weight` rises with each
skip and falls sharply on any real opinion, and it pulls the catalogue back
towards games you will actually recognise. Skip everything and you end up
among Monopoly and Cards Against Humanity; rate one and Xiangqi comes straight
back. Rating a famous game tells the model less, which is the deliberate
trade: a shrug tells it nothing at all.

Each card says why it was chosen (*Because you liked Go*), which turns a
recommendation into a claim the user can disagree with and makes a bad
neighbour list visible rather than merely felt.

Sessions can be saved to `data/sessions/swipes.jsonl` — every metric in this
project comes from *simulated* swiping reconstructed from BGG ratings, and
these logs are the only record of what people do in the actual interface.

## Deploying

Streamlit Community Cloud builds from the repo, so the two artifacts the app
reads are committed (~17 MB together) while the 400 MB export and the 146 MB
ratings matrix stay out of git.

```bash
tabled prepare                  # writes games.parquet (with popularity)
tabled fit --model item-item    # writes item_item.npz, ~2 min
```

Note the missing `--holdout`: that flag exists so evaluation stays honest by
ignoring 5,000 users, which is the wrong trade for something being served.

Then commit `data/processed/games.parquet` and `data/processed/item_item.npz`,
push, and point Community Cloud at `app.py`. `requirements.txt` installs the
package itself via `.[app]`, so dependencies stay declared once in
`pyproject.toml`.

**The app never loads the ratings matrix.** It exists to fit models; the one
thing serving used it for, counting ratings per game, is now a column in
`games.parquet`. Skipping it takes the process from 352 MB to **188 MB**,
comfortably inside Community Cloud's 1 GB, and removes a 146 MB read from
every cold start, which is most of what a visitor waits through when the app
wakes from sleep.

Add `assets/powered-by-bgg.png` for the BoardGameGeek attribution the data
terms require; see [assets/README.md](assets/README.md). Without it the app
falls back to a text link, so attribution is never silently missing.

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

### Results

nDCG@20 under popular-first seeding, 5,000 held-out users:

| model | k=3 | k=5 | k=10 | k=20 | coverage |
|---|---|---|---|---|---|
| item-item | 0.171 | **0.185** | **0.170** | **0.126** | 6–8% |
| als | 0.163 | 0.171 | 0.144 | 0.093 | 4–7% |
| popularity | **0.175** | 0.134 | 0.070 | 0.027 | 0.1% |

Item-item wins across the range the app operates in and covers far more of the
catalogue. ALS does overtake it at k≥10 — but only under random seeding, and
only by about 1%. See [docs/PLAN.md](docs/PLAN.md) for the full comparison.

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
    item_item.py     shrunk cosine similarity with top-k neighbours
    mf.py            implicit ALS with user fold-in
    store.py         save and load fitted models
  eval/
    split.py         hold out whole users
    metrics.py       recall@N, nDCG@N, coverage, popularity bias
    simulate.py      replay the swipe flow across a sweep of k
  serve/
    session.py       the four reactions, export/import, the swipe log
    policy.py        cold start, popularity gate, card selection
    diversify.py     one game per series, so sequels cannot fill a list
    labels.py        year suffixes for the 249 duplicated game names
    profile.py       a written description of what someone likes
app.py               the Streamlit shell — wiring only
```

Layering runs `config → data → models → {eval, serve} → cli`; nothing imports
upwards or sideways, and `eval` and `serve` are siblings so neither can reach
into the other. `tests/test_architecture.py` enforces it.

[kaggle]: https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek
