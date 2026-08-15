# Tabled — project plan

A board game recommender jumpstarted with initial Kaggle data, with a swipe interface:
the user reacts to games one at a time, and each reaction sharpens the next
suggestion.

Phases 0 and 1 are **done** — the numbers below are measured, not estimated.

---

## What the data actually looks like

Measured by `tabled profile` over the full `user_ratings.csv`:

| | |
|---|---|
| ratings | 18,942,152 |
| users | 411,374 |
| games | 21,925 |
| **density** | **0.21%** |
| ratings per user | median 12, mean 46, max 6,493 |
| ratings per game | median 125, mean 864, max 107,760 |
| ratings ≥ 7.5 | 44.2% |

Findings worth carrying forward:

- **The join is clean.** Every rated game appears in `games.csv` and every game
  in `games.csv` has at least one rating. No orphan handling needed.
- **Both distributions are severely long-tailed.** The median user has 12
  ratings and the median game 125, but the means are 46 and 864. Averages will
  mislead throughout this project; report medians.
- **32,150 duplicate `(user, game)` pairs exist** in the raw file. SciPy sums
  duplicate coordinates when assembling a sparse matrix, so left alone these
  become ratings of 16 or 20. `prepare` drops them explicitly.
- **Ratings are a 1-10 scale with halves and stray decimals** — 94 distinct
  values, not 10. About 1.02M ratings (5.4%) sit on arbitrary decimals like
  2.3, which is legitimate: BGG accepts free-form values.
- **Eight ratings fall outside the 1-10 scale**, three of them at zero.
  Statistically nothing, but a zero written into a sparse matrix is
  indistinguishable from *not rated*, so it would become a silently wrong
  observation rather than a visibly bad one. `prepare` drops them.
- **Popularity is extremely concentrated.** The top game has 107,760 ratings
  against a median of 125. This makes "recommend the most popular unrated
  game" a genuinely strong baseline, and it is why one is mandatory in
  Phase 2.

---

## Phase 0 — Understand the data ✅

`tabled profile` → `data/processed/profile.json`.

One chunked pass, no full load. Produces the table above plus a retention grid
showing how many games and interactions survive each candidate filter, which
is what Phase 1's thresholds were chosen from rather than guessed.

## Phase 1 — Structure the data ✅

`tabled prepare` → four artifacts in `data/processed/`:

| artifact | contents |
|---|---|
| `ratings.npz` | CSR matrix, users × games, float32, explicit 1-10 ratings |
| `games.parquet` | one row per matrix column, in column order |
| `users.npy` | one username per matrix row, in row order |
| `manifest.json` | thresholds, shapes, density, what was dropped |

Two chunked passes: count support, then build. Peak memory is set by the
surviving interactions, not the 400 MB of CSV — measured peak RSS is 1.1 GB,
and the whole build takes about 40 seconds from cold.

**Result at `min_game=50`, `min_user=10`:**

| | raw | kept |
|---|---|---|
| users | 411,374 | 223,798 |
| games | 21,925 | 17,140 |
| interactions | 18,942,144 | 18,154,249 |
| density | 0.21% | 0.47% |

95.8% of all ratings survive. Filtering converged in 4 rounds, and 32,150
duplicate `(user, game)` pairs were averaged rather than summed.

**Verification.** Each matrix column's mean rating was checked against
`games.csv`'s own `AvgRating` for the ten most-rated games; all agree to
within 0.06, which is strong evidence the column indices line up with the
games they claim to be. As a signal check, cosine similarity on mean-centred
columns gives Gloomhaven ↔ Terraforming Mars +0.55, Catan ↔ Carcassonne +0.24,
and Catan ↔ Gloomhaven −0.26 — the ordering a board gamer would predict.

**Design decisions made here, and why:**

- **Explicit ratings are stored, not pre-binarised likes.** The swipe UI
  collects thumbs up/down, so it is tempting to collapse the 1-10 scale
  immediately. Doing that at prepare time would freeze a modelling choice into
  the data. `to_implicit(threshold=)` does the collapse at model time instead,
  where the threshold can be tuned and reverted.
- **Filtering converges rather than applying once.** Dropping rare games costs
  some users their ratings, pushing those users under the user threshold,
  which can push a borderline game back under the game threshold. Applying
  each filter once leaves a matrix that does not satisfy its own thresholds.
  Convergence took 4 rounds and moved the game count from 17,552 to 17,140.
- **Only games we have metadata for enter the matrix.** A recommendation the
  UI cannot name or picture is useless.
- **Per-user mean centring is available but not baked in.** BGG raters differ
  more in generosity than in taste; some never go below 6. Centring removes
  that offset so similarity reflects relative preference rather than rating
  style.
- **Support counts are cached.** They depend only on the file, not on any
  threshold, so re-reading 400 MB for every experiment is waste — and
  choosing thresholds well means running several. The cache is keyed on the
  source file size *and* a version constant, so changing the counting rules
  invalidates it instead of silently serving counts from the old rules.
- **Artifacts are written to scratch and copied in.** An interrupted write
  otherwise leaves a truncated `.npz` exactly where the next `load()` looks
  for a good one, and a truncated matrix is not obviously broken — it is just
  missing its tail. This turns that failure into "no artifact".

## Phase 2 — Evaluation harness ✅

`tabled evaluate` → the scoreboard any model must beat.

- **Split by user, not by interaction.** The app serves brand-new users, so
  hold out entire users. Splitting random cells measures a different task than
  the one being shipped.
- **Simulate the swipe flow.** For each held-out user, reveal `k` of their
  ratings as swipes, ask for top-N from everything they have not seen, score
  against their remaining liked games. Sweep `k ∈ {3, 5, 10, 20, 50}` — this
  is the axis the whole comparison turns on.
- **Rank metrics, not RMSE.** The UI never displays a predicted rating; it
  shows one game. Recall@N and nDCG@N measure that. RMSE optimises something
  the product does not do.
- **Three controls, all mandatory:** most-popular, highest-Bayesian-average,
  and random. Given how concentrated popularity is, a model that cannot beat
  most-popular is not earning its complexity.
- **Report coverage and popularity bias alongside accuracy.** A recommender
  that only ever suggests the top 200 games will score well and feel useless.
  Track distinct games recommended across all test users, and the median
  popularity rank of what gets served.

### Baseline results

3,000 held-out users, top-20 lists, users needing ≥25 ratings to be eligible.

**Random swipe seeding** (unbiased, the standard way to quote this):

| model | k=3 | k=5 | k=10 | k=20 | coverage | pop %ile |
|---|---|---|---|---|---|---|
| popularity | **0.239** | **0.236** | **0.229** | **0.208** | 0.1% | 0.1 |
| bayes | 0.156 | 0.154 | 0.148 | 0.136 | 0.1% | 0.4 |
| random | 0.003 | 0.003 | 0.003 | 0.002 | 97% | 50 |

**Popular-first seeding** (what the app will actually do):

| model | k=3 | k=5 | k=10 | k=20 | coverage | pop %ile |
|---|---|---|---|---|---|---|
| popularity | **0.174** | 0.132 | 0.068 | 0.026 | 0.1% | 0.1 |
| bayes | 0.151 | **0.141** | **0.113** | **0.074** | 0.1% | 0.5 |
| random | 0.002 | 0.003 | 0.002 | 0.003 | 97% | 50 |

Figures are nDCG@20.

**The finding that matters: the seeding policy decides how hard the baseline
is.** Under random seeding, popularity looks close to unbeatable — nDCG 0.24
and a 91% hit rate from ignoring the user completely. Under popular-first
seeding it collapses to 0.026 by k=20, an 8× drop, and Bayesian average
overtakes it from k=5.

The reason is not subtle once seen: if the app has already shown someone the
twenty most popular games, those games are no longer available to recommend,
and what the user still likes is by construction *not* mainstream. The
popularity baseline runs out of catalogue.

Two consequences for Phase 3. First, quote both policies or the bar is
meaningless — a model beating popularity at 0.03 has done nothing if the
honest bar is 0.24. Second, and more useful: **the app's own seeding policy
destroys the popularity advantage**, so personalisation has far more room
than the random-seeded table suggests. That is an argument for the design in
Phase 4, not just an evaluation detail.

Also worth noting: popularity and bayes both recommend from **0.1% of the
catalogue** — roughly 20 games, to everyone. For a discovery app that is a
failure mode regardless of nDCG, and it is the number to watch when the real
models arrive.

## Phase 3 — The two recommenders

Shared interface: `fit(matrix)` and `recommend(swipes, n) -> game indices`,
so the app and the harness are model-agnostic.

**Item-item cosine similarity.** Mean-centre, cosine between game columns,
shrink by co-rater count (`sim * n_common / (n_common + λ)`) so two games
sharing four raters do not outrank two sharing four thousand. Keep top-K
neighbours per game (K ≈ 100); the full 17,140² dense matrix is ~1.2 GB, the
truncated one is a few MB. Scoring a new user is a sum over their swipes'
neighbour lists — no fitting, and it explains itself: *because you liked X*.

**Matrix factorisation.** Start with `implicit`'s ALS on binarised likes, and
truncated SVD on mean-centred explicit ratings as a cross-check. New users are
handled by folding in: solve for a user vector from their swipes against fixed
item factors. Sweep factors ∈ {32, 64, 128} and regularisation.

Both must clear the Phase 2 table under *both* seeding policies, and be read
against coverage — a model that beats popularity on nDCG while recommending
the same 20 games has not solved the problem this app exists for.

**On the hypothesis that MF wins because the data is sparse.** Likely correct
for users with real history, and 0.21% density is exactly the regime where
factorisation earns its keep — it generalises across games that share no
raters, which item-item structurally cannot. But the swipe app mostly operates
at `k = 3` to `15`, and a fold-in from 5 observations is a noisy estimate of a
64-dimensional vector, while item-item's neighbour sum degrades gently. So
expect item-item to be competitive or better at low `k` and MF to pull ahead
as `k` grows, with a crossover somewhere in between. Two consequences: the
`k` sweep in Phase 2 is the experiment, not a detail; and the answer may be
"item-item for the first few swipes, MF after", which is a shipping strategy
rather than a tie-break. A blend is the third arm worth running.

## Phase 4 — Swipe policy

Ranking is not the same as choosing what to show next. The card the user sees
should balance:

- **Exploit** — the current top recommendation.
- **Explore** — something informative about an axis still uncertain (weight,
  player count, theme). Early swipes are worth more as questions than as hits.
- **Recognisability** — a card for a game nobody has heard of earns a shrug,
  not a signal. Gate early cards on a minimum popularity, then relax it.

Cold start: seed with a fixed set of ~20 well-known, maximally *divisive*
games (high `StdDev`, high rating count). Games everyone likes carry almost no
information; the ones that split opinion separate users fastest.

Also decide: does a dislike push away only that game, or its neighbours too?
Asymmetric weighting (dislikes weaker than likes) is usually right, since
"not for me right now" and "bad game" look identical through a swipe.

## Phase 5 — Streamlit app

Session state holds the swipe list; each swipe re-scores against the
precomputed artifacts. Like / dislike / skip / seen-it — "skip" and "seen it"
are different signals and conflating them loses information. Show *why* a game
was suggested; it makes the thing feel intelligent and makes bugs visible.

Artifacts load once behind `@st.cache_resource`. Everything the app needs must
be precomputed — no fitting inside a request.

## Phase 6 — Ship

Streamlit Community Cloud, with artifacts built offline and committed to
release storage rather than rebuilt on deploy. Log swipe sessions from the
start: your own users are the only data that measures whether any of this
works, and the offline harness is a proxy for it.

---

## Open questions

1. **Expansions.** If `games.csv` includes expansions as their own rows,
   "you liked Catan, try Catan: Seafarers" is a bad recommendation dressed up
   as a good score. Worth confirming against the archive's other files.
2. **Time.** The dump is a January 2022 snapshot with no timestamps, so
   recency effects and drift are invisible. Nothing to fix, but worth stating
   before drawing conclusions about "trending" anything.
3. **Threshold sensitivity.** `min_game=50` was chosen to keep 99% of
   interactions. Whether a stricter cut helps or hurts is an experiment to run
   in Phase 2, not a decision to defend now.
