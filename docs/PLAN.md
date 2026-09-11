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

## Phase 3 — The two recommenders ✅

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

### Results

nDCG@20, 5,000 held-out users, the same split for every arm. **Bold** is the
best model in each column.

**Random swipe seeding:**

| model | k=3 | k=5 | k=10 | k=20 | coverage |
|---|---|---|---|---|---|
| popularity | **0.238** | **0.235** | 0.227 | 0.210 | 0.1% |
| item-item | 0.183 | 0.208 | 0.237 | 0.246 | 15–34% |
| als | 0.167 | 0.200 | **0.240** | **0.247** | 10–14% |
| bayes | 0.157 | 0.155 | 0.148 | 0.137 | 0.1% |

**Popular-first seeding — what the app will actually do:**

| model | k=3 | k=5 | k=10 | k=20 | coverage |
|---|---|---|---|---|---|
| item-item | 0.171 | **0.185** | **0.170** | **0.126** | 6–8% |
| als | 0.163 | 0.171 | 0.144 | 0.093 | 4–7% |
| popularity | **0.175** | 0.134 | 0.070 | 0.027 | 0.1% |
| bayes | 0.152 | 0.142 | 0.114 | 0.075 | 0.1% |

**The crossover is real, and lands where predicted.** Under random seeding
item-item leads at k=3 and k=5, ALS overtakes it at k=10, and both beat
popularity from k=10 onward. The mechanism was the one expected: fold-in from
five observations is a noisy estimate of a 64-dimensional vector, while a
neighbour sum degrades gently.

**But the margin is small and the app never gets there.** ALS's win at k=10 is
0.240 vs 0.237 — about 1%, well inside anything worth shipping a second model
for. Under the seeding the app actually uses, item-item wins at *every* k, and
the gap widens rather than closing: 35% better by k=20.

The reason is the same effect that broke the popularity baseline in Phase 2.
Once the recognisable games have been shown, what remains to be found is
niche, and that is precisely where 64 latent factors are lossiest — the tail
is what gets compressed away. Item-item keeps a specific neighbour list for
every game, however obscure. Its coverage is 2–3× ALS's throughout, which is
the same fact measured a different way.

**Recommendation: ship item-item.** It wins across the operating range, covers
far more of the catalogue, needs no per-user solve, and can say *because you
liked Gloomhaven* — which in a swipe interface is a feature, not a nicety.
Keep ALS: it is 9 MB and 30 seconds to fit, and it is the natural blend
partner if the k≥10 regime ever matters.

**On the hypothesis that MF wins because the data is sparse** — the reasoning
was sound and the mechanism showed up in the measurements, but the effect is
about 1% where it exists at all, and it does not exist in the regime this app
operates in. Sparsity favours factorisation for users with history; a swipe
app's users do not have history, which turns out to matter more than density.

### Verification beyond the metrics

Neighbour lists were read by eye before the numbers were trusted, because a
plausible nDCG can hide nonsense:

- Gloomhaven → Jaws of the Lion, Pandemic Legacy, Spirit Island, Scythe
- Codenames → Codenames: Duet, Codenames: Pictures, Decrypto, Just One
- Terraforming Mars → Great Western Trail, Scythe, Wingspan, Brass
- Catan → Ticket to Ride, Carcassonne, Citadels
- Twilight Struggle → Through the Ages, Brass, War of the Ring

Each groups by weight and play style rather than by popularity, which is what
the shrinkage term is there to buy.

### One trap worth naming

Both models are fitted against a *specific* held-out split. Evaluating a
cached model against a different split silently tests it on users it trained
on, and the only symptom is that the scores improve. `tabled evaluate` now
refuses to run when the artifact's recorded split does not match the one being
evaluated.

## Phase 4 — Swipe policy and learning from real swipes

Now the highest-value phase, and Phase 3 is the reason why. Every number above
is a *simulation* of swiping, reconstructed from ratings people gave on BGG
years ago in a completely different interface. Real swipe logs are the only
data that measures the actual product, and they capture things the ratings
dump structurally cannot: what someone skipped, how long they hesitated, which
card made them stop.

Logging should go in before the UI ships, not after — the seam already exists.
`Swipes` is the type every model consumes, so a session is just an append-only
list of `(game_index, reaction, timestamp)` and replaying one reconstructs the
exact model input. Worth recording from day one:

- **reaction** — like / dislike / skip / already-played, kept distinct.
  Collapsing "skip" into "dislike" destroys the most interesting signal there
  is, because a skip is usually *indifference or unfamiliarity*, not distaste.
- **position in the session**, since the 3rd swipe and the 30th mean different
  things.
- **what was shown and not chosen** — the card the user saw is a decision the
  ratings dump has no equivalent for, and it is what makes the log worth more
  than another copy of BGG.

That last point is the real prize: implicit feedback on *impressions* rather
than ratings. It is also why the harness above should eventually be replayed
against logged sessions instead of simulated ones.

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

## Phase 5 — Streamlit app ✅

`streamlit run app.py`. Artifacts load once behind `@st.cache_resource`;
nothing is fitted inside a request.

All the decisions live in `tabled.serve`, and `app.py` only wires them to
widgets — which is why the session logic is tested without a browser, and the
Streamlit layer is tested separately with `AppTest`.

**Cold start.** Twelve recognisable, divisive games, spread across the weight
range: Monopoly, Chess, Cards Against Humanity, Magic, Go, Poker, Diplomacy,
Fluxx, Twilight Imperium, The Game of Life. Games everyone likes separate
nobody; the ones that split opinion sort people fastest. The weight spread
matters too — the most divisive popular games skew heavy, and opening with six
sprawling strategy titles asks the same question six times.

**Four reactions, kept distinct.** Like and dislike feed the model; skip and
already-played are recorded but excluded from taste.

The subtle one is skip, which has *two* different meanings depending on where
you are looking. All four reactions take a game out of the card stream —
re-showing a skipped card reads as the app not listening. But only like,
dislike and played remove a game from the results list. A skip means "I don't
recognise this", which is the best possible reason to recommend something, so
excluding skipped games from the top 10 would suppress exactly the
suggestions the user is most likely to find useful.

Hence two views on a session: `shown` governs what gets served as a card,
`judged` governs what can appear as a result. A skip is *ask me later*, not
*never again*.

**Cards are sampled, not maximised.** Drawn by softmax from the top 25
eligible games. Always serving the argmax makes every session with similar
taste identical and gives the model no way to discover it was wrong.

**Explanations.** *Because you liked Go.* Free from item-item, since the score
literally is a sum of per-neighbour terms — and impossible from the
factorisation, which was part of the Phase 3 recommendation.

### Not knowing the games is its own failure mode

A user new to the hobby skips repeatedly, and a run of skips is demoralising:
nothing happens, and every card is a title they have never heard of. Worse,
the app looks broken when it is in fact working exactly as designed.

`Session.skip_weight` tracks this in `[0, 1)`. Each skip moves it half the
remaining distance to 1; each like or dislike knocks three quarters off it.
Recovery is deliberately faster than the build-up, so one engaged answer
undoes a short run of shrugs. Both updates are fractions of the remaining
distance, which keeps the value in range with no clamping.

It feeds the popularity gate rather than adding a second mechanism, pulling
the eligible set back towards the famous end in proportion to how much the
user has been shrugging. Measured on the real catalogue, twelve swipes in:

| skip_weight | games eligible |
|---|---|
| 0.00 | 6,892 |
| 0.50 | 3,534 |
| 0.94 | 575 |

A user who skips everything is held among Monopoly, Magic, Cards Against
Humanity and Diplomacy. The moment they rate one, the weight falls from 1.00
to 0.25 and Xiangqi and Shogi come straight back.

**Why popularity rather than collaborative signal.** The tempting alternative
is to find games rated by people who rated what this user rated. But that
estimates *P(they would like it)*, which is what the recommender already
optimises — using it here would collapse the distinction the mechanic exists
to make. What we actually want is *P(they have an opinion at all)*, which is
mostly *P(they have heard of it)*, and raw popularity is the direct proxy for
that. The second method would also quietly bias towards games the user likes,
turning a familiarity problem into a taste one.

The cost is real and worth stating: a rating of Monopoly carries far less
information than a rating of something obscure. This deliberately trades
information per answer for getting an answer at all.

`skip_weight` is recomputed from the event history rather than accumulated,
so undo and session import stay correct for free. A running counter would
drift out of step with the events it claims to summarise.

### The bug this phase caught

The popularity gate was first written as an absolute floor: 20,000 ratings at
the first card relaxing to 200 by the fifteenth. It looked right on the real
catalogue and quietly broke everywhere else — on the test fixture, whose games
top out at a few hundred raters, the relaxed gate of 200 excluded *every*
game the user could have wanted, so sessions converged on nothing. The
end-to-end session test is what surfaced it.

It is now a percentile of the catalogue's own popularity distribution: top 1%
at the first card, relaxing to the median by the fifteenth. On the real data
that reproduces the original numbers almost exactly (172 games eligible at
the first card, 8,575 by the fifteenth) while working on any catalogue.
Absolute thresholds tuned against one dataset are a recurring trap in this
project — the same shape of mistake as the rating scale in Phase 1.

### Should users name favourites up front? Yes — measured, not guessed

Adding a `favourites` seed policy to the harness (reveal the user's own
highest-rated games, standing in for an "add games you love" picker) answers
this directly. nDCG@20, item-item, same 5,000 held-out users:

| seeding | k=3 | k=5 | k=10 |
|---|---|---|---|
| favourites | **0.245** | **0.230** | 0.193 |
| random | 0.183 | 0.208 | 0.237 |
| popular-first | 0.171 | 0.185 | 0.170 |

**Three named favourites beat ten swipes** under either other policy. And the
comparison understates it: naming your favourites removes your *best* games
from the pool of things left to find, so the favourites arm is scored against
a harder target set and still wins. It also wins while volunteering no
dislikes at all, which is the one real cost of a picker.

So the app opens with an optional picker. Optional because it only works for
people who already know what they like and can recall the names — which is
not everyone the app is for, and swiping remains the path for them.

The declining trend *within* the favourites row is an artifact of the same
effect, not evidence that more input hurts: each extra revealed favourite
strips another of the best answers out of the target set. Compare across
policies at fixed k, never down a column.

### Near-duplicate recommendations

Liking The Red Dragon Inn returned Red Dragon Inn 2, 3, 4, 5, 6 and 7 — six of
ten slots. Every one is a correct prediction and the list is useless. nDCG
cannot see this; it rewards each of those hits.

**A similarity threshold cannot fix it**, which is worth recording because it
is the obvious first idea. On the real data:

| pair | similarity | verdict |
|---|---|---|
| Red Dragon Inn → RDI 2 | 0.339 | near-duplicate, cut it |
| Ticket to Ride → TTR: Europe | 0.293 | arguably the same |
| Codenames → Codenames: Duet | 0.255 | different game, keep it |
| Codenames → Codenames: Pictures | 0.193 | different game, keep it |

The cases interleave. Any cutoff removing RDI 2 also removes Codenames: Duet.

The `Family` column does work, because it encodes publisher intent rather than
statistical closeness — 5,694 of 17,140 games carry one, and it groups exactly
the series that cause trouble. Recommendation lists now allow one game per
family, and the swipe stream stops offering a series after two reactions to
it. Games without a family fall back to a deliberately loose similarity cap,
which exists to catch outright twins rather than make fine judgements.

Result on the real data:

- **Red Dragon Inn** → keeps RDI 2, promotes Munchkin, Fluxx, Guillotine,
  Exploding Kittens, BANG!, Smash Up, Betrayal at House on the Hill.
- **Codenames** → keeps Duet, drops Pictures, promotes Concordia.
- **Ticket to Ride** → keeps Europe, drops Nordic Countries and Märklin.

There is a judgement embedded here worth revisiting with real users: the cap
assumes you want the *best* member of a series and nothing else. Someone who
loves Red Dragon Inn may well want to know there are six more.

### A real session

Liking Go and then heavy strategy games walks the recommender from
Tigris & Euphrates through Through the Ages, Terra Mystica, Agricola and
Brass, ending on Great Western Trail, Gaia Project and Caverna. From a single
like on Go the top eight are Chess, Shogi, Xiangqi, YINSH, DVONN, ZÈRTZ and
TZAAR — the abstract-strategy shelf, in order.

## Phase 6 — Accounts, and shipping

**Export/import exists now**: a session downloads as JSON and re-uploads to
restore, so people can come back without re-swiping and without an account.
Restore resolves games by **BGGId, not by matrix index** — indices are
positions in one build of the catalogue, so rerunning `prepare` with different
thresholds would silently hand someone else's taste back.

Accounts are the next step up and are genuinely a different problem: storing
identifiable preference data means auth, a real database, and a privacy
position. The export file is worth having regardless, since it is also the
migration path into whatever account system arrives.

## Phase 7 — Ship

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
