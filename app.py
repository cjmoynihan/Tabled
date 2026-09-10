"""
Tabled — the swipe interface.

    streamlit run app.py

Lives outside the package because it is a presentation shell, not library
code: every decision worth testing is in `tabled.serve`, and this file only
wires it to widgets. That split is why the session logic can be tested without
a browser.

Artifacts are loaded once and cached for the process. Nothing here fits
anything — item-item takes minutes to build, so a request that triggered a fit
would simply time out.
"""

from __future__ import annotations

import base64

import numpy as np
import streamlit as st

from tabled import config
from tabled.data import prepare
from tabled.models import store
from tabled.serve import labels, policy, profile
from tabled.serve.session import Reaction, Session

st.set_page_config(page_title="Tabled", page_icon="🎲", layout="centered")

CARD_CSS = """
<style>
  .game-card { border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
               padding: 1.1rem 1.3rem; margin-bottom: .8rem; }
  .game-title { font-size: 1.55rem; font-weight: 650; line-height: 1.2; }
  .game-meta { opacity: .7; font-size: .9rem; margin-top: .35rem; }
  .game-why { font-size: .92rem; margin-top: .7rem; opacity: .85; }
  /* The picks grid. Every card is the same height so the action buttons
     underneath line up across a row: a long title or a long reason would
     otherwise push one column's buttons below its neighbours', and the row
     would look broken. The cover sits in a fixed box and is letterboxed
     rather than cropped, since box art is not a consistent aspect ratio. */
  .pick { min-height: 232px; display: flex; flex-direction: column; }
  .pick-art { height: 130px; display: flex; align-items: center;
              justify-content: center; }
  .pick-art img { max-height: 130px; max-width: 100%; object-fit: contain;
                  border-radius: 8px; }
  .pick-name { font-weight: 600; line-height: 1.25; margin-top: .4rem;
               font-size: .92rem; }
  .pick-meta { opacity: .65; font-size: .78rem; }
  .pick-why  { opacity: .55; font-size: .74rem; margin-top: .15rem; }

  .stButton button { width: 100%; }

  .bgg-credit { text-align: center; opacity: .75; margin-top: 2.5rem; }
  .bgg-credit img { max-width: 160px; }

  /* The four swipe actions are the primary control in the whole app, so they
     get roughly half again the default size. Scoped to the swipe container:
     the per-pick buttons in the results grid sit three-to-a-column and would
     wrap badly at this size. */
  .st-key-swipe-actions .stButton button {
      min-height: 3.6rem;
      font-size: 1.15rem;
      font-weight: 600;
      padding: .8rem .4rem;
  }
  .st-key-swipe-actions [data-testid="stHorizontalBlock"] { gap: .5rem; }
</style>
"""

PLACEHOLDER = "https://placehold.co/200x200?text=no+cover"


@st.cache_resource(show_spinner="Loading the catalogue…")
def load():
    """
    Load only what serving needs.

    Notably not the ratings matrix: it exists to fit models, and the one thing
    the app used it for is now a column. Skipping it takes the process from
    352 MB to 191 MB and removes a 146 MB read from every cold start, which is
    most of what a visitor waits for when the app wakes from sleep.
    """
    ds = prepare.load_catalogue()
    try:
        model = store.load_item_item(config.ITEM_ITEM_NPZ, ds.shape[1])
    except FileNotFoundError:
        st.error("No fitted model found. Run `tabled fit --model item-item` "
                 "first, then reload this page.")
        st.stop()
    return (ds, model, policy.cold_start_seeds(ds, n=12),
            labels.display_names(ds.games))


def cover(row) -> str:
    path = row.get("ImagePath")
    return path if isinstance(path, str) and path.startswith("http") else PLACEHOLDER


def state() -> Session:
    if "session" not in st.session_state:
        st.session_state.session = Session()
        st.session_state.rng = np.random.default_rng()
        st.session_state.card = None
        st.session_state.seeded = False
    return st.session_state.session


def react(session: Session, game: int, reaction: Reaction) -> None:
    session.record(game, reaction)
    st.session_state.card = None          # force a fresh pick next run
    st.rerun()


# --------------------------------------------------------------------------
# opening: name a few favourites
# --------------------------------------------------------------------------

def seed_picker(ds, session: Session, names) -> None:
    """
    An optional head start.

    Worth its own screen because the harness says so: three games named
    outright score better than ten swipes, and against a harder target set,
    since naming your favourites removes your best games from what is left to
    find. It stays optional, because it only works for people who already know
    what they like, which is not everyone the app is for.
    """
    st.subheader("Start with a few games you love")
    st.caption("Optional, and worth doing. Naming three games gets you "
               "further than ten swipes. Skip it if you would rather browse.")

    choices = st.multiselect(
        "Search for games you like", options=ds.games["game_index"].tolist(),
        format_func=lambda i: names[i],
        key="seed_choices", max_selections=10,
        placeholder="Type a game name…")

    left, right = st.columns(2)
    if left.button("Start swiping →", type="primary",
                   disabled=not choices):
        for game in choices:
            session.record(int(game), Reaction.LIKE)
        st.session_state.seeded = True
        st.session_state.card = None
        st.rerun()
    if right.button("Skip this, just show me games"):
        st.session_state.seeded = True
        st.rerun()


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------

def show_card(ds, model, session, game: int, names) -> None:
    row = ds.games.loc[game]

    players = f"{int(row['MinPlayers'])} to {int(row['MaxPlayers'])} players"
    if row["MinPlayers"] == row["MaxPlayers"]:
        players = f"{int(row['MinPlayers'])} players"
    year = int(row["YearPublished"])
    meta = (f"{year if year > 0 else 'ancient'} &middot; {players} "
            f"&middot; {int(row['ComMinPlaytime'])} to "
            f"{int(row['ComMaxPlaytime'])} min "
            f"&middot; weight {row['GameWeight']:.1f}/5")

    why = policy.explain(model, session.swipes(), game, ds.games, names)
    why_html = (f'<div class="game-why">Because you liked <b>{why}</b></div>'
                if why else
                '<div class="game-why">A famous game that people disagree '
                'about. A good way to start.</div>')

    st.image(cover(row), width=240)
    st.markdown(
        f'<div class="game-card"><div class="game-title">{names[game]}</div>'
        f'<div class="game-meta">{meta}</div>{why_html}</div>',
        unsafe_allow_html=True)

    with st.container(key="swipe-actions"):
        a, b, c, d = st.columns(4, gap="small")
        if a.button("👍 Like", key=f"l{game}"):
            react(session, game, Reaction.LIKE)
        if b.button("👎 Nope", key=f"d{game}"):
            react(session, game, Reaction.DISLIKE)
        if c.button("🤷 Skip", key=f"s{game}"):
            react(session, game, Reaction.SKIP)
        if d.button("✓ Played", key=f"p{game}"):
            react(session, game, Reaction.PLAYED)

    st.caption("Only **Like** and **Nope** shape recommendations. **Skip** "
               "moves on without judging, and skipped games can still turn up "
               "in your top 10. **Played** takes it off the list for good.")


# --------------------------------------------------------------------------
# the picks
# --------------------------------------------------------------------------

def show_picks(ds, model, session, names, columns: int = 5) -> None:
    swipes = session.swipes()
    if len(swipes) == 0:
        st.info("Like or pass on a few games and your top ten will appear "
                "here.")
        return

    picks = policy.top_picks(model, swipes, ds, session.judged, n=10)
    if not len(picks):
        st.info("Nothing left to suggest. You have reacted to everything we "
                "would recommend.")
        return

    st.caption("One game per series, so a run of sequels cannot fill the "
               "list. Games you skipped can appear here, since skipping meant "
               "you did not know it. React to any of these to swap it out.")

    for start in range(0, len(picks), columns):
        for column, game in zip(st.columns(columns), picks[start:start + columns]):
            row = ds.games.loc[game]
            why = policy.explain(model, swipes, int(game), ds.games, names)
            with column:
                # Cover, title, meta and reason go out as one fixed-height
                # block. Rendering them as separate Streamlit elements is what
                # let a two-line title shove this column's buttons a row
                # lower than its neighbours'.
                st.markdown(
                    f'<div class="pick">'
                    f'<div class="pick-art"><img src="{cover(row)}"></div>'
                    f'<div class="pick-name">{names[game]}</div>'
                    f'<div class="pick-meta">weight {row["GameWeight"]:.1f}'
                    f' &middot; BGG {row["AvgRating"]:.1f}</div>'
                    f'<div class="pick-why">{f"like {why}" if why else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

                x, y, z = st.columns(3)
                if x.button("👍", key=f"pl{game}", help="Like this"):
                    react(session, game, Reaction.LIKE)
                if y.button("👎", key=f"pd{game}", help="Not for me"):
                    react(session, game, Reaction.DISLIKE)
                if z.button("⊘", key=f"pp{game}",
                            help="Move past this one"):
                    react(session, game, Reaction.DISMISS)


# --------------------------------------------------------------------------

def taste_panel(ds, session: Session) -> None:
    """
    What we think this person likes, in their own terms.

    Worth more than the raw counts it replaces: a written description is a
    claim the user can immediately agree or disagree with, which a ranked list
    of ten titles is not. It is computed from what they actually said, never
    from the model's inferences, so it cannot simply agree with itself.
    """
    counts = session.counts()
    liked, passed = counts["like"], counts["dislike"]
    seen = len(session.events)

    a, b, c = st.columns(3)
    a.metric("Seen", seen)
    b.metric("Liked", liked)
    c.metric("Passed", passed)

    if liked or passed:
        share = liked / (liked + passed)
        st.progress(share, text=f"{share:.0%} of your calls were likes")

    summary = profile.summarise(ds.games, session.swipes().liked)
    if summary:
        st.markdown(profile.sentence(summary))
    elif liked < 3:
        st.caption(f"Like {3 - liked} more game"
                   f"{'s' if 3 - liked != 1 else ''} and a description of "
                   f"your taste will appear here.")


def sidebar(ds, session: Session) -> None:
    with st.sidebar:
        st.header("Your taste")
        taste_panel(ds, session)
        st.divider()

        if st.button("↩ Undo last") and session.undo():
            st.session_state.card = None
            st.rerun()
        if st.button("Start over"):
            st.session_state.clear()
            st.rerun()

        st.divider()
        st.caption("Take your taste with you. No account needed.")
        if session.events:
            st.download_button(
                "⬇ Export my session", session.to_json(ds.games),
                file_name="tabled-session.json", mime="application/json")

        uploaded = st.file_uploader("⬆ Import a session", type="json",
                                    key="import")
        if uploaded is not None and not st.session_state.get("imported"):
            st.session_state.session = Session.from_json(uploaded.getvalue(),
                                                         ds.games)
            st.session_state.imported = True
            st.session_state.card = None
            st.session_state.seeded = True
            st.rerun()

        if session.events:
            st.divider()
            if st.button("💾 Save to the swipe log"):
                path = session.append_to(games=ds.games)
                st.success(f"Appended to {config.display(path)}")
            st.caption("The log is the only record of what people do in this "
                       "interface, as opposed to what they rated on BGG "
                       "years ago.")


def bgg_credit() -> None:
    """
    Attribution, which using this data requires.

    Falls back to a text link when the logo file is absent, so a missing asset
    degrades to something still correct and still linked, rather than a broken
    image or, worse, no attribution at all.
    """
    # Preference order, best first. Alphabetical order would pick .jpeg over
    # .svg, which is backwards: JPEG cannot do transparency, so on the dark
    # theme it renders as a white box around the logo.
    logo = next((p for suffix in (".svg", ".png", ".jpg", ".jpeg")
                 for p in [config.ASSETS_DIR / f"powered-by-bgg{suffix}"]
                 if p.exists()), None)

    if logo is not None:
        encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
        mime = "svg+xml" if logo.suffix.lower() == ".svg" else "png"
        inner = (f'<img src="data:image/{mime};base64,{encoded}" '
                 f'alt="Powered by BoardGameGeek">')
    else:
        inner = "Powered by BoardGameGeek"

    st.markdown(
        f'<div class="bgg-credit">'
        f'<a href="https://boardgamegeek.com" target="_blank" '
        f'rel="noopener">{inner}</a></div>',
        unsafe_allow_html=True)


def main() -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    ds, model, seeds, names = load()
    session = state()

    st.title("🎲 Tabled")
    counts = session.counts()
    st.caption(f"{ds.shape[1]:,} games &middot; {counts['like']} liked, "
               f"{counts['dislike']} passed, {len(session.events)} seen",
               unsafe_allow_html=True)

    if not st.session_state.seeded and not session.events:
        seed_picker(ds, session, names)
        return

    swipe, picks = st.tabs(["Swipe", "Your top 10"])

    with swipe:
        if st.session_state.card is None:
            st.session_state.card = policy.next_card(
                model, session.swipes(), ds, session.shown,
                st.session_state.rng, seeds=seeds)
        game = st.session_state.card
        if game is None:
            st.success("You have been through everything we can suggest.")
        else:
            show_card(ds, model, session, game, names)

    with picks:
        show_picks(ds, model, session, names)

    sidebar(ds, session)
    bgg_credit()


if __name__ == "__main__":
    main()
