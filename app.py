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

import numpy as np
import streamlit as st

from tabled import config
from tabled.data import prepare
from tabled.models import store
from tabled.serve import policy
from tabled.serve.session import Reaction, Session

st.set_page_config(page_title="Tabled", page_icon="🎲", layout="centered")

CARD_CSS = """
<style>
  .game-card { border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
               padding: 1.1rem 1.3rem; margin-bottom: .8rem; }
  .game-title { font-size: 1.55rem; font-weight: 650; line-height: 1.2; }
  .game-meta { opacity: .7; font-size: .9rem; margin-top: .35rem; }
  .game-why { font-size: .92rem; margin-top: .7rem; opacity: .85; }
  .stButton button { width: 100%; }
</style>
"""


@st.cache_resource(show_spinner="Loading the catalogue…")
def load():
    ds = prepare.load()
    try:
        model = store.load_item_item(config.ITEM_ITEM_NPZ, ds.shape[1])
    except FileNotFoundError:
        st.error("No fitted model found. Run `tabled fit --model item-item` "
                 "first, then reload this page.")
        st.stop()
    return ds, model, policy.cold_start_seeds(ds, n=12)


def state(ds) -> Session:
    if "session" not in st.session_state:
        st.session_state.session = Session()
        st.session_state.rng = np.random.default_rng()
        st.session_state.card = None
    return st.session_state.session


def pick_card(model, session, ds, seeds) -> int | None:
    if st.session_state.card is None:
        st.session_state.card = policy.next_card(
            model, session.swipes(), ds, session.shown,
            st.session_state.rng, seeds=seeds)
    return st.session_state.card


def react(session: Session, game: int, reaction: Reaction) -> None:
    session.record(game, reaction)
    st.session_state.card = None          # force a fresh pick next run
    st.rerun()


def show_card(ds, model, session, game: int) -> None:
    row = ds.games.loc[game]

    players = f"{int(row['MinPlayers'])}–{int(row['MaxPlayers'])} players"
    if row["MinPlayers"] == row["MaxPlayers"]:
        players = f"{int(row['MinPlayers'])} players"
    year = int(row["YearPublished"])
    meta = (f"{year if year > 0 else 'ancient'} &middot; {players} "
            f"&middot; {int(row['ComMinPlaytime'])}–{int(row['ComMaxPlaytime'])} min "
            f"&middot; weight {row['GameWeight']:.1f}/5")

    why = policy.explain(model, session.swipes(), game, ds.games)
    why_html = (f'<div class="game-why">Because you liked '
                f'<b>{why}</b></div>' if why else
                '<div class="game-why">A well-known game people disagree '
                'about — a good way to start.</div>')

    if isinstance(row.get("ImagePath"), str) and row["ImagePath"].startswith("http"):
        st.image(row["ImagePath"], width=240)

    st.markdown(
        f'<div class="game-card"><div class="game-title">{row["Name"]}</div>'
        f'<div class="game-meta">{meta}</div>{why_html}</div>',
        unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    if a.button("👍 Like", key=f"l{game}"):
        react(session, game, Reaction.LIKE)
    if b.button("👎 Nope", key=f"d{game}"):
        react(session, game, Reaction.DISLIKE)
    if c.button("🤷 Skip", key=f"s{game}"):
        react(session, game, Reaction.SKIP)
    if d.button("✓ Played", key=f"p{game}"):
        react(session, game, Reaction.PLAYED)

    st.caption("**Skip** and **Played** are recorded but do not shape "
               "recommendations — only likes and dislikes say anything about "
               "taste.")


def show_recommendations(ds, model, session) -> None:
    swipes = session.swipes()
    if len(swipes) == 0:
        return

    st.subheader("Your top picks so far")
    for game in model.recommend(swipes, n=8, exclude=session.shown):
        row = ds.games.loc[game]
        why = policy.explain(model, swipes, game, ds.games)
        st.markdown(
            f"**{row['Name']}** &nbsp; <span style='opacity:.6'>"
            f"weight {row['GameWeight']:.1f} &middot; "
            f"BGG {row['AvgRating']:.1f}"
            + (f" &middot; like {why}" if why else "") +
            "</span>", unsafe_allow_html=True)


def main() -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    ds, model, seeds = load()
    session = state(ds)

    st.title("🎲 Tabled")
    counts = session.counts()
    st.caption(f"{ds.shape[1]:,} games &middot; {counts['like']} liked, "
               f"{counts['dislike']} passed, {len(session.events)} seen",
               unsafe_allow_html=True)

    game = pick_card(model, session, ds, seeds)
    if game is None:
        st.success("You have been through everything we can suggest.")
    else:
        show_card(ds, model, session, game)

    with st.sidebar:
        st.header("Session")
        st.write(counts)
        if st.button("↩ Undo last") and session.undo():
            st.session_state.card = None
            st.rerun()
        if st.button("Start over"):
            st.session_state.clear()
            st.rerun()
        if session.events and st.button("💾 Save this session"):
            path = session.append_to(games=ds.games)
            st.success(f"Appended to {config.display(path)}")
        st.caption("Saved sessions are the only record of what people did in "
                   "this interface, as opposed to what they rated on BGG "
                   "years ago.")

    show_recommendations(ds, model, session)


if __name__ == "__main__":
    main()
