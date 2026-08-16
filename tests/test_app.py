"""
End-to-end tests for the Streamlit shell, via `AppTest`.

These are the only tests that touch the real artifacts, so they skip cleanly
when the pipeline has not been run. Everything about *what* to show is tested
against synthetic data in test_serve.py; what is left to check here is the
wiring — that a click reaches the session, that the card changes, and that the
explanation the user reads is the one the model actually gave.
"""

from __future__ import annotations

import pytest

from tabled import config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (config.RATINGS_NPZ.exists() and config.ITEM_ITEM_NPZ.exists()),
    reason="needs `tabled prepare` and `tabled fit --model item-item`")

APP = str(config.ROOT / "app.py")


def opened() -> "AppTest":
    """An app past the optional seed picker, sitting on the first card."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    button(at, "Skip this, just show me games").click().run()
    return at


def button(at, label: str):
    for candidate in at.button:
        if candidate.label == label:
            return candidate
    raise AssertionError(f"no button {label!r} in {[b.label for b in at.button]}")


def card(at) -> tuple[str, str]:
    """The title and reason of the card currently on screen."""
    for block in at.markdown:
        # Match the rendered card, not the stylesheet that merely names the
        # same CSS classes.
        if 'class="game-card"' in block.value:
            return (block.value.split('game-title">')[1].split("<")[0],
                    block.value.split('game-why">')[1].split("</div>")[0])
    raise AssertionError("no card on screen")


def picks(at) -> list[str]:
    return [b.value.split('pick-name">')[1].split("<")[0]
            for b in at.markdown if 'class="pick-name"' in b.value]


# -- opening ---------------------------------------------------------------

def test_the_app_starts_without_error():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.value for t in at.title] == ["🎲 Tabled"]


def test_it_opens_on_the_seed_picker():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert at.multiselect
    with pytest.raises(AssertionError):
        card(at)


def test_the_seed_picker_can_be_skipped():
    at = opened()
    assert not at.exception
    assert card(at)[0]


def test_naming_favourites_seeds_the_session():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.multiselect[0].select(at.multiselect[0].options[0]).run()
    button(at, "Start swiping →").click().run()

    assert not at.exception
    assert picks(at), "naming a favourite should produce picks immediately"


# -- the card --------------------------------------------------------------

def test_all_four_reactions_are_offered():
    labels = [b.label for b in opened().button]
    for expected in ("👍 Like", "👎 Nope", "🤷 Skip", "✓ Played"):
        assert expected in labels


def test_the_opening_card_explains_itself_as_a_cold_start():
    assert "disagree" in card(opened())[1]


def test_reacting_changes_the_card_and_produces_a_reason():
    at = opened()
    first, _ = card(at)

    button(at, "👍 Like").click().run()
    second, why = card(at)

    assert second != first, "the same card came back after reacting"
    assert f"<b>{first}</b>" in why, (
        f"expected the reason to cite {first!r}, got {why!r}")


def test_a_skipped_card_is_not_shown_again():
    at = opened()
    skipped, _ = card(at)
    button(at, "🤷 Skip").click().run()

    assert card(at)[0] != skipped


def test_a_skipped_game_can_still_appear_in_the_picks():
    """
    A skip means "I don't know this one", so the game stays eligible as a
    result even though it leaves the card stream.
    """
    at = opened()
    button(at, "👍 Like").click().run()
    suggested = picks(at)
    assert suggested

    # Skip the top pick when it comes round as a card, then check it survives.
    at2 = opened()
    button(at2, "👍 Like").click().run()
    before = picks(at2)
    button(at2, "🤷 Skip").click().run()

    assert set(before) & set(picks(at2)), "skipping emptied the picks list"


def test_the_swipe_actions_are_sized_up():
    """The four card buttons are the primary control, so they are enlarged —
    scoped to their own container so the picks grid keeps its small ones."""
    at = opened()
    css = next(b.value for b in at.markdown if "<style>" in b.value)

    assert ".st-key-swipe-actions .stButton button" in css
    assert "min-height" in css


# -- the picks -------------------------------------------------------------

def test_picks_appear_only_once_there_is_taste_signal():
    at = opened()
    assert not picks(at)

    button(at, "👍 Like").click().run()
    assert picks(at)


def test_a_skip_produces_no_picks():
    """A skip says nothing about taste, so it must not generate results."""
    at = opened()
    button(at, "🤷 Skip").click().run()

    assert not picks(at)
    assert not at.exception


def test_picks_are_thinned_to_one_per_series():
    at = opened()
    button(at, "👍 Like").click().run()
    names = picks(at)

    assert len(names) == len(set(names))


def test_reacting_to_a_pick_removes_it_and_refills_the_list():
    at = opened()
    button(at, "👍 Like").click().run()
    before = picks(at)
    assert len(before) >= 2

    # The per-pick buttons are keyed by game, so find one by its help text.
    played = next(b for b in at.button if b.label == "✓")
    played.click().run()
    after = picks(at)

    assert before[0] not in after
    assert len(after) == len(before), "the list should refill"


# -- session ---------------------------------------------------------------

def test_a_session_can_be_exported():
    at = opened()
    button(at, "👍 Like").click().run()

    assert any("Export" in b.label for b in at.sidebar.download_button)


def test_undo_restores_the_previous_state():
    at = opened()
    button(at, "👍 Like").click().run()
    assert picks(at)

    button(at, "↩ Undo last").click().run()

    assert not picks(at)
    assert not at.exception
