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


def card(at) -> tuple[str, str]:
    """The title and reason of the card currently on screen."""
    for block in at.markdown:
        # Match the rendered card, not the stylesheet that merely names the
        # same CSS classes.
        if 'class="game-card"' in block.value:
            return (block.value.split('game-title">')[1].split("<")[0],
                    block.value.split('game-why">')[1].split("</div>")[0])
    raise AssertionError("no card on screen")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def test_the_app_starts_without_error(app):
    assert [t.value for t in app.title] == ["🎲 Tabled"]


def test_all_four_reactions_are_offered(app):
    labels = [b.label for b in app.button]
    for expected in ("👍 Like", "👎 Nope", "🤷 Skip", "✓ Played"):
        assert expected in labels


def test_the_opening_card_explains_itself_as_a_cold_start(app):
    _, why = card(app)
    assert "disagree" in why


def test_reacting_changes_the_card_and_produces_a_reason():
    at = AppTest.from_file(APP, default_timeout=120).run()
    first, _ = card(at)

    at.button[0].click().run()                      # Like
    second, why = card(at)

    assert second != first, "the same card came back after reacting"
    assert f"<b>{first}</b>" in why, (
        f"expected the reason to cite {first!r}, got {why!r}")


def test_recommendations_appear_only_once_there_is_taste_signal():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not [s for s in at.subheader if "top picks" in s.value]

    at.button[0].click().run()
    assert [s for s in at.subheader if "top picks" in s.value]


def test_a_skip_does_not_count_as_taste():
    """Skips must not produce recommendations — they say nothing about
    what the user likes, only that this card was not interesting."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button[2].click().run()                      # Skip

    assert not [s for s in at.subheader if "top picks" in s.value]
    assert not at.exception


def test_a_skipped_card_is_not_shown_again():
    at = AppTest.from_file(APP, default_timeout=120).run()
    skipped, _ = card(at)
    at.button[2].click().run()

    assert card(at)[0] != skipped


def test_undo_restores_the_previous_state():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button[0].click().run()
    assert [s for s in at.subheader if "top picks" in s.value]

    undo = next(b for b in at.button if b.label == "↩ Undo last")
    undo.click().run()

    assert not [s for s in at.subheader if "top picks" in s.value]
    assert not at.exception
