"""
Tests for the anti-shrug mechanic.

A user who does not know many games skips repeatedly, and a run of skips is
demoralising: nothing happens, and every card is a title they have never heard
of. `skip_weight` detects that and pulls the catalogue back towards games they
will recognise, trading information per answer for getting any answer at all.

The tests cover the arithmetic, the fact that it is derived from history
rather than accumulated, and the behaviour that actually matters: skipping a
lot must visibly change what comes next.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabled.models.item_item import ItemItem
from tabled.serve import policy
from tabled.serve.session import SKIP_FALL, SKIP_RISE, Reaction, Session

CORE = 20


@pytest.fixture
def model(clustered):
    return ItemItem(k=20, shrinkage=1.0).fit(clustered)


# -- the arithmetic --------------------------------------------------------

def test_it_starts_at_zero():
    assert Session().skip_weight == 0.0


def test_each_skip_closes_half_the_remaining_gap():
    s = Session()
    s.record(1, Reaction.SKIP)
    assert s.skip_weight == pytest.approx(SKIP_RISE)

    s.record(2, Reaction.SKIP)
    assert s.skip_weight == pytest.approx(SKIP_RISE + SKIP_RISE * (1 - SKIP_RISE))


def test_it_approaches_one_without_reaching_it():
    """Fractions of the remaining distance keep it in range with no clamp,
    which is why nothing here needs a min() or a max()."""
    s = Session()
    for game in range(40):
        s.record(game, Reaction.SKIP)

    assert 0.99 < s.skip_weight < 1.0


def test_an_opinion_knocks_most_of_it_off():
    s = Session()
    s.record(1, Reaction.SKIP)
    raised = s.skip_weight

    s.record(2, Reaction.LIKE)
    assert s.skip_weight == pytest.approx(raised * (1 - SKIP_FALL))


def test_recovery_is_faster_than_the_build_up():
    """One engaged answer should undo a short run of shrugs."""
    s = Session()
    for game in range(3):
        s.record(game, Reaction.SKIP)
    after_skips = s.skip_weight

    s.record(99, Reaction.DISLIKE)
    assert s.skip_weight < after_skips / 2


def test_dislikes_count_as_engagement_too():
    liked, disliked = Session(), Session()
    for s, reaction in ((liked, Reaction.LIKE), (disliked, Reaction.DISLIKE)):
        s.record(1, Reaction.SKIP)
        s.record(2, reaction)

    assert liked.skip_weight == pytest.approx(disliked.skip_weight)


def test_played_and_dismiss_leave_it_alone():
    """Neither is a rating, so neither moves the weight. Marking something as
    played does show recognition, which is a reasonable future refinement."""
    s = Session()
    s.record(1, Reaction.SKIP)
    before = s.skip_weight

    s.record(2, Reaction.PLAYED)
    s.record(3, Reaction.DISMISS)
    assert s.skip_weight == pytest.approx(before)


# -- derived, not accumulated ----------------------------------------------

def test_undo_rewinds_the_weight():
    s = Session()
    s.record(1, Reaction.SKIP)
    s.record(2, Reaction.SKIP)
    high = s.skip_weight

    s.undo()
    assert s.skip_weight < high
    assert s.skip_weight == pytest.approx(SKIP_RISE)


def test_an_imported_session_keeps_its_weight(clustered):
    original = Session()
    for game in range(3):
        original.record(game, Reaction.SKIP)

    restored = Session.from_json(original.to_json(clustered.games),
                                 clustered.games)
    assert restored.skip_weight == pytest.approx(original.skip_weight)


# -- what it actually does -------------------------------------------------

def test_the_gate_tightens_as_skipping_rises(clustered):
    popularity = clustered.popularity()
    gates = [policy.popularity_gate(10, popularity, w)
             for w in (0.0, 0.25, 0.5, 0.75, 1.0)]

    assert gates == sorted(gates), "more skipping should demand more fame"
    assert gates[-1] > gates[0]


def test_full_skip_weight_matches_the_opening_gate(clustered):
    """At the limit it asks for the same recognisability as the very first
    card, which is the most famous thing we ever show."""
    popularity = clustered.popularity()
    assert (policy.popularity_gate(50, popularity, 1.0)
            == pytest.approx(policy.popularity_gate(0, popularity, 0.0)))


def test_skipping_leads_to_more_familiar_games(clustered, model):
    """
    The behaviour the whole mechanic exists for.

    Compares the *same* session with and without the weight applied, rather
    than running two divergent sessions. Two sessions would mostly measure
    depletion: a skipper burns through the famous games and is then forced
    down the tail whatever the gate says, which says nothing about whether
    the gate did its job.
    """
    popularity = clustered.popularity()

    session = Session()
    session.record(CORE + 1, Reaction.LIKE)
    for game in range(CORE + 2, CORE + 12):
        session.record(game, Reaction.SKIP)
    assert session.skip_weight > 0.9

    def sample(weight):
        return np.array([
            popularity[policy.next_card(
                model, session.swipes(), clustered, session.shown,
                np.random.default_rng(seed), skip_weight=weight)]
            for seed in range(30)])

    assert np.median(sample(session.skip_weight)) > np.median(sample(0.0))


def test_an_engaged_user_is_unaffected(clustered, model):
    """Someone who never skips should see exactly what they saw before the
    mechanic existed."""
    session = Session()
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)

    session.record(CORE + 1, Reaction.LIKE)
    with_weight = policy.next_card(model, session.swipes(), clustered,
                                   session.shown, rng_a,
                                   skip_weight=session.skip_weight)
    without = policy.next_card(model, session.swipes(), clustered,
                               session.shown, rng_b, skip_weight=0.0)

    assert session.skip_weight == 0.0
    assert with_weight == without
