"""Tests for the feedback state machine (issue #14).

The feedback module is pure: it has no clock, no AppKit, and no I/O.  These
tests exercise its public seam directly: ``turn_started()``, ``turn_finished()``,
``expire()``, and ``current_badge()``.
"""

import pytest

from src.feedback import Feedback, Badge, BadgeKind


class TestSingleTurn:
    """One turn produces Thinking, then Outcome, then nothing."""

    def test_turn_started_shows_thinking(self):
        feedback = Feedback()

        feedback.turn_started()

        badge = feedback.current_badge()
        assert badge is not None
        assert badge.kind is BadgeKind.THINKING

    def test_successful_turn_shows_green_outcome_then_expires(self):
        feedback = Feedback()

        feedback.turn_started()
        feedback.turn_finished(True)

        badge = feedback.current_badge()
        assert badge is not None
        assert badge.kind is BadgeKind.OUTCOME
        assert badge.success is True

        feedback.expire()

        assert feedback.current_badge() is None

    def test_failed_turn_shows_red_outcome_then_expires(self):
        feedback = Feedback()

        feedback.turn_started()
        feedback.turn_finished(False)

        badge = feedback.current_badge()
        assert badge is not None
        assert badge.kind is BadgeKind.OUTCOME
        assert badge.success is False

        feedback.expire()

        assert feedback.current_badge() is None


class TestOverlappingTurns:
    """With turns queued, the state stays Thinking and each finish gets its own Outcome."""

    def test_two_turns_stay_thinking_until_first_finishes(self):
        feedback = Feedback()

        feedback.turn_started()  # A
        feedback.turn_started()  # B

        assert feedback.current_badge().kind is BadgeKind.THINKING

        feedback.turn_finished(True)  # A resolves, B still running

        badge = feedback.current_badge()
        assert badge.kind is BadgeKind.OUTCOME
        assert badge.success is True

        # B is still outstanding, so the badge returns to Thinking after expiry.
        feedback.expire()

        assert feedback.current_badge().kind is BadgeKind.THINKING

    def test_two_turns_show_two_outcomes(self):
        feedback = Feedback()

        feedback.turn_started()
        feedback.turn_started()

        feedback.turn_finished(True)
        feedback.expire()

        feedback.turn_finished(False)

        badge = feedback.current_badge()
        assert badge.kind is BadgeKind.OUTCOME
        assert badge.success is False

        feedback.expire()

        assert feedback.current_badge() is None

    def test_new_turn_while_outcome_showing_returns_to_thinking(self):
        feedback = Feedback()

        feedback.turn_started()
        feedback.turn_finished(True)
        assert feedback.current_badge().kind is BadgeKind.OUTCOME

        feedback.turn_started()  # a new turn is more important than the lingering outcome

        badge = feedback.current_badge()
        assert badge.kind is BadgeKind.THINKING

        feedback.turn_finished(True)
        assert feedback.current_badge().kind is BadgeKind.OUTCOME


class TestQueueFailure:
    """A failed turn shows red and the remaining queue continues."""

    def test_mid_queue_failure_shows_red_then_continues(self):
        feedback = Feedback()

        feedback.turn_started()  # A
        feedback.turn_started()  # B

        feedback.turn_finished(False)  # A fails, B still running

        assert feedback.current_badge().success is False

        feedback.expire()

        assert feedback.current_badge().kind is BadgeKind.THINKING

        feedback.turn_finished(True)  # B succeeds

        assert feedback.current_badge().success is True

        feedback.expire()

        assert feedback.current_badge() is None


class TestExpiry:
    """Expiry without outstanding turns clears the badge; with outstanding turns it
    returns to Thinking."""

    def test_expire_with_empty_queue_yields_nothing(self):
        feedback = Feedback()

        feedback.turn_started()
        feedback.turn_finished(True)
        feedback.expire()

        assert feedback.current_badge() is None

    def test_expire_with_turns_still_outstanding_yields_thinking(self):
        feedback = Feedback()

        feedback.turn_started()
        feedback.turn_started()
        feedback.turn_finished(True)

        feedback.expire()

        assert feedback.current_badge().kind is BadgeKind.THINKING


class TestInterleavings:
    """No sequence of starts, finishes, and expiries ever produces more than one badge."""

    @pytest.mark.parametrize(
        "actions,expected",
        [
            (["start"], BadgeKind.THINKING),
            (["start", "finish"], BadgeKind.OUTCOME),
            (["start", "finish", "expire"], None),
            (["start", "finish", "start"], BadgeKind.THINKING),
            (["start", "start", "finish"], BadgeKind.OUTCOME),
            (["start", "start", "finish", "expire"], BadgeKind.THINKING),
            (["start", "start", "finish", "finish", "expire"], None),
            (
                ["start", "start", "finish", "expire", "finish", "expire"],
                None,
            ),
        ],
    )
    def test_each_interleaving_yields_exactly_one_badge(self, actions, expected):
        feedback = Feedback()
        result_map = {
            "start": feedback.turn_started,
            "finish": lambda: feedback.turn_finished(True),
            "expire": feedback.expire,
        }

        for action in actions:
            result_map[action]()

        badge = feedback.current_badge()
        if expected is None:
            assert badge is None
        else:
            assert badge is not None
            assert badge.kind is expected
