"""feedback module: pure badge state machine.

Public interface
----------------
feedback = Feedback()
feedback.turn_started()              # a turn has been enqueued
feedback.turn_finished(ok)           # a turn has resolved
feedback.expire()                    # the Outcome badge interval has elapsed
badge = feedback.current_badge()     # Badge | None

The state machine tracks a count of outstanding turns.  It never owns a
timer or a clock; expiry is driven by the caller (the HUD).  It also never
produces more than one badge at a time.
"""

from dataclasses import dataclass
from enum import Enum, auto


class BadgeKind(Enum):
    """The three possible badge states."""

    THINKING = auto()
    OUTCOME = auto()


@dataclass(frozen=True)
class Badge:
    """A badge the HUD should currently be showing.

    For ``BadgeKind.OUTCOME`` the ``success`` flag carries the result: ``True``
    for a completed turn, ``False`` for a failed one.  It is ``None`` for the
    ``THINKING`` badge.
    """

    kind: BadgeKind
    success: bool | None = None


class Feedback:
    """Pure state machine for HUD badge sequencing.

    Rules:
    - any outstanding turn produces ``THINKING``
    - a turn resolving produces ``OUTCOME`` with the result
    - ``expire()`` with turns still outstanding returns to ``THINKING``
    - ``expire()`` with no turns outstanding produces nothing
    - a new ``turn_started()`` clears a stale Outcome and immediately shows Thinking
    """

    def __init__(self) -> None:
        self._outstanding = 0
        self._outcome: Badge | None = None

    def turn_started(self) -> None:
        """A turn has been enqueued; a new turn means Thinking immediately."""
        self._outstanding += 1
        # A fresh turn is more salient than a lingering outcome.
        self._outcome = None

    def turn_finished(self, ok: bool) -> None:
        """A turn has resolved; show its Outcome and track the queue depth."""
        if self._outstanding > 0:
            self._outstanding -= 1
        self._outcome = Badge(BadgeKind.OUTCOME, success=ok)

    def expire(self) -> None:
        """The current Outcome badge has finished its display interval."""
        self._outcome = None

    def current_badge(self) -> Badge | None:
        """Return the badge that should currently be showing, or ``None``."""
        if self._outcome is not None:
            return self._outcome
        if self._outstanding > 0:
            return Badge(BadgeKind.THINKING)
        return None
