"""Tests for the HUD module (issue #12 null implementation).

The null HUD exists so the app can hand the main thread to AppKit without
building a real panel.  Its public behaviour — run, stop, and the turn
lifecycle signals — is tested here.  The real AppKit path is exercised
separately; in unit tests we force the fallback so the test process never
blocks on an actual run loop.
"""

import threading
import time

import AppKit  # type: ignore[import-untyped]
import pytest

from src.feedback import Badge, BadgeKind
from src.hud import NullHud, HudPanel, badge_glyph
from src.display import Displays


class TestNullHudFallback:
    """NullHud run/stop lifecycle, forcing the no-AppKit fallback."""

    def test_run_blocks_until_stop_is_called(self):
        hud = NullHud()
        hud._appkit_available = lambda: False

        started = threading.Event()

        def run_in_thread():
            started.set()
            hud.run()

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()
        started.wait(timeout=1.0)

        assert t.is_alive()

        hud.stop()
        t.join(timeout=1.0)

        assert not t.is_alive()

    def test_stop_is_idempotent(self):
        hud = NullHud()
        hud._appkit_available = lambda: False

        hud.stop()
        hud.stop()

        # Should not raise and should still allow a subsequent run to return.
        t = threading.Thread(target=hud.run, daemon=True)
        t.start()
        time.sleep(0.05)
        hud.stop()
        t.join(timeout=1.0)

        assert not t.is_alive()

    def test_turn_started_and_finished_are_no_ops(self):
        hud = NullHud()

        # These must never raise, regardless of run state.
        hud.turn_started()
        hud.turn_finished(True)
        hud.turn_finished(False)


# ---------------------------------------------------------------------------
# Badge-to-glyph mapping
# ---------------------------------------------------------------------------


class TestBadgeGlyph:
    """The badge-to-glyph mapping is a pure function with no AppKit fake."""

    def test_thinking_badge_maps_to_orange_dot(self):
        assert badge_glyph(Badge(BadgeKind.THINKING)) == "\N{LARGE ORANGE CIRCLE}"

    def test_successful_outcome_maps_to_green_dot(self):
        assert badge_glyph(Badge(BadgeKind.OUTCOME, success=True)) == "\N{LARGE GREEN CIRCLE}"

    def test_failed_outcome_maps_to_red_dot(self):
        assert badge_glyph(Badge(BadgeKind.OUTCOME, success=False)) == "\N{LARGE RED CIRCLE}"


# ---------------------------------------------------------------------------
# Fakes for the AppKit integration tests
# ---------------------------------------------------------------------------


class FakeApplication:
    def __init__(self):
        self.activation_policy = None

    def setActivationPolicy_(self, policy):
        self.activation_policy = policy


class FakeNSApplication:
    _app = FakeApplication()

    @classmethod
    def sharedApplication(cls):
        return cls._app


class _FakePanelAlloc:
    def __init__(self, fake_class):
        self._fake_class = fake_class

    def initWithContentRect_styleMask_backing_defer_(
        self, rect, style_mask, backing, defer
    ):
        panel = self._fake_class()
        panel.rect = rect
        panel.style_mask = style_mask
        panel.backing = backing
        panel.defer = defer
        self._fake_class.instances.append(panel)
        return panel


class FakeNSPanel:
    instances: list["FakeNSPanel"] = []

    def __init__(self):
        self.level = None
        self.collection_behavior = None
        self.sharing_type = None
        self.ignores_mouse = None
        self.floating = None
        self.becomes_key_only = None
        self.hides_on_deactivate = None
        self.has_shadow = None
        self.opaque = None
        self.background_color = None
        self.content_view = None
        self.is_visible = False
        self.is_closed = False

    @classmethod
    def alloc(cls):
        return _FakePanelAlloc(cls)

    def setLevel_(self, level):
        self.level = level

    def setCollectionBehavior_(self, behavior):
        self.collection_behavior = behavior

    def setSharingType_(self, sharing_type):
        self.sharing_type = sharing_type

    def setIgnoresMouseEvents_(self, flag):
        self.ignores_mouse = flag

    def setFloatingPanel_(self, flag):
        self.floating = flag

    def setBecomesKeyOnlyIfNeeded_(self, flag):
        self.becomes_key_only = flag

    def setHidesOnDeactivate_(self, flag):
        self.hides_on_deactivate = flag

    def setHasShadow_(self, flag):
        self.has_shadow = flag

    def setOpaque_(self, flag):
        self.opaque = flag

    def setBackgroundColor_(self, color):
        self.background_color = color

    def setContentView_(self, view):
        self.content_view = view

    def orderFront_(self, sender):
        self.is_visible = True

    def orderOut_(self, sender):
        self.is_visible = False

    def close(self):
        self.is_closed = True
        self.is_visible = False

class FakeLabel:
    def __init__(self, rect):
        self.rect = rect
        self.string = None
        self.bezeled = None
        self.draws_background = None
        self.background_color = None
        self.text_color = None
        self.font = None
        self.alignment = None
        self.editable = None
        self.selectable = None
        self.bordered = None

    def setStringValue_(self, value):
        self.string = value

    def setBezeled_(self, flag):
        self.bezeled = flag

    def setDrawsBackground_(self, flag):
        self.draws_background = flag

    def setBackgroundColor_(self, color):
        self.background_color = color

    def setTextColor_(self, color):
        self.text_color = color

    def setFont_(self, font):
        self.font = font

    def setAlignment_(self, alignment):
        self.alignment = alignment

    def setEditable_(self, flag):
        self.editable = flag

    def setSelectable_(self, flag):
        self.selectable = flag

    def setBordered_(self, flag):
        self.bordered = flag


class _FakeLabelAlloc:
    def initWithFrame_(self, rect):
        return FakeLabel(rect)


class FakeNSTextField:
    @classmethod
    def alloc(cls):
        return _FakeLabelAlloc()


class FakeRunLoop:
    calls = 0

    def __init__(self):
        self.mode = None
        self.date = None

    def runMode_beforeDate_(self, mode, date):
        self.mode = mode
        self.date = date
        FakeRunLoop.calls += 1
        time.sleep(0.001)
        return True


class FakeNSRunLoop:
    @classmethod
    def currentRunLoop(cls):
        return FakeRunLoop()


class FakeTimer:
    instances: list["FakeTimer"] = []

    def __init__(self):
        self.interval = None
        self.target = None
        self.selector = None
        self.user_info = None
        self.repeats = None
        self.invalidated = False

    @classmethod
    def scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        cls, interval, target, selector, user_info, repeats
    ):
        timer = cls()
        timer.interval = interval
        timer.target = target
        timer.selector = selector
        timer.user_info = user_info
        timer.repeats = repeats
        cls.instances.append(timer)
        return timer

    def invalidate(self):
        self.invalidated = True

    def fire(self):
        """Simulate a one-shot timer firing on the run loop."""
        if self.target is None or self.selector is None:
            return
        method_name = self.selector.replace(":", "_")
        getattr(self.target, method_name)(self.user_info)


class FakeNSTimer:
    @classmethod
    def scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        cls, interval, target, selector, user_info, repeats
    ):
        return FakeTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, target, selector, user_info, repeats
        )


class FakeNSDate:
    @classmethod
    def distantFuture(cls):
        return None


class FakeDisplayBackend:
    def __init__(self, monitors):
        self.monitors = monitors

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def make_hud(display_index=1):
    monitors = [
        {"left": 0, "top": 0, "width": 2880, "height": 900},
        {"left": 0, "top": 0, "width": 1440, "height": 900},
        {"left": 1440, "top": 0, "width": 1440, "height": 900},
    ]
    return HudPanel(
        displays=Displays(backend_factory=lambda: FakeDisplayBackend(monitors)),
        display_index=display_index,
    )


# ---------------------------------------------------------------------------
# AppKit HUD panel tests
# ---------------------------------------------------------------------------


class TestHudPanel:
    """HudPanel creates an AppKit panel and renders Thinking and Outcome badges."""

    def setup_method(self):
        FakeNSPanel.instances.clear()
        FakeTimer.instances.clear()
        FakeRunLoop.calls = 0

    def test_turn_started_creates_and_shows_thinking_badge(self, monkeypatch):
        monkeypatch.setattr("src.hud.NSPanel", FakeNSPanel)
        monkeypatch.setattr("src.hud.NSTextField", FakeNSTextField)

        hud = make_hud()
        hud.turn_started()

        assert len(FakeNSPanel.instances) >= 1
        panel = FakeNSPanel.instances[-1]

        assert panel.is_visible is True
        assert panel.content_view is not None
        assert panel.content_view.string == "\N{LARGE ORANGE CIRCLE}"
        assert panel.rect.origin.x == 1396.0
        assert panel.rect.origin.y == 12.0

        assert panel.style_mask == (
            AppKit.NSBorderlessWindowMask | AppKit.NSNonactivatingPanelMask
        )
        assert panel.level == AppKit.NSStatusWindowLevel
        assert panel.collection_behavior == (
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            | AppKit.NSWindowCollectionBehaviorCanJoinAllApplications
        )
        assert panel.sharing_type == AppKit.NSWindowSharingNone
        assert panel.ignores_mouse is True
        assert panel.floating is True
        assert panel.becomes_key_only is True
        assert panel.hides_on_deactivate is False
        assert panel.has_shadow is False
        assert panel.opaque is False
        assert panel.background_color is AppKit.NSColor.clearColor()

        # The dot lives in a small square panel with the glyph filling it.
        assert panel.rect.size.width == panel.rect.size.height
        label = panel.content_view
        assert label.rect.size.width == panel.rect.size.width
        assert label.rect.size.height == panel.rect.size.height
        assert label.font is not None
        assert label.font.pointSize() >= 18.0
        assert label.draws_background is False
        assert label.editable is False
        assert label.selectable is False
        assert label.bordered is False
        assert label.alignment == AppKit.NSCenterTextAlignment

    def test_turn_finished_success_shows_green_dot(self, monkeypatch):
        monkeypatch.setattr("src.hud.NSPanel", FakeNSPanel)
        monkeypatch.setattr("src.hud.NSTextField", FakeNSTextField)

        hud = make_hud()
        hud.turn_started()
        panel = FakeNSPanel.instances[-1]

        hud.turn_finished(True)

        assert panel.is_visible is True
        assert panel.content_view.string == "\N{LARGE GREEN CIRCLE}"

    def test_turn_finished_failure_shows_red_dot(self, monkeypatch):
        monkeypatch.setattr("src.hud.NSPanel", FakeNSPanel)
        monkeypatch.setattr("src.hud.NSTextField", FakeNSTextField)

        hud = make_hud()
        hud.turn_started()
        panel = FakeNSPanel.instances[-1]

        hud.turn_finished(False)

        assert panel.is_visible is True
        assert panel.content_view.string == "\N{LARGE RED CIRCLE}"

    def test_outcome_badge_dismisses_after_interval(self, monkeypatch):
        monkeypatch.setattr("src.hud.NSPanel", FakeNSPanel)
        monkeypatch.setattr("src.hud.NSTextField", FakeNSTextField)
        monkeypatch.setattr("src.hud.NSTimer", FakeNSTimer)

        hud = make_hud()
        hud.turn_started()
        panel = FakeNSPanel.instances[-1]

        hud.turn_finished(True)

        assert panel.is_visible is True
        assert panel.content_view.string == "\N{LARGE GREEN CIRCLE}"

        assert len(FakeTimer.instances) == 1
        timer = FakeTimer.instances[-1]
        assert timer.selector == "expire:"
        assert timer.repeats is False
        assert timer.interval == 1.5

        timer.fire()

        assert panel.is_visible is False

    def test_outcome_badge_returns_to_thinking_when_queue_not_empty(self, monkeypatch):
        monkeypatch.setattr("src.hud.NSPanel", FakeNSPanel)
        monkeypatch.setattr("src.hud.NSTextField", FakeNSTextField)
        monkeypatch.setattr("src.hud.NSTimer", FakeNSTimer)

        hud = make_hud()

        # Two turns are started; the first finishes while the second is still queued.
        hud.turn_started()
        hud.turn_started()
        panel = FakeNSPanel.instances[-1]

        hud.turn_finished(True)
        assert panel.content_view.string == "\N{LARGE GREEN CIRCLE}"

        first_timer = FakeTimer.instances[-1]
        first_timer.fire()

        assert panel.is_visible is True
        assert panel.content_view.string == "\N{LARGE ORANGE CIRCLE}"

        # The second turn finishes and its Outcome badge dismisses, leaving nothing.
        hud.turn_finished(False)
        assert panel.content_view.string == "\N{LARGE RED CIRCLE}"

        second_timer = FakeTimer.instances[-1]
        second_timer.fire()

        assert panel.is_visible is False

    def test_run_closes_panel_on_stop(self, monkeypatch):
        monkeypatch.setattr("src.hud.NSPanel", FakeNSPanel)
        monkeypatch.setattr("src.hud.NSTextField", FakeNSTextField)
        monkeypatch.setattr("src.hud.NSApplication", FakeNSApplication)
        monkeypatch.setattr("src.hud.NSRunLoop", FakeNSRunLoop)
        monkeypatch.setattr("src.hud.NSTimer", FakeNSTimer)
        monkeypatch.setattr("src.hud.NSDate", FakeNSDate)

        hud = make_hud()
        hud.turn_started()
        panel = FakeNSPanel.instances[-1]
        assert panel.is_closed is False

        threading.Timer(0.05, hud.stop).start()
        hud.run()

        assert panel.is_closed is True

    def test_run_on_main_thread_uses_appkit_run_loop(self, monkeypatch):
        monkeypatch.setattr("src.hud.NSApplication", FakeNSApplication)
        monkeypatch.setattr("src.hud.NSRunLoop", FakeNSRunLoop)
        monkeypatch.setattr("src.hud.NSTimer", FakeNSTimer)
        monkeypatch.setattr("src.hud.NSDate", FakeNSDate)

        hud = make_hud()
        threading.Timer(0.05, hud.stop).start()
        hud.run()

        assert FakeNSApplication._app.activation_policy == AppKit.NSApplicationActivationPolicyAccessory
        assert FakeRunLoop.calls > 0

        timer = FakeTimer.instances[-1]
        assert timer.target is not None
        assert timer.selector == "tick:"
        assert timer.repeats is True
        assert timer.interval == 0.2
        assert timer.invalidated is True

    def test_run_on_non_main_thread_uses_fallback(self, monkeypatch):
        hud = make_hud()
        started = threading.Event()

        def run_in_thread():
            started.set()
            hud.run()

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()
        started.wait(timeout=1.0)

        assert t.is_alive()

        hud.stop()
        t.join(timeout=1.0)

        assert not t.is_alive()

    def test_run_degrades_to_fallback_when_appkit_fails(self, monkeypatch):
        class ExplodingNSApplication:
            @classmethod
            def sharedApplication(cls):
                raise RuntimeError("AppKit unavailable")

        monkeypatch.setattr("src.hud.NSApplication", ExplodingNSApplication)

        hud = make_hud()
        threading.Timer(0.05, hud.stop).start()
        hud.run()

        # The run() method returns without raising; the test process would have
        # hung if it had blocked on a real run loop.

    def test_turn_started_and_finished_are_safe_before_run(self, monkeypatch):
        monkeypatch.setattr("src.hud.NSPanel", FakeNSPanel)
        monkeypatch.setattr("src.hud.NSTextField", FakeNSTextField)
        monkeypatch.setattr("src.hud.NSTimer", FakeNSTimer)

        hud = make_hud()
        # turn_started/turn_finished may be called before run() starts.
        hud.turn_started()
        hud.turn_finished(True)

        # The panel shows the green Outcome badge and, once its timer fires,
        # dismisses itself — all without a real run loop running.
        panel = FakeNSPanel.instances[-1]
        assert panel.is_visible is True
        assert panel.content_view.string == "\N{LARGE GREEN CIRCLE}"

        timer = FakeTimer.instances[-1]
        timer.fire()

        assert panel.is_visible is False
