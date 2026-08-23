"""HUD module: on-screen feedback badges.

Public interface
----------------
hud = NullHud()                    # no visible panel
hud.run()                          # blocks the main thread
hud.stop()                         # stops the run loop
hud.turn_started()                 # show the Thinking badge
hud.turn_finished(ok)              # hide the badge (single-turn slice)

hud = HudPanel(monitor_index=1)    # real AppKit panel
hud.run()                          # runs NSApplication on the main thread
hud.stop()                         # tear down

The panel is non-activating, floats above the menu bar, follows the user into
full-screen Spaces, ignores mouse events, and is excluded from mss screenshots
via NSWindowSharingNone.  If AppKit or the panel fails to initialise, the HUD
degrades to the non-AppKit fallback so the meeting continues.
"""

import threading
import weakref
from typing import Any, Mapping, Protocol

# mss gives the actual display the attendee chose for screenshots.  NSScreen may
# order displays differently, so we anchor the badge to the mss monitor.
try:
    import mss
    _MSS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _MSS_AVAILABLE = False
    mss = None  # type: ignore[assignment]

# Import AppKit/ObjC at module load if present.  The implementations must still
# be importable when they are absent, so any failure here is non-fatal.
try:
    import objc  # type: ignore[import-untyped]
    from AppKit import (  # type: ignore[import-untyped]
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSCenterTextAlignment,
        NSColor,
        NSFont,
        NSHUDWindowMask,
        NSMakeRect,
        NSNonactivatingPanelMask,
        NSPanel,
        NSStatusWindowLevel,
        NSTextField,
        NSWindowCollectionBehaviorCanJoinAllApplications,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowSharingNone,
    )
    from Foundation import (  # type: ignore[import-untyped]
        NSDate,
        NSDefaultRunLoopMode,
        NSObject,
        NSRunLoop,
        NSTimer,
    )
    _APPKIT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _APPKIT_AVAILABLE = False
    objc = None
    NSApplication = None
    NSBackingStoreBuffered = None
    NSCenterTextAlignment = None
    NSColor = None
    NSFont = None
    NSHUDWindowMask = None
    NSMakeRect = None
    NSNonactivatingPanelMask = None
    NSPanel = None
    NSStatusWindowLevel = None
    NSTextField = None
    NSWindowCollectionBehaviorCanJoinAllApplications = None
    NSWindowCollectionBehaviorCanJoinAllSpaces = None
    NSWindowCollectionBehaviorFullScreenAuxiliary = None
    NSWindowSharingNone = None
    NSDate = None
    NSDefaultRunLoopMode = None
    NSObject = None
    NSRunLoop = None
    NSTimer = None

# ---------------------------------------------------------------------------
# AppKit helpers
# ---------------------------------------------------------------------------

_Bridge: Any = None

if _APPKIT_AVAILABLE:
    class _BridgeImpl(NSObject):
        """Small dispatch target that lives on the main thread.

        Holds a weak reference back to the HudPanel so it can call the
        Python-level show/hide helpers.  Also serves as the repeating
        interpreter-pump timer target.
        """

        _hud_ref: Any = None

        def initWithHud_(self, hud: Any) -> Any:
            self = objc.super(_BridgeImpl, self).init()
            if self is None:
                return None
            self._hud_ref = weakref.ref(hud)  # type: ignore[attr-defined]
            return self

        def showThinking_(self, _obj: Any) -> None:
            hud = self._hud_ref()  # type: ignore[attr-defined]
            if hud is not None:
                hud._show_thinking()

        def hide_(self, _obj: Any) -> None:
            hud = self._hud_ref()  # type: ignore[attr-defined]
            if hud is not None:
                hud._hide()

        def tick_(self, _timer: Any) -> None:
            # Load-bearing: this timer hands control back to the Python
            # interpreter every ~0.2 s while the run loop is in a C call.
            # Without it, a SIGINT queued by Ctrl+C would be unable to run.
            pass

    _Bridge = _BridgeImpl


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class Hud(Protocol):
    """Collaborator interface for the on-screen feedback window."""

    def run(self) -> None:
        """Block the calling thread until stop() is called."""
        ...

    def stop(self) -> None:
        """Unblock the run() call."""
        ...

    def turn_started(self) -> None:
        """A turn has been enqueued; show the Thinking badge."""
        ...

    def turn_finished(self, ok: bool) -> None:
        """A turn has resolved; hide the badge in this slice."""
        ...


# ---------------------------------------------------------------------------
# Shared AppKit run-loop logic
# ---------------------------------------------------------------------------

class _BaseHud:
    """Base class for NullHud and HudPanel.

    Owns the stop event, AppKit activation policy, the interpreter-pump timer,
    and the fallback to a plain event wait.  Subclasses provide turn lifecycle
    and any run-loop cleanup.
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._app: Any | None = None
        self._pump_timer: Any | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block until stop() is called.

        On the main thread with AppKit present this sets the process to an
        accessory application and runs the main run loop.  Otherwise it falls
        back to a plain event wait.
        """
        if not self._appkit_available():
            return self._run_fallback()

        if threading.current_thread() is not threading.main_thread():
            return self._run_fallback()

        return self._run_appkit()

    def stop(self) -> None:
        """Signal the run loop to exit.  Safe to call multiple times."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _appkit_available(self) -> bool:
        """Return True when the AppKit/ObjC frameworks can be imported."""
        return _APPKIT_AVAILABLE

    def _run_fallback(self) -> None:
        """Block without AppKit.

        The short timeout keeps the interpreter responsive to signals, matching
        the spirit of the AppKit pump timer.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.2)

    def _run_appkit(self) -> None:
        """Run the main run loop with an interpreter-pump timer.

        If any AppKit call fails, fall back to a plain event wait so the
        meeting keeps running without a HUD.
        """
        if _Bridge is None:
            return self._run_fallback()

        try:
            # Allow HudPanel to create its bridge before starting the run loop.
            self._prepare_appkit()

            app = NSApplication.sharedApplication()
            self._app = app

            # No Dock icon, no app switcher entry, no focus stealing.
            app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

            run_loop = NSRunLoop.currentRunLoop()

            pump = _Bridge.alloc().init()
            self._pump_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.2, pump, "tick:", None, True
            )

            while not self._stop_event.is_set():
                try:
                    run_loop.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.distantFuture())
                except KeyboardInterrupt:
                    # Ctrl+C reaches the main thread because the pump timer
                    # returns control to the interpreter.
                    self._stop_event.set()
        except Exception:
            # AppKit failed at runtime.  Do not crash the meeting; degrade to
            # the non-AppKit wait so stop() can still be honored.
            return self._run_fallback()
        finally:
            if self._pump_timer is not None:
                self._pump_timer.invalidate()
                self._pump_timer = None
            self._after_run_loop()
            self._stop_event.set()

    def _prepare_appkit(self) -> None:
        """Hook for subclasses to set up the bridge before the run loop starts."""
        pass

    def _after_run_loop(self) -> None:
        """Hook for subclasses to clean up after the run loop exits."""
        pass


# ---------------------------------------------------------------------------
# Null/fallback implementation
# ---------------------------------------------------------------------------

class NullHud(_BaseHud):
    """Default HUD: no panel, no badges, but it owns the main run loop.

    If AppKit is not available, or if run() is called on a non-main thread,
    it falls back to a plain threading.Event so the app still functions.
    """

    def turn_started(self) -> None:
        """No-op in the null implementation."""
        pass

    def turn_finished(self, ok: bool) -> None:
        """No-op in the null implementation."""
        # 'ok' is part of the protocol but not used until the Outcome badge
        # slice.  The parameter is intentionally unused in this slice.
        pass


# ---------------------------------------------------------------------------
# Real AppKit panel
# ---------------------------------------------------------------------------

class HudPanel(_BaseHud):
    """A small non-activating floating panel that shows the Thinking badge.

    The panel is created on the main thread and all badge updates are
    marshalled to the main thread internally, so callers never see AppKit.
    """

    _PANEL_WIDTH = 120
    _PANEL_HEIGHT = 36
    _PANEL_PADDING = 12

    def __init__(self, monitor_index: int = 1) -> None:
        super().__init__()
        self._monitor_index = monitor_index
        self._monitor, self._main_height = self._load_geometry()
        self._panel: Any | None = None
        self._label: Any | None = None
        self._bridge: Any | None = None

        if self._appkit_available() and threading.current_thread() is threading.main_thread():
            try:
                self._bridge = _Bridge.alloc().initWithHud_(self)
            except Exception:
                # Bridge creation failed; turn_* methods will become no-ops.
                self._bridge = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the run loop to exit and hide the panel."""
        super().stop()
        if self._bridge is not None:
            try:
                self._bridge.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "hide:", None, False
                )
            except Exception:
                pass

    def turn_started(self) -> None:
        """Show the Thinking badge on the main thread."""
        self._call_on_main("showThinking_", "showThinking:")

    def turn_finished(self, ok: bool) -> None:
        """Hide the badge on the main thread (single-turn slice).

        'ok' is not used in this slice; the Outcome badge slice will use it.
        """
        _ = ok
        self._call_on_main("hide_", "hide:")

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _prepare_appkit(self) -> None:
        """Create the bridge before the run loop starts, if it wasn't already."""
        if self._bridge is not None:
            return
        try:
            self._bridge = _Bridge.alloc().initWithHud_(self)
        except Exception:
            self._bridge = None

    def _after_run_loop(self) -> None:
        """Hide the panel when the run loop exits."""
        self._hide()

    def _call_on_main(self, py_method: str, objc_selector: str) -> None:
        """Call a bridge method, either directly or via the main run loop."""
        if not self._appkit_available() or self._bridge is None:
            return
        if threading.current_thread() is threading.main_thread():
            try:
                getattr(self._bridge, py_method)(None)
            except Exception:
                pass
        else:
            try:
                self._bridge.performSelectorOnMainThread_withObject_waitUntilDone_(
                    objc_selector, None, False
                )
            except Exception:
                pass

    def _load_geometry(self) -> tuple[Mapping[str, int] | None, int]:
        """Return the selected mss monitor and the main display height.

        The main display height is needed to convert mss top-left Quartz
        coordinates into AppKit bottom-left coordinates.
        """
        if not _MSS_AVAILABLE or mss is None:
            return None, 0

        try:
            with mss.MSS() as sct:
                all_monitors = sct.monitors
                if not all_monitors:
                    return None, 0

                if 0 <= self._monitor_index < len(all_monitors):
                    selected = all_monitors[self._monitor_index]
                else:
                    selected = all_monitors[1] if len(all_monitors) > 1 else all_monitors[0]

                physical = all_monitors[1:]
                if physical:
                    main = next(
                        (m for m in physical if m.get("left") == 0 and m.get("top") == 0),
                        physical[0],
                    )
                    main_height = main.get("height", selected.get("height", 0))
                else:
                    main_height = selected.get("height", 0)

                return selected, main_height
        except Exception:
            return None, 0

    def _content_rect(self) -> Any:
        """Return the panel frame for the bottom-right of the chosen screen."""
        monitor = self._monitor
        if monitor is None:
            x = float(self._PANEL_PADDING)
            y = float(self._PANEL_PADDING)
        else:
            left = float(monitor.get("left", 0))
            top = float(monitor.get("top", 0))
            width = float(monitor.get("width", 0))
            height = float(monitor.get("height", 0))

            x = left + width - self._PANEL_WIDTH - self._PANEL_PADDING
            # Convert mss top-left Quartz coordinates to AppKit bottom-left.
            y = -(top + height) + self._main_height + self._PANEL_PADDING

        return NSMakeRect(
            float(x),
            float(y),
            float(self._PANEL_WIDTH),
            float(self._PANEL_HEIGHT),
        )

    def _ensure_panel(self) -> None:
        """Create the panel and its badge label on the main thread."""
        if self._panel is not None:
            return

        rect = self._content_rect()
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSHUDWindowMask | NSNonactivatingPanelMask,
            NSBackingStoreBuffered,
            False,
        )

        # Float above the menu bar and full-screen content.
        panel.setLevel_(NSStatusWindowLevel)

        # Follow the user across Spaces and into full-screen apps.
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorCanJoinAllApplications
        )

        # Do not let the panel appear in screenshots taken by this process.
        panel.setSharingType_(NSWindowSharingNone)

        # Pass clicks through to whatever is underneath.
        panel.setIgnoresMouseEvents_(True)

        # A floating, non-activating panel that does not hide on deactivate.
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setHidesOnDeactivate_(False)

        label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(0.0, 0.0, float(self._PANEL_WIDTH), float(self._PANEL_HEIGHT))
        )
        label.setStringValue_("Thinking")
        label.setBezeled_(False)
        label.setDrawsBackground_(True)
        label.setBackgroundColor_(NSColor.blackColor())
        label.setTextColor_(NSColor.whiteColor())
        label.setFont_(NSFont.boldSystemFontOfSize_(14.0))
        label.setAlignment_(NSCenterTextAlignment)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setBordered_(False)

        panel.setContentView_(label)
        self._panel = panel
        self._label = label

    def _show_thinking(self) -> None:
        """Create (if necessary) and show the Thinking badge."""
        try:
            if self._panel is None:
                self._ensure_panel()
            if self._label is not None:
                self._label.setStringValue_("Thinking")
            if self._panel is not None:
                self._panel.orderFront_(None)
        except Exception:
            # A failed panel is not worth crashing the meeting over.
            self._panel = None
            self._label = None

    def _hide(self) -> None:
        """Hide the panel, if one exists."""
        try:
            if self._panel is not None:
                self._panel.orderOut_(None)
        except Exception:
            self._panel = None
            self._label = None
