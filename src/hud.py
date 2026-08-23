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
from typing import Any, Protocol

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
        NSMainMenuWindowLevel,
        NSMakeRect,
        NSNonactivatingPanelMask,
        NSPanel,
        NSScreen,
        NSStatusWindowLevel,
        NSTextField,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorStationary,
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
    NSMainMenuWindowLevel = None
    NSMakeRect = None
    NSNonactivatingPanelMask = None
    NSPanel = None
    NSScreen = None
    NSStatusWindowLevel = None
    NSTextField = None
    NSWindowCollectionBehaviorCanJoinAllSpaces = None
    NSWindowCollectionBehaviorFullScreenAuxiliary = None
    NSWindowCollectionBehaviorStationary = None
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
# Null/fallback implementation
# ---------------------------------------------------------------------------

class NullHud:
    """Default HUD: no panel, no badges, but it owns the main run loop.

    If AppKit is not available, or if run() is called on a non-main thread,
    it falls back to a plain threading.Event so the app still functions.
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

    def turn_started(self) -> None:
        """No-op in the null implementation."""
        pass

    def turn_finished(self, ok: bool) -> None:
        """No-op in the null implementation."""
        pass

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
            self._stop_event.set()


# ---------------------------------------------------------------------------
# Real AppKit panel
# ---------------------------------------------------------------------------

class HudPanel:
    """A small non-activating floating panel that shows the Thinking badge.

    The panel is created on the main thread and all badge updates are
    marshalled to the main thread internally, so callers never see AppKit.
    """

    _PANEL_WIDTH = 120
    _PANEL_HEIGHT = 36
    _PANEL_PADDING = 12

    def __init__(self, monitor_index: int = 1) -> None:
        self._stop_event = threading.Event()
        self._monitor_index = monitor_index
        self._app: Any | None = None
        self._pump_timer: Any | None = None
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
        """Signal the run loop to exit and hide the panel."""
        self._stop_event.set()
        if self._bridge is not None:
            try:
                self._bridge.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "hide:", None, False
                )
            except Exception:
                pass

    def turn_started(self) -> None:
        """Show the Thinking badge on the main thread."""
        if not self._appkit_available() or self._bridge is None:
            return
        if threading.current_thread() is threading.main_thread():
            try:
                self._bridge.showThinking_(None)
            except Exception:
                pass
        else:
            try:
                self._bridge.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "showThinking:", None, False
                )
            except Exception:
                pass

    def turn_finished(self, ok: bool) -> None:
        """Hide the badge on the main thread (single-turn slice)."""
        del ok  # not used until the Outcome badge slice
        if not self._appkit_available() or self._bridge is None:
            return
        if threading.current_thread() is threading.main_thread():
            try:
                self._bridge.hide_(None)
            except Exception:
                pass
        else:
            try:
                self._bridge.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "hide:", None, False
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _appkit_available(self) -> bool:
        """Return True when the AppKit/ObjC frameworks can be imported."""
        return _APPKIT_AVAILABLE

    def _run_fallback(self) -> None:
        """Block without AppKit."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.2)

    def _run_appkit(self) -> None:
        """Run the main run loop, creating the panel lazily on demand."""
        if _Bridge is None:
            return self._run_fallback()

        try:
            # Recreate the bridge in case __init__ failed or this is the first
            # call after a fallback return.
            self._bridge = _Bridge.alloc().initWithHud_(self)

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
            # AppKit failed at runtime.  Degrade to the non-AppKit wait.
            return self._run_fallback()
        finally:
            if self._pump_timer is not None:
                self._pump_timer.invalidate()
                self._pump_timer = None
            self._hide()
            self._stop_event.set()

    def _screen(self) -> Any | None:
        """Return the NSScreen the badge should be anchored to."""
        if NSScreen is None:
            return None
        try:
            screens = NSScreen.screens()
            if screens and 0 <= self._monitor_index - 1 < len(screens):
                return screens[self._monitor_index - 1]
            return screens[0] if screens else None
        except Exception:
            return None

    def _content_rect(self) -> Any:
        """Return the panel frame for the bottom-right of the chosen screen."""
        screen = self._screen()
        if screen is None:
            x = float(self._PANEL_PADDING)
            y = float(self._PANEL_PADDING)
        else:
            frame = screen.frame()
            x = frame.origin.x + frame.size.width - self._PANEL_WIDTH - self._PANEL_PADDING
            y = frame.origin.y + self._PANEL_PADDING
        return NSMakeRect(
            max(0.0, float(x)),
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
            | NSWindowCollectionBehaviorStationary
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
