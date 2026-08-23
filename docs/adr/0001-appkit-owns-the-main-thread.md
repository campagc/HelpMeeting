# AppKit owns the main thread; the question loop moves to a background thread

The HUD is an `NSPanel`, and AppKit refuses to run its event loop anywhere but the main
thread — which the terminal question loop currently occupies, blocked in `input()`. We
gave the main thread to `NSApplication` and moved `run_input_loop` onto a daemon thread,
rather than isolating the HUD in a helper subprocess, because a second process would need
its own IPC protocol, lifecycle supervision, and kill path for what is ultimately a window
showing two badges.

## Consequences

- `NSApp.run()` is a C call that never returns to Python, so a `SIGINT` handler queued by
  Ctrl+C would never execute. A repeating ~0.2s `NSTimer` exists solely to hand control
  back to the interpreter so the handler can fire. It looks like dead code and is not.
- The input thread exiting (EOF, `Ctrl+D`) no longer ends the process on its own. It must
  set the shutdown event and stop the run loop explicitly, or the app hangs.
- The process sets `NSApplicationActivationPolicyAccessory`: no Dock icon, no focus steal.
- If AppKit or the panel fails to initialise, the app must degrade to a no-op HUD. Losing
  badges is acceptable; losing the meeting is not.
