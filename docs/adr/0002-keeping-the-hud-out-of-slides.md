# Keeping the HUD out of slides

The HUD is a real window, so the window server composites it into any screen grab — and
because the outcome badge appears whenever the assistant happens to finish, it can be
on screen at the exact moment the attendee triggers the next slide. Simply ordering the
grab before the badge is not enough; the risk is the *previous* turn's badge.

We set `NSWindowSharingNone` on the panel. A spike confirmed this excludes the panel
from an `mss` grab, so the Quartz `CGWindowListCreateImage` fallback is not required for
v1. The fallback is kept documented in case future macOS versions change the behavior.

## Spike result

A throwaway script (deleted after the run) created a red `NSPanel` with the chosen
`sharingType`, placed it on the main display, and captured the display with `mss`:

- `NSWindowSharingReadWrite` (2): the captured PNG contained ~160,000 red pixels.
- `NSWindowSharingNone` (0): the captured PNG contained 0 red pixels.

Result determined: 2026-08-23. The panel remained visually on screen in both cases; the
flag only affected whether `mss` included it in the grab.

## Considered options

- **Hide, grab, restore.** Rejected: the window server composites asynchronously, so
  "hidden" is not instantly true. Making it reliable requires a sleep, which is a race
  wearing a fix's clothing — and it flickers.
- **Park the HUD on another display.** Rejected: only works on multi-monitor setups, and
  single-monitor is the common case.
- **Quartz `CGWindowListCreateImage` with `kCGWindowListOptionOnScreenBelowWindow`.**
  Documented fallback, not taken: the spike showed `NSWindowSharingNone` is sufficient
  for `mss` in the same process.

## Consequences

- `NSWindowSharingNone` does exclude the HUD from `mss` grabs taken by the same process.
  The HUD can stay on screen during a grab, live on the captured display, and persist
  for the whole duration of a turn.
- `mss` remains the screen grabber for v1; the documented Quartz fallback is available
  if `NSWindowSharingNone` stops working in a future macOS release.
- Excluding the HUD by construction means its visibility no longer constrains anything
  else.
