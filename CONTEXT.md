# HelpMeeting

A macOS assistant for research meetings you can't follow in real time: it listens to system
audio, transcribes locally, and on demand asks an AI to explain the current slide in the
light of what has just been said.

## Language

### The meeting

**Meeting**:
One recorded sitting, from launch to shutdown, identified by a label the attendee gives at
startup. Owns exactly one archive folder.
_Avoid_: Session, call, talk

**Transcript**:
The cumulative text of everything spoken in the meeting so far.

**Transcript intake**:
The module that turns system playback into transcript. It listens to one playback
source, transcribes in fixed-length chunks, appends the text to the transcript and the
archive, and restarts itself if the source fails. Its only seam is the playback source.
_Avoid_: Audio thread, recorder, listener

**Delta**:
The stretch of transcript that has accumulated since the previous turn. Reading a delta
consumes it — the next delta starts where this one ended.
_Avoid_: New speech, increment, diff

**Slide**:
A screenshot of the chosen display at the moment the attendee asked for help. Named for
intent, not content — it is a slide even if the screen shows a whiteboard or a demo.
_Avoid_: Screenshot, capture, frame

**Display**:
One physical screen, numbered from 1 in the order the system reports them. The attendee
chooses one at startup; the slide is grabbed from it and the HUD is anchored to it. The
Display module alone knows screen geometry and coordinate systems.
_Avoid_: Monitor, screen (as a noun), mss index

### Asking for help

**Turn**:
One complete request-and-response exchange with the assistant. Every turn consumes the
current delta and is recorded in the archive. Comes in exactly two kinds.

**Explain turn**:
A turn triggered by a hotkey: slide plus delta, asking what is new on screen.
_Avoid_: Hotkey turn, snapshot turn

**Question turn**:
A turn triggered by the attendee typing: free-form text plus delta, no new slide.

**Trigger**:
A hotkey press that asks for an explain turn. Multiple key combinations are triggers for
the same thing; a trigger too soon after the previous one is dropped rather than queued.

### Feedback

**HUD**:
A small always-on-top window that floats above every application, including full-screen
ones, and never takes focus. It carries no content — only feedback about what the app is
doing. Shows at most one badge at a time.
_Avoid_: Overlay, notification, toast, popup

**Badge**:
One state shown in the HUD. A new badge replaces the current one; badges never stack or
queue. There are exactly two.
_Avoid_: Alert, message, indicator

**Thinking badge**:
Shown while any turn is outstanding — from the moment the attendee asks, not from the
moment the assistant is actually called. Persistent, because the wait is long enough that
silence would read as the app having died. It doubles as confirmation that the slide was
taken.

**Outcome badge**:
Shown briefly when a turn resolves — one badge with two readings, done and failed. It
reports whether the turn produced a usable answer.
_Avoid_: Ready badge, done badge, error badge

## Flagged ambiguities

**"Overlay"** — the PRD uses it for a deferred Flask page that would *render explanations*
(a reading surface). It must not be used for the HUD, which renders no explanations at all.
Prefer "reading surface" for the former and "HUD" for the latter.

**"Hotkey"** — ambiguous now that more than one key combination exists. A **hotkey** is a
specific key combination; a **trigger** is the request it produces. Debouncing applies to
triggers, not to hotkeys, so two different hotkeys share one debounce window.

## Example dialogue

> **Dev**: When they hit the hotkey, does the HUD show before or after we grab the slide?
>
> **Attendee**: After. If the HUD is on screen when you grab, it lands in the slide and
> the AI sees it.
>
> **Dev**: So the badge confirming the trigger comes after the slide exists.
>
> **Attendee**: Right. And then a second badge when the turn finishes and the explanation
> is in the archive.
>
> **Dev**: Two badges per explain turn. What about a question turn?
>
> **Attendee**: No slide, but still the same first badge when the turn starts — so I know
> the request registered — and the same finished badge when the answer lands.
