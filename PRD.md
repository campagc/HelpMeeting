# PRD — HelpMeeting (Semi-Live Meeting Assistant)

## Problem Statement

I regularly attend highly technical research meetings (over Zoom, Google Meet, and similar) where the subject matter is well above my background. In real time I lose the thread: I can't tell what a slide is actually about, why it matters, or how it connects to what was said a minute ago. By the time I've parsed one slide, the speaker has moved on. I need on-demand help understanding what is being discussed *as it happens*, and I want a permanent, browsable record of each talk so I can revisit it later.

## Solution

A macOS background app that continuously listens to whatever audio is playing through my Mac (the meeting), transcribes it locally, and \u2014 whenever I press a global hotkey \u2014 takes a screenshot of the current slide, bundles it with the new speech since my last request, and asks Gemini to explain what's happening at the level of "a smart person from an adjacent field." The explanation appears in a live-updating Markdown view. I can also type free-form questions at any time, and the assistant answers using the full accumulated context of the meeting (every slide and the whole transcript), so it builds on what it already told me instead of repeating itself. Every meeting is saved permanently as a folder (transcript, slide images, machine-readable history, and a human-readable log).

## User Stories

1. As a confused attendee, I want the app to capture my Mac's system audio without me configuring the meeting app, so that it works identically for Zoom, Google Meet, Teams, or any source.
2. As an attendee, I want the meeting transcribed continuously and locally, so that there is no per-minute cost and no transcript data leaves my machine until I explicitly ask for help.
3. As an attendee, I want to start a meeting by running a single command, so that setup friction is minimal.
4. As an attendee, I want to give each meeting a short label at startup, so that I can find a specific talk in my archive months later.
5. As an attendee, I want to pick the spoken language at startup (defaulting to English), so that transcription is accurate and doesn't flip languages mid-sentence.
6. As an attendee, I want to pick the explanation language at startup (defaulting to the spoken language), so that I read explanations in whatever is easiest for me to absorb.
7. As an attendee on a multi-monitor setup, I want to choose which display holds the meeting at startup, so that screenshots capture the right screen instead of a blank desktop.
8. As an attendee, I want to press a global hotkey while the meeting app is focused, so that I can request help without alt-tabbing away from the talk.
9. As an attendee, I want each hotkey press to send the current slide image plus only the new speech since my last request, so that the assistant explains what's new and doesn't repeat earlier explanations.
10. As an attendee, I want the assistant to retain every prior slide and explanation in one ongoing conversation, so that it understands the trajectory of the talk and connects the current slide to what came before.
11. As an attendee, I want to type free-form questions at any time, so that I can ask about anything in the meeting using the full accumulated context.
12. As an attendee, I want my typed questions to also carry the latest speech (but not a redundant new screenshot), so that the assistant stays current while keeping question turns cheap.
13. As an attendee, I want explanations targeted at an adjacent-field expert by default (define jargon, explain significance and connections), so that the help is deep enough to actually follow a research talk.
14. As an attendee, I want to edit the assistant's persona/instructions in a simple text file, so that I can dial the explanation level up or down without changing code.
15. As an attendee, I want to read explanations in a live-updating Markdown view that renders slide images inline, so that async explanations don't clobber my typed input in the terminal.
16. As an attendee, I want a clear inline message when the AI is rate-limited or unreachable instead of a crash, so that one flaky moment doesn't end my session.
17. As an attendee, I want accidental double-presses of the hotkey ignored, so that I don't burn through my free-tier request budget.
18. As an attendee, I want the app to keep running even if transcription hits an error, so that I never silently lose the live help.
19. As an attendee, I want every meeting saved as its own folder containing the transcript, slide images, a machine-readable history, and a human-readable log, so that I have a permanent, browsable archive.
20. As an attendee, I want all artifacts flushed cleanly when I stop the app with Ctrl+C, so that nothing is left half-written.
21. As a budget-conscious user, I want the whole thing to fit within the Gemini free tier, so that I incur no cost.
22. As a privacy-conscious user, I want my API key kept out of source control, so that I don't leak credentials.
23. As a future-me, I want the saved history format to be rich enough to resume a past meeting later, so that I retain the option even though resume isn't built yet.
24. As a developer vibe-coding this later, I want the code split into small single-responsibility modules, so that I can hand one file at a time to an AI assistant without wading through a monolith.

## Implementation Decisions

**Platform & runtime**
- macOS on Apple Silicon. All work happens in an isolated Python 3.12 virtualenv; the system Python (3.14) is left untouched to avoid missing-wheel/compile failures.
- Single process with multiple threads (transcription thread, global-hotkey listener, terminal question-input loop) sharing in-memory state. The transcription thread is wrapped in a restart-on-exception loop so a transcription failure cannot take the app down.

**Audio capture & transcription**
- System audio is captured via BlackHole 2ch routed through a macOS Multi-Output Device (validated in Phase 0). Because we capture system audio, no meeting-app integration is ever required \u2014 Zoom, Meet, Teams, etc. all work identically.
- Audio is read with `sounddevice` (requires `portaudio`), delivered as in-memory numpy arrays. An ffmpeg-subprocess pipe is the documented known-good fallback.
- Transcription uses `faster-whisper` with the `small` model on CPU, treated as a tunable knob (drop to `base` if the Mac struggles, raise to `medium` if jargon is garbled).
- The audio stream is chunked into fixed ~10-second windows; each window is transcribed and appended to the transcript. Sophisticated VAD/overlap chunking is intentionally avoided because the transcript consumer is an LLM that tolerates minor boundary noise.

**Trigger, screenshot & hotkey**
- A global hotkey (`Control+Option+Space` by default, defined as a single configurable constant) triggers a slide snapshot; it works while the meeting app is focused via `pynput` (requires macOS Accessibility permission).
- Screenshots use `mss`, capturing the full screen of a display chosen at startup when more than one monitor is present (default: main display; requires macOS Screen Recording permission).
- Hotkey presses are debounced (~2s) to protect the free-tier request budget.

**The AI conversation**
- Uses the `google-genai` SDK with a single long-lived chat session per meeting. The model name is a configurable constant defaulting to the current free-tier Flash model (exact name confirmed at first key test; Gemini 1.5 is deprecated).
- Two turn types share the same chat:
  - Hotkey turn = current slide screenshot + transcript delta + a fixed instruction to explain only what is new.
  - Question turn = the typed question + transcript delta, with no new screenshot (the latest slide already lives in the chat history).
  - Both turn types advance the transcript delta pointer.
- The system prompt lives in an editable `system_prompt.md` file loaded at startup. Default persona: explain to a smart person from an adjacent field \u2014 assume general technical literacy, define domain jargon, and explain why the slide matters and how it connects to prior slides. ELI5 is available on demand per question.
- Gemini call failures are handled in three layers: retry with backoff (2\u20133 attempts), never crash, and show a clear inline message (e.g. rate-limited \u2192 wait and retry). The transcription thread keeps running regardless.

**Languages**
- Spoken language and explanation language are chosen at startup. Spoken defaults to English; explanation defaults to the spoken-language setting. Technical terms remain in English regardless of explanation language.

**Persistence**
- One folder per meeting under `meetings/<date_time_label>/` containing: `transcript.txt`, a `slides/` directory of slide PNGs, `history.json` (the chat as data, with images stored as file *paths* rather than embedded base64), and `session.md` (a human-readable log with slides rendered inline).
- Artifacts are written after every turn. Ctrl+C is trapped to flush the last transcript chunk and finalize `history.json`/`session.md`.
- The format is deliberately resume-capable, but reloading a past meeting into a live chat is out of scope for v1.

**Output surface (v1)**
- Explanations print to the terminal and are written to `session.md`. The intended reading surface is a live-refreshing Markdown preview of `session.md` (e.g. the editor's built-in preview), which renders slides inline and avoids the terminal input/output collision. A Flask overlay page is optional and deferred.

**Module breakdown**
- `transcript` (deep) \u2014 holds the cumulative transcript and read pointer. Interface: `append(text)`, `take_delta()` returns new text since the last call and advances the pointer. Encapsulates the "stay current / don't repeat" logic.
- `storage` (deep) \u2014 owns the meeting folder and all artifacts. Interface conceptually: `start_meeting(label, langs)`, `save_slide(image) -> path`, `append_transcript(text)`, `record_turn(role, content, explanation)` (writes `history.json` + `session.md`), `finalize()`.
- `assistant` (deep) \u2014 wraps the Gemini chat session. Interface conceptually: `explain_slide(image, delta) -> str`, `ask_question(text, delta) -> str`, with retry/backoff internal. Constructed with a Gemini client so it can be tested against a fake.
- `audio` (shallow/hardware) \u2014 sounddevice capture + faster-whisper transcription thread; pushes text into `transcript` and `storage`.
- `capture` (shallow/hardware) \u2014 hotkey listener + screenshot.
- `config` (shallow) \u2014 constants (hotkey, model name, chunk seconds, paths) and loaders for `.env` and `system_prompt.md`.
- `main` (shallow glue) \u2014 startup prompts, thread orchestration, question-input loop, signal handling.

**Security**
- The Gemini API key lives in a `.env` file loaded via `python-dotenv`. `.env` is in `.gitignore` from the first commit so the key never enters source control.

## Testing Decisions

Tests should verify external behavior through each module's public interface only \u2014 not internal implementation details \u2014 so the tests stay valid as internals are refactored during vibe-coding.

Modules to be tested:
- **`transcript`** \u2014 pure logic, highest-value tests. Verify: `take_delta()` returns exactly the text appended since the previous call; consecutive `take_delta()` calls with no new text return empty; deltas never overlap or drop text across many append/take cycles.
- **`storage`** \u2014 file-I/O behavior against a temporary directory. Verify: `start_meeting` creates the correctly named folder and skeleton files; `save_slide` writes a PNG and returns a path that `history.json` references (path, not base64); `record_turn` appends well-formed entries to both `history.json` and `session.md`; `finalize` leaves no half-written files.
- **`assistant`** \u2014 tested against a fake/mock Gemini client (no network). Verify: a hotkey turn submits image + delta and a question turn submits text + delta with no image; both advance the transcript pointer exactly once; retry/backoff retries the configured number of times on a transient error and surfaces a graceful result (never raises) on persistent failure.

The hardware-bound modules (`audio`, `capture`) and the `main` orchestration are validated manually by running the app, not unit-tested. There is no prior art in this repo (greenfield project), so tests establish the initial conventions; standard `pytest` with a `tests/` directory and temp-dir fixtures is assumed.

## Out of Scope

- Reloading/resuming a past meeting into a live Gemini chat (format supports it; flow not built in v1).
- The Flask browser overlay (deferred; live Markdown preview is the v1 reading surface).
- Windows or Intel-Mac support (macOS Apple Silicon only).
- Per-meeting-app integration or speaker diarization / who-said-what attribution.
- Explicit Gemini context caching (transcript and images are small enough that resending is cheap; newer models do implicit caching automatically).
- VAD/overlapping-window transcription chunking (added only if fixed windows prove inadequate).
- macOS Keychain credential storage.
- Automated installation of BlackHole / Multi-Output Device setup (manual one-time setup, already completed).

## Further Notes

- Phase 0 (proving BlackHole system-audio capture with ffmpeg) is complete; on this machine BlackHole is audio device index 0.
- Build phases: Phase 1 = venv + portaudio + the transcription thread filling `transcript.txt` live; Phase 2 = `assistant` + `storage` with a manual screenshot+transcript round-trip to Gemini (also confirms the live model name and free-tier limits); Phase 3 = wire the hotkey, question loop, startup prompts, and folders in `main`; Phase 4 (optional) = Flask overlay.
- Remaining build-time unknowns, all deliberately deferred: the exact current free-tier Flash model name and its requests-per-minute/day limits, and the real-world accuracy of the `small` Whisper model on these meetings (tuned by ear in Phase 1).
- No issue tracker / MCP server is configured, so this PRD lives as a local file rather than a tracked issue; it could be published later if a tracker is connected.
