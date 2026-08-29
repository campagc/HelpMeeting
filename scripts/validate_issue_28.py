#!/usr/bin/env python3
"""Real-Mac validation runner for issue #28.

Automates the checks that do not require a human-in-the-loop, then emits a
Markdown report with the full acceptance checklist ready for the manual
observations to be filled in.

Usage:
    python scripts/validate_issue_28.py
    python scripts/validate_issue_28.py --output validation_report_issue_28.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sounddevice as sd  # type: ignore[import-untyped]

from src.system_playback import default_system_playback_source


_ACCEPTANCE = [
    ("First-use permission behavior requests Screen & System Audio Recording access, exits cleanly, and gives correct relaunch guidance", False),
    ("After relaunch, HelpMeeting starts with MacBook Air Speakers selected and never changes the active output", False),
    ("Playing meeting speech grows the Transcript through ScreenCaptureKit", False),
    ("Volume up, volume down, mute, and Control Center volume remain usable during capture", False),
    ("Captured speech remains detectable at more than one audible speaker volume", False),
    ("An Explain turn consumes the current Delta and Slide and records its result in the archive", False),
    ("A Question turn consumes the current Delta and records both sides of the Turn in the archive", False),
    ("HUD badges and Slide exclusion behave as before", False),
    ("Stopping the Meeting releases native capture and finalizes the archive without hanging", False),
    ("The audio diagnostic reports a non-silent native signal and a non-empty Whisper transcription", False),
    ("Explicitly selecting the BlackHole fallback starts the known path and never occurs automatically", False),
    ("Returning to the default setting selects ScreenCaptureKit again without manual output routing", False),
    ("The final output-device, Transcript, Delta, Turn, archive, diagnostic, and shutdown observations are recorded on this ticket", False),
]


def _run_pytest() -> tuple[bool, str]:
    print("Running automated test suite...", file=sys.stderr)
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    duration = time.monotonic() - started
    success = result.returncode == 0
    return success, f"{duration:.2f}s:\n{result.stdout}\n{result.stderr}"


def _active_output() -> str:
    try:
        return str(sd.query_devices(kind="output")["name"])
    except Exception as exc:
        return f"unavailable ({exc})"


def _selected_source() -> str:
    try:
        return default_system_playback_source().description
    except Exception as exc:
        return f"could not resolve source ({exc})"


def _report(test_ok: bool, test_details: str, source: str, output: str) -> str:
    status = "PASS" if test_ok else "FAIL"
    lines = [
        f"# Issue #28 Real-Mac Validation Report",
        "",
        f"_Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Automated pre-flight",
        "",
        f"- Full automated test suite: **{status}** ({test_details.strip()})",
        f"- Selected audio source: `{source}`",
        f"- Active output device: `{output}`",
        "",
        "## Manual acceptance checklist",
        "",
        "Replace the `[ ]` with `[x]` and add notes as each step is completed.",
        "",
    ]
    for text, _ in _ACCEPTANCE:
        lines.append(f"- [ ] {text}")
        lines.append("  - _notes:_ ")
    lines.append("")
    lines.append("## Remaining observations to record")
    lines.append("")
    lines.append("- Final output-device state:")
    lines.append("- Final Transcript/Delta state:")
    lines.append("- Explain and Question Turn archive entries:")
    lines.append("- Archive/diagnostic files:")
    lines.append("- Shutdown behavior:")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Issue #28 real-Mac validation runner")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="File to write the Markdown report to (default: print to stdout)",
    )
    args = parser.parse_args(argv)

    test_ok, test_details = _run_pytest()
    source = _selected_source()
    output = _active_output()
    report = _report(test_ok, test_details, source, output)

    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0 if test_ok else 1


if __name__ == "__main__":
    sys.exit(main())
