#!/usr/bin/env python3
"""Audio capture diagnostic.

Confirms that meeting audio actually reaches the capture device (BlackHole)
and that faster-whisper can transcribe it.  Run this while audio is playing
through your Mac's output (routed into BlackHole via a Multi-Output Device).

Usage:
    python scripts/verify_audio.py            # capture 8s from BlackHole
    python scripts/verify_audio.py 5          # capture 5s
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd

from src.main import resolve_audio_device

_SAMPLE_RATE = 16_000


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    print("Available audio devices:")
    print(sd.query_devices())
    print()

    try:
        device = resolve_audio_device(sd.query_devices())
    except RuntimeError as exc:
        print(f"*** {exc}")
        return 1

    name = sd.query_devices(device)["name"]
    print(f"Capturing {seconds}s from device {device}: {name}")
    print("--> PLAY SOME AUDIO NOW (a video, music, the meeting) <--")

    recording = sd.rec(
        int(seconds * _SAMPLE_RATE),
        samplerate=_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    audio = recording[:, 0]

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"\nCaptured {len(audio)} samples | RMS={rms:.6f} | peak={peak:.6f}")

    if rms < 1e-4:
        print(
            "\n*** SILENCE: no audio reached the capture device.\n"
            "    Set your Mac's sound OUTPUT to a Multi-Output Device that\n"
            "    includes 'BlackHole 2ch' (System Settings -> Sound -> Output),\n"
            "    then re-run this while audio is playing."
        )
        return 1

    from src.config import WHISPER_MODEL_SIZE

    print(f"\nSignal detected. Transcribing with faster-whisper ({WHISPER_MODEL_SIZE})…")
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio, language="en", beam_size=5, vad_filter=False)
    text = " ".join(seg.text.strip() for seg in segments).strip()

    print("-" * 60)
    print(text if text else "(no speech recognized — was it speech, or just music/noise?)")
    print("-" * 60)
    print("\n*** SUCCESS: audio is reaching the app and can be transcribed. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
