#!/usr/bin/env python3
"""Phase-2 manual round-trip verification (issue #5).

Sends a sample slide image + transcript delta through the Assistant and
prints the model's explanation.  Requires a real GEMINI_API_KEY in .env.

Usage:
    python scripts/verify_roundtrip.py
"""
from pathlib import Path
import sys

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from src import config
from src.assistant import Assistant
from src.transcript import Transcript


SAMPLE_SLIDE = Path("tests/fixtures/sample_slide.png")
SAMPLE_DELTA = (
    "So what we're seeing here is that the gradient descent converges "
    "in about 50 epochs when we use the Adam optimizer with a learning "
    "rate of 3e-4. The key insight is that batch normalization before "
    "the activation function gives us a 12% improvement on the validation set."
)


def main():
    print("=" * 60)
    print("Phase-2 Round-Trip Verification (issue #5)")
    print("=" * 60)

    # Load config (reads .env for GEMINI_API_KEY)
    cfg = config.load()
    print(f"\nModel: {cfg.gemini_model_name}")
    print(f"API key: {cfg.api_key[:8]}...{cfg.api_key[-4:]}")

    # Create real Gemini client
    client = genai.Client(api_key=cfg.api_key)

    # Create transcript and assistant
    transcript = Transcript()
    transcript.append(SAMPLE_DELTA)

    assistant = Assistant(
        client=client,
        model=cfg.gemini_model_name,
        transcript=transcript,
    )

    # Load sample slide
    image_bytes = SAMPLE_SLIDE.read_bytes()
    print(f"Sample slide: {SAMPLE_SLIDE} ({len(image_bytes)} bytes)")

    # Take delta and send
    delta = transcript.take_delta()
    print(f"Transcript delta: {delta[:80]}...")
    print("\nSending explain_slide request...")
    print("-" * 60)

    response = assistant.explain_slide(image_bytes=image_bytes, delta=delta)

    print(response)
    print("-" * 60)

    # Check for error markers
    if response.startswith("["):
        print("\n*** FAILED: Got an error response. See above. ***")
        return 1

    print("\n*** SUCCESS: Got a sensible explanation from the model. ***")
    print(f"Response length: {len(response)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
