#!/usr/bin/env python3
"""Gemini TTS voice A/B harness — pick the kids-storyteller narrator before bulk synthesis.

Synthesizes a SHORT excerpt of one Marathi + one Hindi story with several Gemini voices,
using a sweet/animated children's-storyteller style prompt, so you can listen and lock ONE
voice for the full training dataset.

Setup (once):
  /Users/roshnisundrani/TTS/TTS_finetuning/venv/bin/pip install google-genai soundfile

Run (from repo root):
  venv/bin/python scripts/dataprep_gemini/tts_ab.py
  # or pick voices / model / your own sample files:
  venv/bin/python scripts/dataprep_gemini/tts_ab.py --voices Leda,Sulafat,Puck \
      --model gemini-2.5-pro-preview-tts \
      --marathi-file story_texts/kawda_test.txt --hindi-file story_texts/hindi_test_Story.txt

Outputs -> gemini_ab/<voice>_<lang>.wav  (24 kHz mono). Listen with: afplay gemini_ab/Leda_marathi.wav
The Gemini API key is read from .env (a line whose key contains "gemini"); never printed.
NOTE: TTS model IDs are preview and change — if a model 404s, check the current id at
https://ai.google.dev/gemini-api/docs/speech-generation and pass it via --model.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Kid-friendly candidates (descriptor): Leda=Youthful, Sulafat=Warm, Puck=Upbeat,
# Aoede=Breezy, Achird=Friendly, Sadachbia=Lively, Laomedeia=Upbeat.
DEFAULT_VOICES = ["Leda", "Sulafat", "Puck"]

# The style directive that turns Gemini into a sweet, animated kids storyteller.
# Keep this CONSISTENT for the whole training dataset so Parler learns one stable style.
STYLE = (
    "You are a warm, cheerful children's storyteller narrating a story for kids aged 2 to 12. "
    "Speak in a sweet, animated, expressive voice with a lively but clear pace, gentle warmth, "
    "and playful energy. Pronounce every word clearly. Narrate this story:"
)

GEMINI_SR = 24000  # Gemini TTS returns 24 kHz, 16-bit, mono PCM


def load_gemini_key(env_path: Path) -> str:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if "gemini" in k.lower():
                return v.strip().strip('"').strip("'")
    import os
    return os.environ.get("GEMINI_API_KEY", "")


def first_sentences(text: str, n: int, max_chars: int) -> str:
    """Take the first n sentences (or up to max_chars) — enough to judge the voice, cheap."""
    parts = re.split(r"(?<=[।.!?])\s+|\n+", text.strip())
    parts = [p.strip() for p in parts if p.strip() and "�" not in p]  # drop broken (�) lines
    out = " ".join(parts[:n])
    return out[:max_chars]


def write_wav(path: Path, pcm: bytes, rate: int = GEMINI_SR) -> float:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return len(pcm) / 2 / rate  # seconds


def synth(client, model: str, voice: str, prompt: str):
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    return resp.candidates[0].content.parts[0].inline_data.data  # PCM bytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES), help="comma-separated voice names")
    ap.add_argument("--model", default="gemini-2.5-pro-preview-tts", help="Gemini TTS model id (Pro by default)")
    ap.add_argument("--marathi-file", type=Path, default=ROOT / "story_texts" / "kawda_test.txt")
    ap.add_argument("--hindi-file", type=Path, default=ROOT / "story_texts" / "hindi_test_Story.txt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "gemini_ab")
    ap.add_argument("--sentences", type=int, default=5, help="how many opening sentences to test")
    ap.add_argument("--max-chars", type=int, default=700, help="hard cap on excerpt length")
    ap.add_argument("--style", default=STYLE, help="style directive prepended to the text")
    ap.add_argument("--style-file", type=Path, default=None, help="read the style directive from a file (overrides --style)")
    ap.add_argument("--env", type=Path, default=ROOT / ".env")
    args = ap.parse_args()
    if args.style_file and args.style_file.is_file():
        args.style = args.style_file.read_text(encoding="utf-8").strip()

    try:
        from google import genai  # noqa: F401
    except ImportError:
        sys.exit("ERROR: install the SDK first:\n  venv/bin/pip install google-genai soundfile")

    key = load_gemini_key(args.env)
    if not key:
        sys.exit(f"ERROR: no Gemini key found in {args.env} (need a line like 'gemini-key=...')")

    from google import genai
    client = genai.Client(api_key=key)

    samples = []
    for lang, f in (("marathi", args.marathi_file), ("hindi", args.hindi_file)):
        if not f.is_file():
            print(f"WARN: {f} missing — skipping {lang}")
            continue
        excerpt = first_sentences(f.read_text(encoding="utf-8"), args.sentences, args.max_chars)
        if excerpt:
            samples.append((lang, excerpt))
            print(f"[{lang}] excerpt ({len(excerpt)} chars): {excerpt[:70]}...")

    if not samples:
        sys.exit("ERROR: no usable sample text found.")

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nModel: {args.model} | voices: {voices} | out: {args.out_dir}\n")

    results = []
    for voice in voices:
        for lang, excerpt in samples:
            prompt = f"{args.style}\n\n{excerpt}"
            out = args.out_dir / f"{voice}_{lang}.wav"
            try:
                pcm = synth(client, args.model, voice, prompt)
                dur = write_wav(out, pcm)
                print(f"  OK  {voice:<10} {lang:<8} -> {out.name}  ({dur:.1f}s)")
                results.append((voice, lang, f"{dur:.1f}s"))
            except Exception as exc:
                print(f"  ERR {voice:<10} {lang:<8} -> {exc}")
                results.append((voice, lang, f"FAIL: {str(exc)[:80]}"))
            time.sleep(1.0)  # be gentle on rate limits

    print("\n=== A/B done ===")
    for v, l, s in results:
        print(f"  {v:<10} {l:<8} {s}")
    print(f"\nListen:  afplay {args.out_dir}/Leda_marathi.wav   (or: open {args.out_dir})")
    print("Pick the best voice, then we lock it for the full 3h+3h synthesis.")


if __name__ == "__main__":
    main()
