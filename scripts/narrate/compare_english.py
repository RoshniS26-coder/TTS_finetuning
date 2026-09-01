#!/usr/bin/env python3
"""Compare English TTS: base Parler-TTS vs the fine-tuned Betacraft model.

The Betacraft model (roshni-sorigin/mar-hin-betacraft-tts) was fine-tuned
ONLY on Marathi/Hindi (see CLAUDE.md v7 SYNTHETIC DATASET) — English is
expected to sound broken/off-voice. This script exists to LISTEN and confirm
that, not because English is a supported target.

Usage:
  venv/bin/python scripts/narrate/compare_english.py \
      --text "Once upon a time, in a small village, there lived a clever fox."
Outputs:
  reports/english_compare/base_parler.wav
  reports/english_compare/betacraft_finetuned.wav
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("compare_english")

BASE_MODEL = "parler-tts/parler-tts-mini-v1"
BETACRAFT_MODEL = "roshni-sorigin/mar-hin-betacraft-tts"

BASE_DESCRIPTION = (
    "A female speaker with a clear, warm, animated storytelling voice "
    "delivers her speech at a moderate pace. Very clear audio with no "
    "background noise."
)
# Same caption template the betacraft model was trained on (Sunita=Marathi
# speaker slot) — reused here on English text just to probe behavior.
BETACRAFT_DESCRIPTION = (
    "Sunita narrates a children's story in an expressive, warm, animated "
    "storytelling tone with emotional variation, at a moderate pace. "
    "Very clear audio with no background noise."
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synthesize(model_id: str, description: str, text: str, device: str, out_path: Path, seed: int = 42) -> None:
    log.info("Loading %s on %s ...", model_id, device)
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()
    prompt_tok = AutoTokenizer.from_pretrained(model_id)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    desc_ids = desc_tok(description, return_tensors="pt").input_ids.to(device)
    prompt_ids = prompt_tok(text, return_tensors="pt").input_ids.to(device)

    set_seed(seed)
    log.info("Generating: %s", text[:80])
    with torch.no_grad():
        audio = model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids)

    wav = audio.cpu().numpy().squeeze()
    sr = model.config.sampling_rate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wav, sr, format="WAV")
    log.info("Wrote %s (%.1fs)", out_path, len(wav) / sr)

    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda:0":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Once upon a time, in a small village, there lived a clever fox.")
    parser.add_argument("--out-dir", default="reports/english_compare")
    args = parser.parse_args()

    device = _pick_device()
    out_dir = Path(args.out_dir)

    synthesize(BASE_MODEL, BASE_DESCRIPTION, args.text, device, out_dir / "base_parler.wav")
    synthesize(BETACRAFT_MODEL, BETACRAFT_DESCRIPTION, args.text, device, out_dir / "betacraft_finetuned.wav")

    log.info("Done. Compare:\n  %s\n  %s", out_dir / "base_parler.wav", out_dir / "betacraft_finetuned.wav")


if __name__ == "__main__":
    main()
