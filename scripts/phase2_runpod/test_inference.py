#!/usr/bin/env python3
"""Post-training inference smoke test for fine-tuned Marathi Parler-TTS (RunPod)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

DEFAULT_MODEL_DIR = Path("/workspace/marathi_tts_output")
DEFAULT_OUTPUT = Path("/workspace/test.wav")
DEFAULT_PROMPT = "एक आटपाट नगर होतं. तिथे एक राजा राहत होता."
DESCRIPTION = (
    # MUST match the v5 training caption byte-for-byte (build_multispeaker_dataset.py
    # CAPTION with name=Sunita, pace=moderate) — Parler voice/style fidelity degrades
    # if the inference description diverges from what was seen in training.
    "Sunita narrates a children's story in an expressive, warm, animated "
    "storytelling tone with emotional variation, at a moderate pace. "
    "Very clear audio with no background noise."
)
# Backstop only (~30s of DAC tokens, >20s training max): caps a runaway noisy
# tail; EOS still does the real stopping. Not meant to truncate real speech.
MAX_NEW_TOKENS = 2580

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--description", default=DESCRIPTION)
    parser.add_argument("--device", default=None, help="cuda:0 or cpu (default: cuda if available)")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def clean_edges(wav, sr, fade_ms=15, pad_ms=30, silence_thresh=1e-3):
    """Trim leading/trailing near-silence and fade the edges.

    Parler-TTS emits a faint onset/tail artefact and rarely ends at zero
    amplitude, which shows up as a click/static at the clip edges.
    """
    if wav.ndim != 1 or wav.size == 0:
        return wav
    above = np.abs(wav) > silence_thresh
    if not above.any():
        return wav
    pad = int(sr * pad_ms / 1000.0)
    first = max(0, int(np.argmax(above)) - pad)
    last = min(len(wav), len(wav) - int(np.argmax(above[::-1])) + pad)
    wav = wav[first:last].copy()
    n = int(sr * fade_ms / 1000.0)
    if n > 0 and wav.size > 2 * n:
        ramp = (0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))).astype(np.float32)
        wav[:n] *= ramp
        wav[-n:] *= ramp[::-1]
    return wav.astype(np.float32)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.model_dir.is_dir():
        log.error("Model directory not found: %s", args.model_dir)
        sys.exit(1)

    log.info("Loading model from %s on %s", args.model_dir, device)
    model = ParlerTTSForConditionalGeneration.from_pretrained(str(args.model_dir)).to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))

    desc_tokenizer_path = model.config.text_encoder._name_or_path
    log.info("Loading description tokenizer from %s", desc_tokenizer_path)
    desc_tokenizer = AutoTokenizer.from_pretrained(desc_tokenizer_path)

    description_ids = desc_tokenizer(args.description, return_tensors="pt").input_ids.to(device)
    prompt_ids = prompt_tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    log.info("Generating audio for prompt: %s", args.prompt[:80])
    with torch.no_grad():
        audio = model.generate(
            input_ids=description_ids,
            prompt_input_ids=prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    sampling_rate = model.config.sampling_rate
    waveform = audio.cpu().numpy().squeeze().astype(np.float32)
    waveform = clean_edges(waveform, sampling_rate)
    sf.write(str(args.output), waveform, sampling_rate)

    log.info("Wrote %s (sr=%d, %.2f s)", args.output, sampling_rate, len(waveform) / sampling_rate)


if __name__ == "__main__":
    main()
