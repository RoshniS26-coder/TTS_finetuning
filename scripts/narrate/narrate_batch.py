#!/usr/bin/env python3
"""Batch narration for the v5 fine-tuned Parler-TTS model.

Loads the model ONCE and renders many (speaker x story) combinations. Speaker
identity = the NAME in the caption; keep the rest of the caption byte-identical to
training (build_multispeaker_dataset.py CAPTION) — only NAME and PACE vary, or voice
fidelity drops. All v5 speakers were trained at pace="moderate"; "fairly fast" etc.
is an experiment the model didn't see, so treat those outputs as best-effort.

Language is set by the STORY TEXT, not the caption: Marathi text -> Marathi (Sunita/
Usha/Maya), Hindi text -> Hindi (Sarita).

Run on the pod from /workspace (stories must already be uploaded there):
  python narrate_batch.py --model-dir /workspace/marathi_tts_output --in-dir /workspace --out-dir /workspace/narrations_v5

Edit the lists in the CONFIG block to control exactly what gets generated.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Must match the v5 training caption exactly (only {name} and {pace} vary).
CAPTION = ("{name} narrates a children's story in an expressive, warm, animated "
           "storytelling tone with emotional variation, at a {pace} pace. "
           "Very clear audio with no background noise.")
# Backstop only (~30s of DAC tokens > 20s training max); EOS does the real stopping.
MAX_NEW_TOKENS = 2580

# ===================== CONFIG — edit to control the run =====================
# Marathi stories get each of the Marathi voices below.
# v6 model (Sunita + Divya). Marathi stories -> Sunita. (story_superhero.txt is MARATHI
# content despite the title.)
MARATHI_STORIES = [
    "kawda_test.txt",
    "hushar_sasa_lion_story_test.txt",
    "birds_story_test.txt",
    "story_superhero.txt",
]
MARATHI_VOICES = [("Sunita", "moderate")]

# Hindi story -> Divya (the Hindi voice in v6).
HINDI_JOBS = [("Divya", "moderate", "hindi_test_Story.txt")]

# One-off experiments: (name, pace, story_file). Empty for v6.
# (To cross-test the Hindi voice on Marathi, add ("Divya","moderate","story_superhero.txt").)
EXTRA_JOBS = []
# ===========================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("narrate_batch")


def split_sentences(text: str) -> list[str]:
    rough = re.split(r"(?<=[।.!?])\s+|\n+", text.strip())
    return [s.strip() for s in rough if s and s.strip()]


def clean_edges(wav: np.ndarray, sr: int, fade_ms: int = 15, pad_ms: int = 30,
                silence_thresh: float = 1e-3) -> np.ndarray:
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=Path("/workspace/marathi_tts_output"))
    ap.add_argument("--in-dir", type=Path, default=Path("/workspace"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/narrations_v5"))
    ap.add_argument("--gap-ms", type=int, default=350, help="silence between sentences (ms)")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (default: auto)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                    help="attention impl at model load. sdpa = built-in (no install), small free speedup; "
                         "eager = original; flash_attention_2 = needs `pip install flash-attn`.")
    args = ap.parse_args()

    # Build the job list: (name, pace, story_file)
    jobs: list[tuple[str, str, str]] = []
    for story in MARATHI_STORIES:
        for name, pace in MARATHI_VOICES:
            jobs.append((name, pace, story))
    jobs.extend(HINDI_JOBS)
    jobs.extend(EXTRA_JOBS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Loading model from %s on %s (attn=%s)", args.model_dir, device, args.attn)
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            str(args.model_dir), attn_implementation=args.attn).to(device)
    except Exception as exc:  # build may not support the requested impl -> fall back
        log.warning("attn_implementation=%s failed (%s) — falling back to default", args.attn, exc)
        model = ParlerTTSForConditionalGeneration.from_pretrained(str(args.model_dir)).to(device)
    prompt_tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    gap = np.zeros(int(sr * args.gap_ms / 1000.0), dtype=np.float32)

    log.info("Planned %d narrations:", len(jobs))
    for name, pace, story in jobs:
        log.info("   %-7s | %-11s | %s", name, pace, story)

    for idx, (name, pace, story) in enumerate(jobs, 1):
        infile = args.in_dir / story
        if not infile.is_file():
            log.warning("[%d/%d] MISSING %s — skipping", idx, len(jobs), infile)
            continue
        desc = CAPTION.format(name=name, pace=pace)
        out = args.out_dir / f"{name.lower()}_{pace.replace(' ', '')}_{infile.stem}.wav"
        log.info("[%d/%d] %s (%s) <- %s -> %s", idx, len(jobs), name, pace, story, out.name)
        desc_ids = desc_tok(desc, return_tensors="pt").input_ids.to(device)
        chunks = split_sentences(infile.read_text(encoding="utf-8"))
        pieces: list[np.ndarray] = []
        for j, chunk in enumerate(chunks, 1):
            try:
                pid = prompt_tok(chunk, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    audio = model.generate(input_ids=desc_ids, prompt_input_ids=pid, max_new_tokens=MAX_NEW_TOKENS)
                wav = audio.cpu().numpy().squeeze().astype(np.float32)
                if wav.ndim == 1 and wav.size > 0:
                    pieces.append(clean_edges(wav, sr))
                    pieces.append(gap)
            except Exception as exc:  # one bad sentence shouldn't kill the whole story
                log.warning("   chunk %d failed: %s", j, exc)
        if pieces:
            full = np.concatenate(pieces)
            sf.write(str(out), full, sr)
            log.info("   wrote %s (%.1fs, %d chunks)", out.name, len(full) / sr, len(chunks))
        else:
            log.warning("   no audio produced for %s", out.name)

    log.info("Done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
