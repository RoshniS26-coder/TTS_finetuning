#!/usr/bin/env python3
"""Diagnose 'stuck at edges': generate ONE sentence and save BOTH the raw model
output and the edge-cleaned version, so we can tell whether the artifact is in the
MODEL (raw already stuck/droning) or the post-processing/concatenation (raw is clean).

Run on the pod:
  python diag_edges.py --model-dir /root/marathi_tts_output \
      --description "Usha narrates a children's story in an expressive, warm, animated storytelling tone with emotional variation, at a moderate pace. Very clear audio with no background noise." \
      --prompt "एका जंगलात एक बलाढ्य सिंह राहत होता." --out-prefix /root/diag_usha

Listen to <prefix>_raw.wav vs <prefix>_clean.wav:
  - raw already has a stuck/droning tail  -> MODEL (undertrained, weak EOS). Fix = retrain.
  - raw is clean, only concat sounds bad  -> post-processing. Fix = clean_edges / gaps.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# import the SAME clean_edges the narrator uses, to test the real path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from narrate_story import clean_edges  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out-prefix", default="/root/diag")
    ap.add_argument("--max-new-tokens", type=int, default=2580,
                    help="lower this (e.g. 600) to test if a high cap causes a long drone tail")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    ptok = AutoTokenizer.from_pretrained(args.model_dir)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate

    desc = dtok(args.description, return_tensors="pt").input_ids.to(device)
    prompt = ptok(args.prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        audio = model.generate(input_ids=desc, prompt_input_ids=prompt,
                               max_new_tokens=args.max_new_tokens)
    raw = audio.cpu().numpy().squeeze().astype(np.float32)

    raw_path = f"{args.out_prefix}_raw.wav"
    clean_path = f"{args.out_prefix}_clean.wav"
    sf.write(raw_path, raw, sr)
    sf.write(clean_path, clean_edges(raw, sr), sr)

    # crude tail-energy report: RMS of the last 0.5s vs the loudest 0.5s window
    win = int(0.5 * sr)
    if raw.size > win:
        tail_rms = float(np.sqrt(np.mean(raw[-win:] ** 2)))
        peak_rms = max(float(np.sqrt(np.mean(raw[i:i + win] ** 2)))
                       for i in range(0, raw.size - win, win))
        print(f"raw: {raw.size / sr:.2f}s | tail-0.5s RMS={tail_rms:.4f} | peak-0.5s RMS={peak_rms:.4f} "
              f"| tail/peak={tail_rms / (peak_rms + 1e-9):.2f}  (high ratio => droning tail = MODEL)")
    print("wrote", raw_path, "and", clean_path, f"(sr={sr})")


if __name__ == "__main__":
    main()
