#!/usr/bin/env python3
"""Post-hoc voice conversion to REMOVE cross-sentence timbre drift.

Parler generates each sentence with a slightly different voice (no persistent
speaker vector). This forces a whole narration to ONE reference voice using
FreeVC (zero-shot) — content/prosody kept, speaker timbre made uniform.

Pipeline:  Parler (temp 0.7) narration  ->  this script  ->  single-voice audio.

Run on a GPU pod (needs coqui-tts; ~2 GB of models load into RAM/VRAM — do NOT
run on an 8 GB Mac, it OOM-kills).

Setup:
  pip install "coqui-tts[codec]"
Usage:
  python voice_convert.py --source story.wav --reference clean_sunita.wav --output story_vc.wav
  # batch a folder:
  python voice_convert.py --source-dir narrations/ --reference clean_sunita.wav --out-dir narrations_vc/
"""
from __future__ import annotations
import argparse
from pathlib import Path

MODEL = "voice_conversion_models/multilingual/vctk/freevc24"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, help="one narration wav to convert")
    ap.add_argument("--source-dir", type=Path, help="folder of wavs to convert (batch)")
    ap.add_argument("--reference", type=Path, required=True,
                    help="clean wav of the TARGET voice (all output takes this timbre)")
    ap.add_argument("--output", type=Path, help="output wav (with --source)")
    ap.add_argument("--out-dir", type=Path, help="output folder (with --source-dir)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args()

    import torch
    from TTS.api import TTS
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> loading FreeVC on {device} ...", flush=True)
    tts = TTS(MODEL).to(device)

    if not args.reference.is_file():
        raise SystemExit(f"reference not found: {args.reference}")

    if args.source_dir:
        out = args.out_dir or (args.source_dir.parent / f"{args.source_dir.name}_vc")
        out.mkdir(parents=True, exist_ok=True)
        wavs = sorted(args.source_dir.glob("*.wav"))
        print(f">> {len(wavs)} files -> {out}", flush=True)
        for w in wavs:
            dst = out / w.name
            print(f"   {w.name}", flush=True)
            tts.voice_conversion_to_file(source_wav=str(w), target_wav=str(args.reference),
                                         file_path=str(dst))
    else:
        if not args.source or not args.output:
            raise SystemExit("provide --source and --output (or --source-dir/--out-dir)")
        print(f">> {args.source.name} -> {args.output}", flush=True)
        tts.voice_conversion_to_file(source_wav=str(args.source), target_wav=str(args.reference),
                                     file_path=str(args.output))
    print(">> DONE", flush=True)


if __name__ == "__main__":
    main()
