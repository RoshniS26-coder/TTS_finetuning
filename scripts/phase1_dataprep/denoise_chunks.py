#!/usr/bin/env python3
"""Denoise dataset chunks (remove steady background buzz/hum/hiss) with noisereduce.

noisereduce = spectral-gating denoiser, pure numpy/scipy (no torchaudio, no
dependency hell). Good for STEADY buzz. Install once:
    python -m pip install noisereduce soundfile

Workflow (test SMALL first, then full):
    # 1) denoise the first 10 chunks, A/B listen vs originals
    python scripts/phase1_dataprep/denoise_chunks.py \
        --input-dir marathi_dataset_v2/audio --output-dir marathi_dataset_v2_dn/audio --limit 10
    # 2) if clean (buzz gone, no 'musical'/watery artefacts), run all (drop --limit)

--prop-decrease controls strength: 0.75 default (gentle-ish). LOWER (0.5) = gentler
/ fewer artefacts; HIGHER (0.9) = more buzz removed but more artefact risk. For TTS,
err gentle — the model LEARNS any artefact you bake in.

NOTE: removes ACOUSTIC noise only. Does NOT fix self-corrections/disfluencies in the
narration (that's content — needs segment editing). For best quality on the full set,
re-run DeepFilterNet on the pod GPU in a clean env; noisereduce is the local quick test.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("denoise")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None, help="write denoised here (omit if --in-place)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite originals via a temp file (no 2x disk needed). DESTRUCTIVE — "
                         "verify on a sample dir first; there is no undo.")
    ap.add_argument("--start", type=int, default=0, help="skip the first N files (offset into sorted list) — for testing a range")
    ap.add_argument("--limit", type=int, default=0, help="denoise only N files after --start (0=all) — for a quick test")
    ap.add_argument("--prop-decrease", type=float, default=0.5,
                    help="0..1 noise reduction strength. 0.5 default (gentle: keeps voice brightness). "
                         "Higher removes more buzz but muffles/darkens the voice.")
    ap.add_argument("--slowdown", type=float, default=1.0,
                    help="time-stretch via ffmpeg atempo (pitch-preserved, good for speech): "
                         "0.90 = 10%% slower, 0.85 = 15%%, 1.0 = no change.")
    ap.add_argument("--stationary", action="store_true", default=True, help="steady buzz -> stationary (default on)")
    args = ap.parse_args()

    try:
        import noisereduce as nr
    except ImportError as e:
        raise SystemExit(f"Missing dep ({e}). Install: python -m pip install noisereduce soundfile")

    if not args.in_place and args.output_dir is None:
        raise SystemExit("provide --output-dir, or use --in-place")
    files = sorted(args.input_dir.glob("*.wav"))
    if args.start:
        files = files[args.start:]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .wav in {args.input_dir}")
    if not args.in_place:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = "IN-PLACE (overwrite)" if args.in_place else str(args.output_dir)
    log.info("Denoising %d files | prop_decrease=%.2f | %s -> %s",
             len(files), args.prop_decrease, args.input_dir, dest)

    done = 0
    for f in files:
        try:
            wav, sr = sf.read(str(f), dtype="float32", always_2d=True)  # (frames, ch)
            mono = wav.mean(axis=1)                                     # -> mono
            reduced = nr.reduce_noise(y=mono, sr=sr, stationary=args.stationary,
                                      prop_decrease=args.prop_decrease)
            # Restore loudness: denoising removes energy -> RMS-match to the original
            # (peak-guarded against clipping), so the denoised clip isn't quieter.
            o_rms = float(np.sqrt(np.mean(mono ** 2)))
            d_rms = float(np.sqrt(np.mean(reduced ** 2)))
            if o_rms > 1e-8 and d_rms > 1e-8:
                reduced = reduced * (o_rms / d_rms)
            peak = float(np.max(np.abs(reduced))) if reduced.size else 0.0
            if peak > 0.99:
                reduced = reduced * (0.99 / peak)
            # Unified write: denoised -> temp, optional ffmpeg atempo slowdown, then
            # atomically replace the target (original if in-place, else output-dir).
            target = f if args.in_place else (args.output_dir / f.name)
            tmp_dn = target.with_suffix(".dnpre.wav")
            sf.write(str(tmp_dn), reduced.astype(np.float32), sr)
            if args.slowdown != 1.0:
                tmp_out = target.with_suffix(".dnout.wav")
                subprocess.run(["ffmpeg", "-y", "-i", str(tmp_dn), "-filter:a",
                                f"atempo={args.slowdown}", str(tmp_out), "-loglevel", "error"], check=True)
                tmp_dn.unlink(missing_ok=True)
                tmp_out.replace(target)
            else:
                tmp_dn.replace(target)
            done += 1
            if done % 50 == 0:
                log.info("  %d/%d", done, len(files))
        except Exception as exc:  # one bad clip shouldn't kill the batch
            log.warning("  %s failed: %s", f.name, exc)

    log.info("Done: %d/%d denoised -> %s (transcripts unchanged; point the build at this audio dir)",
             done, len(files), args.output_dir)


if __name__ == "__main__":
    main()
