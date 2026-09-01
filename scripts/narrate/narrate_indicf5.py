"""Narrate a Marathi text file with AI4Bharat IndicF5 (flow-matching, voice-clone).

IndicF5 is zero-shot: it CLONES the voice in a short reference clip, so it needs
--ref-audio (a clean 3-10s wav) and --ref-text (that clip's exact transcript).
Gated repo -> needs an HF token with access to ai4bharat/IndicF5.

  # run from repo root:
  python scripts/narrate/narrate_indicf5.py --text-file story_texts/lion_story.txt \
      --ref-audio ref_clip.wav --ref-text "एकदा एक मोठा सिंह जंगलात झोपला होता." \
      --output model_outputs/indic_f5/lion_indicf5.wav
"""
from __future__ import annotations
import argparse, logging, re, sys
from pathlib import Path
import numpy as np, soundfile as sf
import torch, torchaudio

# torchaudio>=2.9 routes .load() through torchcodec, which isn't installed and is
# incompatible with the system's FFmpeg 8. f5_tts only uses torchaudio.load() to
# read the reference WAV -> patch it to use soundfile (already works) instead.
def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
    data, sr = sf.read(str(uri), dtype="float32", always_2d=True)  # (frames, channels)
    t = torch.from_numpy(data).T  # (channels, frames)
    if num_frames and num_frames > 0:
        t = t[:, frame_offset:frame_offset + num_frames]
    elif frame_offset:
        t = t[:, frame_offset:]
    return (t if channels_first else t.T), sr
torchaudio.load = _sf_load

from transformers import AutoModel

MODEL = "ai4bharat/IndicF5"
SR = 24000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("f5")


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[।.!?])\s+|\n+", text.strip()) if s.strip()]


def to_float32(a):
    a = np.asarray(a).squeeze()
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    return a.astype(np.float32)


def clean_edges(wav, sr, fade_ms=15, pad_ms=30, silence_thresh=1e-3):
    """Trim leading/trailing near-silence and fade so chunks splice click-free.

    Same edge fix used in the Parler scripts: the generator rarely ends exactly
    at zero amplitude, so splicing raw output into the silence gap clicks at
    every sentence boundary. Trimming dead edges + a raised-cosine fade makes
    each chunk start/end at zero.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", type=Path)
    ap.add_argument("--text", type=str)
    ap.add_argument("--ref-audio", type=Path, required=True)
    ap.add_argument("--ref-text", type=str, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--output", type=Path, default=Path("output_indicf5.wav"))
    ap.add_argument("--gap-ms", type=int, default=350)
    args = ap.parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        sys.exit("ERROR: provide --text-file or --text")

    chunks = split_sentences(text)
    if not chunks:
        sys.exit("ERROR: no text after splitting")
    log.info("%d chunks | model=%s | ref=%s", len(chunks), args.model, args.ref_audio)

    log.info("Loading %s ...", args.model)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    # CRITICAL: without this the flow-matching runs on CPU (~minutes/sentence).
    # On GPU it's a few seconds/sentence.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = model.to(device)
    except Exception as exc:
        log.warning("could not move model to %s (%s) — may stay on CPU", device, exc)
    log.info("model device: %s", device)
    gap = np.zeros(int(SR * args.gap_ms / 1000.0), dtype=np.float32)

    pieces = []
    for i, c in enumerate(chunks, 1):
        log.info("  [%d/%d] %s", i, len(chunks), c[:50])
        try:
            audio = model(c, ref_audio_path=str(args.ref_audio), ref_text=args.ref_text)
            wav = to_float32(audio)
            if wav.ndim == 1 and wav.size:
                pieces += [clean_edges(wav, SR), gap]
            else:
                log.warning("    skipped (empty audio)")
        except Exception as exc:
            log.warning("    chunk %d failed: %s", i, exc)

    full = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), full, SR)
    log.info("Wrote %s (%.1fs @ %dHz)", args.output, len(full) / SR, SR)


if __name__ == "__main__":
    main()
