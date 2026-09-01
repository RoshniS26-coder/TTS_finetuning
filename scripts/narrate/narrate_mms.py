"""Narrate a Marathi text file with Meta's MMS-TTS (facebook/mms-tts-mar).

MMS-TTS is a VITS single-speaker model per language. Unlike Parler there is NO
voice description / caption — it maps text directly to a waveform. Public repo,
no HF login required.

  # run from repo root:
  python scripts/narrate/narrate_mms.py --text-file story_texts/lion_story.txt --output model_outputs/mms_marathi/lion_mms.wav
"""
from __future__ import annotations
import argparse, logging, re, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from transformers import VitsModel, AutoTokenizer

MODEL = "facebook/mms-tts-mar"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mms")


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[।.!?])\s+|\n+", text.strip()) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", type=Path)
    ap.add_argument("--text", type=str)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--output", type=Path, default=Path("output_mms.wav"))
    ap.add_argument("--gap-ms", type=int, default=350)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        sys.exit("ERROR: provide --text-file or --text")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    chunks = split_sentences(text)
    if not chunks:
        sys.exit("ERROR: no text after splitting")
    log.info("%d chunks | device=%s | model=%s", len(chunks), device, args.model)

    log.info("Loading %s ...", args.model)
    model = VitsModel.from_pretrained(args.model).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    if getattr(tok, "is_uroman", False):
        log.warning("Tokenizer expects romanized input (uroman) — Devanagari may be mishandled.")
    sr = model.config.sampling_rate
    gap = np.zeros(int(sr * args.gap_ms / 1000.0), dtype=np.float32)

    pieces = []
    for i, c in enumerate(chunks, 1):
        log.info("  [%d/%d] %s", i, len(chunks), c[:50])
        inp = tok(c, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                out = model(**inp).waveform
            wav = out.cpu().numpy().squeeze().astype(np.float32)
            if wav.ndim == 1 and wav.size:
                pieces += [wav, gap]
            else:
                log.warning("    skipped (empty audio)")
        except Exception as exc:
            log.warning("    chunk %d failed: %s", i, exc)

    full = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), full, sr)
    log.info("Wrote %s (%.1fs @ %dHz)", args.output, len(full) / sr, sr)


if __name__ == "__main__":
    main()
