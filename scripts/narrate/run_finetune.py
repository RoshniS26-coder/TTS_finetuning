"""Narrate a Marathi text file with your FINE-TUNED Parler-TTS model.

Your repo is PRIVATE -> run `huggingface-cli login` first.

  # run from repo root:
  python scripts/narrate/run_finetune.py --text-file story_texts/lion_story.txt --output lion_finetuned.wav
"""
from __future__ import annotations
import argparse, logging, re, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

FT_MODEL = "roshni-sorigin/marathi-parler-tts"
DESCRIPTION = ("Sunita narrates with Narration emotion in an expressive, warm storytelling "
               "tone at a moderate pace. Very clear audio with no background noise.")
# Backstop only (~30s of DAC tokens, >20s training max): caps a runaway noisy
# tail; EOS still does the real stopping. Not meant to truncate real speech.
MAX_NEW_TOKENS = 2580

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ft")


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[।.!?])\s+|\n+", text.strip()) if s.strip()]


def clean_edges(wav, sr, fade_ms=15, pad_ms=30, silence_thresh=1e-3):
    """Trim leading/trailing near-silence and fade so chunks splice click-free.

    Parler-TTS rarely ends at zero amplitude and emits a faint onset/tail
    artefact; splicing raw output into the zero-gap clicks at every boundary.
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
    ap.add_argument("--model", default=FT_MODEL)
    ap.add_argument("--description", default=DESCRIPTION)
    ap.add_argument("--output", type=Path, default=Path("output_finetuned.wav"))
    ap.add_argument("--seed", type=int, default=42)
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
    model = ParlerTTSForConditionalGeneration.from_pretrained(args.model).to(device)
    ptok = AutoTokenizer.from_pretrained(args.model)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    desc = dtok(args.description, return_tensors="pt").input_ids.to(device)
    gap = np.zeros(int(sr * args.gap_ms / 1000.0), dtype=np.float32)

    pieces = []
    for i, c in enumerate(chunks, 1):
        log.info("  [%d/%d] %s", i, len(chunks), c[:50])
        torch.manual_seed(args.seed + i)
        pid = ptok(c, return_tensors="pt").input_ids.to(device)
        try:
            with torch.no_grad():
                audio = model.generate(
                    input_ids=desc, prompt_input_ids=pid, max_new_tokens=MAX_NEW_TOKENS
                )
            wav = audio.cpu().numpy().squeeze().astype(np.float32)
            if wav.ndim == 1 and wav.size:
                pieces += [clean_edges(wav, sr), gap]
            else:
                log.warning("    skipped (empty audio)")
        except Exception as exc:
            log.warning("    chunk %d failed: %s", i, exc)

    full = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    sf.write(str(args.output), full, sr)
    log.info("Wrote %s (%.1fs)", args.output, len(full) / sr)


if __name__ == "__main__":
    main()
