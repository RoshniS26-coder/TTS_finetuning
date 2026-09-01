#!/usr/bin/env python3
"""Long-form Marathi narration with a fine-tuned Parler-TTS model.

Splits the story into sentence-sized chunks (Parler-TTS generates best in the
2-20s range it was trained on), generates each chunk with the fixed Sunita
narration caption, inserts a short pause between sentences, and concatenates
everything into a single wav.

Usage:
  python narrate_story.py --text-file /workspace/my_story.txt --output /workspace/story.wav
  python narrate_story.py --text "एक आटपाट नगर होतं."
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
from transformers import AutoTokenizer, set_seed

# Parler's classifier-free-guidance decode path passes `padding_mask` to the DAC codec,
# but transformers' DacModel.forward() doesn't accept it -> drop it (no-op without guidance).
try:
    from transformers import DacModel as _DacModel
    _dac_fwd = _DacModel.forward
    def _dac_fwd_patched(self, *a, **k):
        k.pop("padding_mask", None)
        return _dac_fwd(self, *a, **k)
    _DacModel.forward = _dac_fwd_patched
except Exception:
    pass

DEFAULT_MODEL_DIR = Path("/workspace/marathi_tts_output")
DEFAULT_OUTPUT = Path("/workspace/story.wav")
DESCRIPTION = (
    # MUST match the v5 training caption byte-for-byte (build_multispeaker_dataset.py
    # CAPTION with name=Sunita, pace=moderate) — Parler voice/style fidelity degrades
    # if the inference description diverges from what was seen in training.
    "Sunita narrates a children's story in an expressive, warm, animated "
    "storytelling tone with emotional variation, at a moderate pace. "
    "Very clear audio with no background noise."
)
# Backstop only: ~30s of DAC tokens (>20s training max) so a runaway tail can't
# spill on; EOS still does the real stopping. Not meant to truncate real speech.
MAX_NEW_TOKENS = 2580

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("narrate")


def clean_text_for_tts(s: str) -> str:
    """Strip characters the model can't speak, which otherwise cause hallucinated
    words / mispronunciation (e.g. a title '**राजू – एक छोटा सुपरहिरो**' with markdown
    bold and an en-dash makes it drop 'राजू' and invent 'बाघ'). Removes markdown
    emphasis/heading/list markers and normalises dashes to a comma-pause."""
    s = re.sub(r"[*_`#>~]+", "", s)                 # markdown emphasis/heading/quote
    s = re.sub(r"^\s*[-•]\s+", "", s)                # leading list bullets
    s = s.replace("–", ", ").replace("—", ", ")      # en/em dash -> pause
    s = re.sub(r"[|<>\[\]{}\"]", " ", s)             # stray brackets/pipes/quotes
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip(" ,")


def split_sentences(text: str) -> list[str]:
    """Split on Devanagari danda / western sentence enders and blank lines, then
    clean each chunk of non-speakable markup."""
    # normalise newlines into sentence breaks, then split on enders
    rough = re.split(r"(?<=[।.!?])\s+|\n+", text.strip())
    cleaned = (clean_text_for_tts(s) for s in rough)
    return [s for s in cleaned if s and s.strip()]


def clean_edges(
    wav: np.ndarray,
    sr: int,
    lead_fade_ms: int = 12,
    trail_fade_ms: int = 50,
    pad_ms: int = 30,
    silence_thresh: float = 1e-3,
    trail_energy_ratio: float = 0.07,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Trim leading/trailing near-silence and apply raised-cosine fades.

    Parler-TTS tends to emit a low-level onset artefact and, at the END, a
    hallucinated buzzy/echoey tail (e.g. an elongated fricative) before it stops.
    The amplitude-only silence trim below catches true silence but NOT that tail
    (it's actually voiced, just quieter than the real speech before it) — so a
    second pass walks backward through short RMS-energy frames and hard-cuts at
    the last frame that's still at least `trail_energy_ratio` of the clip's peak
    energy, discarding anything after it instead of just fading over it. The
    trailing fade is then short (~50 ms) since we're cutting close to real
    speech, not smoothing a long buzzy tail. Heuristic, not a true hallucination
    detector: a tail as loud as real speech won't be caught by this.
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

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    n_frames = (wav.size - frame_len) // hop_len + 1 if wav.size > frame_len else 0
    if n_frames > 1:
        rms = np.array(
            [
                np.sqrt(np.mean(wav[i * hop_len : i * hop_len + frame_len].astype(np.float64) ** 2))
                for i in range(n_frames)
            ]
        )
        peak_rms = rms.max()
        if peak_rms > 0:
            speech_frames = np.nonzero(rms > trail_energy_ratio * peak_rms)[0]
            if speech_frames.size > 0:
                last_speech_frame = int(speech_frames[-1])
                cut = min(wav.size, last_speech_frame * hop_len + frame_len + pad)
                if cut < wav.size:
                    wav = wav[:cut]

    nl = int(sr * lead_fade_ms / 1000.0)
    nt = int(sr * trail_fade_ms / 1000.0)
    if nl > 0 and wav.size > nl:
        wav[:nl] *= (0.5 * (1 - np.cos(np.linspace(0, np.pi, nl)))).astype(np.float32)
    if nt > 0 and wav.size > nt:
        wav[-nt:] *= (0.5 * (1 + np.cos(np.linspace(0, np.pi, nt)))).astype(np.float32)
    return wav.astype(np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-file", type=Path, help="UTF-8 file with the story")
    ap.add_argument("--text", type=str, help="inline text (alternative to --text-file)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--description", default=DESCRIPTION)
    ap.add_argument("--gap-ms", type=int, default=350, help="silence between sentences (ms)")
    ap.add_argument("--device", default=None, help="cuda:0 / mps / cpu (default: auto)")
    # Voice-consistency controls. Each sentence is an INDEPENDENT generation and Parler
    # has no persistent speaker vector, so default stochastic sampling makes the timbre
    # drift chunk-to-chunk. Re-seeding with the SAME seed before every chunk gives each
    # one the same RNG start -> steadier voice across the whole story. Lower temperature
    # (or --greedy) further tames dramatic character-voice jumps (e.g. the "crow" shift).
    ap.add_argument("--seed", type=int, default=42,
                    help="reset before EACH chunk for cross-sentence voice consistency")
    ap.add_argument("--no-seed", action="store_true",
                    help="do NOT reset the RNG per chunk — pure Parler default (fresh randomness "
                         "each chunk). Lets timbre vary naturally instead of anchoring to one seed.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature; lower = steadier voice (ignored with --greedy)")
    ap.add_argument("--greedy", action="store_true",
                    help="deterministic greedy decoding (do_sample=False) for maximum consistency")
    ap.add_argument("--top-p", type=float, default=None,
                    help="nucleus sampling: keep smallest token set with cumulative prob >= top_p "
                         "(e.g. 0.8). Constrains voice deviation WITHOUT the low-temp buzz.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="restrict sampling to the top_k tokens each step (e.g. 50)")
    ap.add_argument("--guidance-scale", type=float, default=None,
                    help="classifier-free guidance: >1 forces stronger adherence to the voice caption "
                         "(e.g. 3.0-4.5), which can reduce cross-sentence timbre drift")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        sys.exit("ERROR: provide --text-file or --text")

    # Accept a local fine-tuned dir OR a HuggingFace model id (e.g.
    # ai4bharat/indic-parler-tts) so the same script can A/B the base model.
    model_ref = str(args.model_dir)
    if not args.model_dir.is_dir() and "/" not in model_ref.strip("/"):
        sys.exit(f"ERROR: not a local model dir or HF model id: {model_ref}")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Loading model from %s on %s", args.model_dir, device)
    model = ParlerTTSForConditionalGeneration.from_pretrained(str(args.model_dir)).to(device)
    prompt_tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    desc_path = model.config.text_encoder._name_or_path
    desc_tok = AutoTokenizer.from_pretrained(desc_path)
    sr = model.config.sampling_rate

    desc_ids = desc_tok(args.description, return_tensors="pt").input_ids.to(device)

    chunks = split_sentences(text)
    log.info("Split story into %d sentence chunks", len(chunks))
    if not chunks:
        sys.exit("ERROR: no text to narrate after splitting")

    gap = np.zeros(int(sr * args.gap_ms / 1000.0), dtype=np.float32)
    pieces: list[np.ndarray] = []

    # Generation mode: greedy = maximally consistent voice; else lower-temperature
    # sampling. Combined with per-chunk re-seeding (below) this is what keeps the
    # narrator's voice steady across sentences instead of drifting each chunk.
    seed_desc = "no per-chunk seed (fresh randomness)" if args.no_seed else f"per-chunk seed={args.seed}"
    if args.greedy:
        gen_kwargs = {"do_sample": False}
        log.info("Decoding: GREEDY (deterministic) | %s", seed_desc)
    else:
        gen_kwargs = {"do_sample": True, "temperature": args.temperature}
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            gen_kwargs["top_k"] = args.top_k
        extra = "".join(f" top_p={args.top_p}" if args.top_p is not None else "") + \
                "".join(f" top_k={args.top_k}" if args.top_k is not None else "")
        log.info("Decoding: sampling temp=%.2f%s | %s", args.temperature, extra, seed_desc)
    if args.guidance_scale is not None:
        gen_kwargs["guidance_scale"] = args.guidance_scale
        log.info("Classifier-free guidance_scale=%.1f (stronger caption adherence)", args.guidance_scale)

    for i, chunk in enumerate(chunks, 1):
        log.info("[%d/%d] %s", i, len(chunks), chunk[:60])
        try:
            # Reset the SAME seed before every chunk so each sentence starts from an
            # identical RNG state -> consistent timbre across the whole narration.
            # --no-seed skips this for pure Parler-default behaviour (fresh randomness).
            if not args.no_seed:
                set_seed(args.seed)
            prompt_ids = prompt_tok(chunk, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                audio = model.generate(
                    input_ids=desc_ids,
                    prompt_input_ids=prompt_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    **gen_kwargs,
                )
            wav = audio.cpu().numpy().squeeze().astype(np.float32)
            if wav.ndim != 1 or wav.size == 0:
                log.warning("  skipped (empty/invalid audio)")
                continue
            wav = clean_edges(wav, sr)
            pieces.append(wav)
            pieces.append(gap)
        except Exception as exc:  # keep going; one bad sentence shouldn't kill the run
            log.warning("  chunk %d failed: %s", i, exc)

    if not pieces:
        sys.exit("ERROR: no audio generated")

    full = np.concatenate(pieces)
    sf.write(str(args.output), full, sr)
    log.info("Wrote %s (sr=%d, %.1f s, %d chunks)", args.output, sr, len(full) / sr, len(chunks))


if __name__ == "__main__":
    main()
