#!/usr/bin/env python3
"""Streaming narration demo for the v5 fine-tuned Parler-TTS model.

Shows the "live" latency story. Model is loaded ONCE. Two modes:

  * sentence streaming (default): generate each sentence and write it the instant it's
    ready, so a player could start after sentence 1. Reports time-to-first-audio (TTFA)
    and per-sentence real-time-factor (RTF) so you can SEE the live latency.

  * --token-stream: use ParlerTTSStreamer to emit sub-sentence audio chunks DURING a
    sentence's generation (lowest possible first-audio latency). Falls back to sentence
    streaming if the streamer isn't available in this parler-tts build.

Speaker = the NAME in the caption; keep the caption identical to v5 training (only
{name} + {pace} vary). Language follows the story text (Marathi vs Hindi).

Examples (run on the pod):
  python narrate_stream.py --text-file /workspace/kawda_test.txt --name Sunita --attn sdpa
  python narrate_stream.py --text-file /workspace/kawda_test.txt --name Sunita --token-stream
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from threading import Thread

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

CAPTION = ("{name} narrates a children's story in an expressive, warm, animated "
           "storytelling tone with emotional variation, at a {pace} pace. "
           "Very clear audio with no background noise.")
MAX_NEW_TOKENS = 2580

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("narrate_stream")


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


def load_model(model_dir: Path, attn: str, device: str):
    try:
        m = ParlerTTSForConditionalGeneration.from_pretrained(
            str(model_dir), attn_implementation=attn).to(device)
        log.info("Loaded model (attn_implementation=%s)", attn)
    except Exception as exc:  # build may not support the requested impl
        log.warning("attn=%s failed (%s) — falling back to default", attn, exc)
        m = ParlerTTSForConditionalGeneration.from_pretrained(str(model_dir)).to(device)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-file", type=Path, help="UTF-8 story file")
    ap.add_argument("--text", type=str, help="inline text (alternative to --text-file)")
    ap.add_argument("--name", default="Sunita", help="speaker name in the caption (Sunita/Usha/Maya/Sarita)")
    ap.add_argument("--pace", default="moderate", help="pace word in the caption (trained: moderate)")
    ap.add_argument("--model-dir", type=Path, default=Path("/workspace/marathi_tts_output"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/stream_out"))
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                    help="attention impl at model load (sdpa = built-in free speedup)")
    ap.add_argument("--token-stream", action="store_true",
                    help="sub-sentence token streaming via ParlerTTSStreamer (lowest first-audio latency)")
    ap.add_argument("--play-seconds", type=float, default=0.5,
                    help="token-stream chunk size in seconds (smaller = lower latency, more overhead)")
    ap.add_argument("--gap-ms", type=int, default=350, help="silence between sentences (ms)")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (default: auto)")
    args = ap.parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        sys.exit("ERROR: provide --text-file or --text")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_dir, args.attn, device)
    prompt_tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    desc = CAPTION.format(name=args.name, pace=args.pace)
    desc_ids = desc_tok(desc, return_tensors="pt").input_ids.to(device)

    # Optional sub-sentence streamer
    streamer_cls = None
    if args.token_stream:
        try:
            from parler_tts import ParlerTTSStreamer
            streamer_cls = ParlerTTSStreamer
        except Exception as exc:
            log.warning("ParlerTTSStreamer unavailable (%s) — using sentence streaming", exc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sentences = split_sentences(text)
    if not sentences:
        sys.exit("ERROR: no text to narrate after splitting")
    log.info("Voice=%s pace=%s | %d sentences | mode=%s",
             args.name, args.pace, len(sentences), "token-stream" if streamer_cls else "sentence-stream")

    gap = np.zeros(int(sr * args.gap_ms / 1000.0), dtype=np.float32)
    t0 = time.time()
    ttfa = None
    all_pieces: list[np.ndarray] = []

    for i, sent in enumerate(sentences, 1):
        pid = prompt_tok(sent, return_tensors="pt").input_ids.to(device)
        s_start = time.time()

        if streamer_cls is not None:
            # Token-level streaming: model.generate runs in a thread, we consume chunks live.
            fr = getattr(getattr(model, "audio_encoder", None), "config", None)
            frame_rate = getattr(fr, "frame_rate", None) or 86
            play_steps = max(1, int(frame_rate * args.play_seconds))
            streamer = streamer_cls(model, device=device, play_steps=play_steps)
            kwargs = dict(input_ids=desc_ids, prompt_input_ids=pid,
                          streamer=streamer, max_new_tokens=MAX_NEW_TOKENS)
            th = Thread(target=model.generate, kwargs=kwargs)
            th.start()
            chunks: list[np.ndarray] = []
            for chunk in streamer:
                arr = np.asarray(chunk, dtype=np.float32).squeeze()
                if arr.size == 0:
                    continue
                if ttfa is None:
                    ttfa = time.time() - t0
                    log.info("  >>> FIRST AUDIO CHUNK at T+%.2fs", ttfa)
                chunks.append(arr)
            th.join()
            wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        else:
            with torch.no_grad():
                audio = model.generate(input_ids=desc_ids, prompt_input_ids=pid, max_new_tokens=MAX_NEW_TOKENS)
            wav = audio.cpu().numpy().squeeze().astype(np.float32)
            if ttfa is None:
                ttfa = time.time() - t0
                log.info("  >>> FIRST SENTENCE ready at T+%.2fs (playback can start here)", ttfa)

        wav = clean_edges(wav, sr)
        dur = len(wav) / sr if sr else 0.0
        gen = time.time() - s_start
        rtf = gen / dur if dur > 0 else float("inf")
        out = args.out_dir / f"{args.name.lower()}_{i:03d}.wav"
        sf.write(str(out), wav, sr)
        log.info("[%d/%d] gen %.1fs / audio %.1fs (RTF %.2f) -> %s | %s",
                 i, len(sentences), gen, dur, rtf, out.name, sent[:45])
        all_pieces.append(wav)
        all_pieces.append(gap)

    full = np.concatenate(all_pieces) if all_pieces else np.zeros(0, np.float32)
    full_path = args.out_dir / f"{args.name.lower()}_FULL.wav"
    sf.write(str(full_path), full, sr)
    total = time.time() - t0
    audio_s = len(full) / sr if sr else 0.0
    log.info("Done. TTFA=%.2fs | total %.1fs wall for %.1fs audio (overall RTF %.2f) -> %s",
             ttfa if ttfa is not None else -1, total, audio_s,
             (total / audio_s) if audio_s > 0 else 0.0, full_path)


if __name__ == "__main__":
    main()
