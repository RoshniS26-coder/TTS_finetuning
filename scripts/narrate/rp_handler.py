#!/usr/bin/env python3
"""RunPod Serverless QUEUE handler for the Betacraft TTS model.

This replaces the FastAPI/uvicorn entrypoint used by the old Load Balancer
endpoint. A Queue endpoint has no HTTP server inside the container at all:
runpod.serverless.start() polls RunPod's job queue, calls handler() per job,
and RunPod serialises the return value back to the caller as JSON.

Why the migration (see also betacraft_core.py's docstring):

  * NO GATEWAY TIMEOUT. The Load Balancer cut connections at ~30-40s, which is
    why lib/tts.ts had to chop every request into 2-sentence batches and still
    hit intermittent 502s on slow generations. /runsync waits far longer.
  * REAL JOB DISTRIBUTION. The LB endpoint autoscaled on "request count = 4",
    so a 2- or 3-request burst never triggered a scale-up and every request
    piled onto the one warm worker. A queue distributes jobs to whatever
    workers exist, which is what makes client-side fan-out actually parallel.
  * ASYNC FOR FREE. Long stories can use /run + /status polling on this same
    endpoint instead of holding a connection open.

Request:
    {"input": {"text": "...",              # required (unless warmup)
               "model": "betacraft",       # betacraft (fine-tune) | base (indic-parler-tts)
               "language": "mr",           # mr | hi | en
               "temperature": 0.65,
               "seed": 42,
               "gap_ms": 350,
               "pack_words": 24,                   # 0 = one unit per sentence
               "match_loudness_db": 3.0,           # 0 = no level matching
               "top_p": 0.9,                       # optional override
               "repetition_penalty": null,         # optional override, null = off
               "no_repeat_ngram_size": null,       # optional override, null = off
               "warmup": false}}

Response:
    {"audio_b64": "<base64 WAV, 16-bit PCM>",
     "sr": 44100,
     "duration_sec": 5.2,
     "encoding": "wav/pcm16",
     "sentences": [ ...per-sentence timing/diagnostics... ]}
"""
from __future__ import annotations

import base64
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runpod  # noqa: E402

from betacraft_core import (  # noqa: E402
    DEFAULT_MODEL,
    LOUDNESS_MAX_GAIN_DB,
    MODELS,
    PACK_TARGET_WORDS,
    encode_wav,
    get_bundle,
    synthesize,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rp_handler")

# RunPod caps how large a job response can be. Base64 inflates bytes by ~33%,
# so a single sentence (a few hundred KB of 16-bit PCM) is comfortably fine
# while a full long story is NOT — that path needs to upload to object storage
# and return a URL instead. Warn loudly rather than let RunPod reject the job
# with a less obvious error.
RESPONSE_WARN_BYTES = 8 * 1024 * 1024

# The DEFAULT model is loaded at import time, i.e. during RunPod cold start —
# including its warmup generation. This is deliberate: that cost is paid once
# per worker, where RunPod's own worker-init accounting expects it, instead of
# landing on whichever unlucky request arrives first.
#
# The second model is NOT loaded here. Loading both would roughly double cold
# start for every worker, including the majority that only ever serve the
# default — and cold start is already the dominant latency problem. It loads on
# first use instead, or ahead of time via {"warmup": true, "model": "base"}.
_start = time.time()
BUNDLE = get_bundle(DEFAULT_MODEL)
log.info("Worker ready in %.1fs — polling for jobs", time.time() - _start)


# Sentinel meaning "caller did not mention this knob, use the server default".
# Distinct from an explicit null/None, which means "disable this parameter".
_UNSET = -1.0


def _override(payload: dict, key: str):
    if key not in payload:
        return _UNSET
    return payload[key]


def handler(job):
    payload = job.get("input") or {}

    model_name = payload.get("model", DEFAULT_MODEL)
    if model_name not in MODELS:
        return {"error": f"model must be one of {list(MODELS)}"}

    # A no-op job used to force a cold worker to spin up and load the model
    # BEFORE any audio is actually needed — the app fires these when a child
    # opens the story screen, so the 3-4 min cold start overlaps with them
    # picking a character and the LLM writing the first turn. Passing "model"
    # also pre-loads the non-default model, which otherwise loads lazily on its
    # first real request.
    if payload.get("warmup"):
        bundle = get_bundle(model_name)
        return {"status": "warm", "model": model_name, "device": bundle["device"], "sr": bundle["sr"]}

    text = (payload.get("text") or "").strip()
    if not text:
        return {"error": "input.text is required"}

    bundle = get_bundle(model_name)
    language = payload.get("language", "mr")
    if language not in bundle["speakers"]:
        return {"error": f"model '{model_name}' supports {list(bundle['speakers'])}, not '{language}'"}

    started = time.time()
    try:
        wav, sr, meta = synthesize(
            bundle,
            text=text,
            language=language,
            temperature=payload.get("temperature"),
            seed=int(payload.get("seed", 42)),
            gap_ms=int(payload.get("gap_ms", 350)),
            # pack_words=0 disables sentence packing (one unit per sentence) —
            # the pre-2026-08-31 behaviour, kept reachable for A/B.
            pack_words=int(payload.get("pack_words", PACK_TARGET_WORDS)),
            match_loudness_db=float(payload.get("match_loudness_db", LOUDNESS_MAX_GAIN_DB)),
            gen_overrides={
                "top_p": _override(payload, "top_p"),
                "repetition_penalty": _override(payload, "repetition_penalty"),
                "no_repeat_ngram_size": _override(payload, "no_repeat_ngram_size"),
            },
        )
    except (ValueError, RuntimeError) as exc:
        log.warning("Job failed: %s", exc)
        return {"error": str(exc)}

    audio_bytes = encode_wav(wav, sr)
    if len(audio_bytes) > RESPONSE_WARN_BYTES:
        log.warning(
            "Response payload is %.1f MB before base64 — close to RunPod's job "
            "response limit. Long-form text should upload to object storage and "
            "return a URL instead of inlining audio.",
            len(audio_bytes) / 1024 / 1024,
        )

    duration_sec = len(wav) / sr
    log.info(
        "Job done: model=%s, %.2fs audio, %d unit(s), %.2fs wall (rtf %.2f), %.0f KB wav",
        model_name, duration_sec, len(meta), time.time() - started,
        (time.time() - started) / duration_sec if duration_sec else 0.0,
        len(audio_bytes) / 1024,
    )
    return {
        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
        "sr": sr,
        "duration_sec": round(duration_sec, 2),
        "encoding": "wav/pcm16",
        "model": model_name,
        "sentences": meta,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
