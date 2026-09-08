#!/usr/bin/env python3
"""FastAPI wrapper around the fine-tuned Betacraft Parler-TTS model (mr/hi/en).

As of 2026-08-31 this is the LOCAL DEVELOPMENT entrypoint. The deployed
RunPod path is rp_handler.py (a Serverless QUEUE handler); both call the same
generation code in betacraft_core.py, so behaviour cannot drift between them.

Kept because it's genuinely useful: it lets you exercise the model over plain
HTTP on the Mac without a RunPod round trip, and it still serves the old
Load Balancer endpoint should you need to fall back to it. See
betacraft_core.py and rp_handler.py for why the deployment moved to a queue.

Run:
  venv/bin/uvicorn scripts.narrate.api_server:app --host 0.0.0.0 --port 8008
  (run from the repo root so the "scripts.narrate" package path resolves)
"""
from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from betacraft_core import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SEED,
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
log = logging.getLogger("betacraft_api")

state: dict = {}

# PyTorch's MPS (Metal) / CUDA backends are not safe for concurrent GPU dispatch
# from multiple threads on one device — FastAPI runs sync endpoints in a thread
# pool, so without this, two concurrent /synthesize calls can hard-crash the
# process (observed: no Python traceback, process just vanishes). This lock
# serialises generation so concurrent requests queue instead of racing.
#
# rp_handler.py needs no equivalent: RunPod runs one job per worker, and
# concurrency across users comes from running multiple worker replicas.
GENERATION_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only the default model is loaded up front; the other loads on first use
    # via get_bundle(). See betacraft_core._BUNDLES for why.
    state["bundle"] = get_bundle(DEFAULT_MODEL)
    log.info("Serving requests.")
    yield
    state.clear()


app = FastAPI(title="Betacraft TTS (local dev)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any non-local exposure
    allow_methods=["*"],
    allow_headers=["*"],
)


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str = Field(
        DEFAULT_MODEL,
        description="'betacraft' (fine-tuned, mr/hi trained) or 'base' (stock indic-parler-tts)",
    )
    language: str = Field("mr", description="mr | hi | en (speaker depends on the model)")
    temperature: float | None = Field(None, gt=0, description="None -> server default (0.65)")
    seed: int = Field(DEFAULT_SEED, description="reset before EACH sentence for cross-sentence voice consistency")
    gap_ms: int = Field(350, ge=0, description="silence between units (ms)")
    pack_words: int = Field(
        PACK_TARGET_WORDS, ge=0,
        description="sentence-pack into ~N-word units to match training clip length; 0 = one unit per sentence",
    )
    match_loudness_db: float = Field(
        LOUDNESS_MAX_GAIN_DB, ge=0,
        description="max +/- dB to level-match units toward the median; 0 disables",
    )
    # Sampling overrides, mirroring rp_handler.py. Omit to use the server
    # default; send null to disable a parameter outright. Exists so the two
    # suspect logits processors (repetition_penalty / no_repeat_ngram_size,
    # both off by default now — see betacraft_core.py) can be A/B'd per request.
    top_p: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None


@app.get("/health")
def health():
    if "bundle" not in state:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {
        "status": "ok",
        "device": state["bundle"]["device"],
        "models": {name: list(spec["speakers"]) for name, spec in MODELS.items()},
    }


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/synthesize")
def synthesize_endpoint(req: SynthesizeRequest):
    if "bundle" not in state:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    fields = req.model_fields_set
    overrides = {
        key: (getattr(req, key) if key in fields else -1.0)
        for key in ("top_p", "repetition_penalty", "no_repeat_ngram_size")
    }

    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"model must be one of {list(MODELS)}")

    log.info(
        "Request model=%s lang=%s (waiting for GPU lock: %s)",
        req.model, req.language, GENERATION_LOCK.locked(),
    )
    with GENERATION_LOCK:
        try:
            wav, sr, _meta = synthesize(
                get_bundle(req.model),
                text=req.text,
                language=req.language,
                temperature=req.temperature,
                seed=req.seed,
                gap_ms=req.gap_ms,
                pack_words=req.pack_words,
                match_loudness_db=req.match_loudness_db,
                gen_overrides=overrides,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return Response(content=encode_wav(wav, sr), media_type="audio/wav")
