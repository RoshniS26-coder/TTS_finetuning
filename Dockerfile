# syntax=docker/dockerfile:1
# Builds the Betacraft TTS serving image for RunPod Serverless.
# The model weights are baked in at BUILD time (see the RUN step below) so a
# cold-started worker never has to hit the network for them — see CLAUDE.md /
# api_server.py docstring for why that matters for cold-start latency.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Faster/more memory-efficient attention (see betacraft_core.py's _load_model).
# Separate step, not in requirements-api.txt: --no-build-isolation is a
# pip-install CLI flag, not something a requirements.txt line can carry, and
# it's needed here so flash-attn's build can see the already-installed torch.
# RISK: if no prebuilt wheel matches this exact torch/CUDA/Python combo, pip
# compiles from source (can take 20-45+ minutes) or fails outright — `|| true`
# makes that non-fatal to the build, matching _load_model's own runtime
# fallback to default attention if flash-attn isn't installed/working.
# CONFIRMED WORKING 2026-08-31: worker startup logs report
# "Loaded with attn_implementation=flash_attention_2", so the fallback path
# is not being taken in production. If that log line ever disappears, this
# step silently failed and generation just got slower.
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn install failed — continuing without it"

COPY scripts/narrate/ /app/scripts/narrate/

ENV BETACRAFT_MODEL_DIR=roshni-sorigin/mar-hin-betacraft-tts
ENV BETACRAFT_DEVICE=cuda:0
ENV PYTHONPATH=/app
ENV HF_HOME=/app/hf_cache

# Pre-download the model + its description tokenizer into the image's own HF
# cache using a build-time secret, so the token is never written into an image
# layer or history (unlike a plain ENV/ARG, which persists in the image).
# Build with: docker build --secret id=hf_token,env=HF_TOKEN ...
# BOTH models are cached: the fine-tune (default) and stock indic-parler-tts,
# which the app offers as a cheaper/base comparison and which serves English as
# a properly trained language (the fine-tune only trained mr/hi). HF_HUB_OFFLINE
# below means anything NOT cached here cannot be fetched at runtime — a lazy
# load of the base model on a live worker would fail outright, not just be slow.
# A HEREDOC, not `python -c "..."` with line continuations: the backslashes
# collapse every line into ONE logical line, and Python forbids a compound
# statement (the `for` loop) after semicolon-separated statements there, so the
# one-liner form fails with SyntaxError before it can authenticate at all.
# DOWNLOAD-ONLY, deliberately: an earlier version called
# ParlerTTSForConditionalGeneration.from_pretrained() here, which INSTANTIATES
# both ~0.9B-param models in fp32 (~8 GB peak) purely as a side effect of
# populating the cache — the OOM killer took the build out ("Killed", exit 137,
# ResourceExhausted) on a Docker VM with 5 GB RAM. snapshot_download fetches the
# identical files into the identical HF_HOME cache layout with near-zero RAM, so
# from_pretrained still resolves them offline at runtime. The text-encoder name
# is read out of the downloaded config.json rather than hardcoding flan-t5-large,
# matching the "load the description tokenizer dynamically" rule in CLAUDE.md.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python <<'PY'
import json, os
from huggingface_hub import snapshot_download

for repo in ("roshni-sorigin/mar-hin-betacraft-tts", "ai4bharat/indic-parler-tts"):
    path = snapshot_download(repo, ignore_patterns=["*.msgpack", "*.h5", "*.onnx"])
    enc = json.load(open(os.path.join(path, "config.json")))["text_encoder"]["_name_or_path"]
    snapshot_download(enc, allow_patterns=["*.json", "*.model", "spiece*", "tokenizer*"])
    print("cached", repo, "+ text-encoder tokenizer", enc)
print("Both models + tokenizers cached into image.")
PY

# Runtime never needs the network — everything above is already in HF_HOME.
ENV HF_HUB_OFFLINE=1

# RunPod Serverless QUEUE handler — NOT an HTTP server. runpod.serverless.start()
# polls RunPod's job queue and calls handler() per job, so there is no port to
# expose and no /ping health route for the platform to probe (those were Load
# Balancer requirements and no longer apply).
#
# -u keeps stdout unbuffered so per-sentence timing lines show up in the RunPod
# log viewer as generation happens, instead of only when the process flushes.
#
# To run the FastAPI dev server from this image instead (e.g. to fall back to a
# Load Balancer endpoint), override at run time:
#   docker run ... <image> uvicorn scripts.narrate.api_server:app --host 0.0.0.0 --port 8008
CMD ["python", "-u", "/app/scripts/narrate/rp_handler.py"]
