#!/usr/bin/env bash
# Narrate rabbit_tortoise_Story.txt with all 4 TTS models (sequential, CPU).
set -u
cd /Users/roshnisundrani/TTS/TTS_finetuning

PY=venv/bin/python
STORY=story_texts/rabbit_tortoise_Story.txt
PRETRAINED=/Users/roshnisundrani/.cache/huggingface/hub/models--ai4bharat--indic-parler-tts/snapshots/7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca
FINETUNED=/Users/roshnisundrani/.cache/huggingface/hub/models--roshni-sorigin--marathi-parler-tts/snapshots/2768848086b242fcee2c0ca5323a1223cf304cbc
# HF_TOKEN for the IndicF5 gated repo — read from .env, never hardcode secrets.
# .env keys have spaces/hyphens, so parse instead of `source`.
env_get(){ grep -E "^[[:space:]]*$1[[:space:]]*=" .env | head -1 | cut -d= -f2- | tr -d '"' | xargs; }
export HF_TOKEN="$(env_get 'inicF5-token')"
[ -n "$HF_TOKEN" ] || { echo "ERROR: inicF5-token not found in .env"; exit 1; }

stamp(){ date "+%H:%M:%S"; }

echo "[$(stamp)] ===> MMS-female"
$PY scripts/narrate/narrate_mms.py --text-file "$STORY" \
    --output model_outputs/mms_marathi/rabbit_tortoise_mms.wav \
  && echo "[$(stamp)] DONE mms" || echo "[$(stamp)] FAIL mms"

echo "[$(stamp)] ===> IndicF5"
$PY scripts/narrate/narrate_indicf5.py --text-file "$STORY" \
    --ref-audio ref_clip.wav \
    --ref-text "एकदा एक मोठा सिंह जंगलात झोपला होता." \
    --output model_outputs/indic_f5/rabbit_tortoise_indicf5.wav \
  && echo "[$(stamp)] DONE indicf5" || echo "[$(stamp)] FAIL indicf5"

echo "[$(stamp)] ===> Pretrained Indic Parler-TTS"
$PY scripts/narrate/narrate_story.py --text-file "$STORY" \
    --model-dir "$PRETRAINED" \
    --output model_outputs/pretrained_indic_parler/rabbit_tortoise_indic_parler.wav \
  && echo "[$(stamp)] DONE pretrained" || echo "[$(stamp)] FAIL pretrained"

echo "[$(stamp)] ===> Fine-tuned Parler-TTS"
$PY scripts/narrate/narrate_story.py --text-file "$STORY" \
    --model-dir "$FINETUNED" \
    --output model_outputs/finetuned_parler/rabbit_tortoise_finetuned.wav \
  && echo "[$(stamp)] DONE finetuned" || echo "[$(stamp)] FAIL finetuned"

echo "[$(stamp)] ===> ALL COMPLETE"
