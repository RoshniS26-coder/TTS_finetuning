#!/usr/bin/env bash
# One-shot RunPod bootstrap for Marathi Parler-TTS fine-tuning.
set -euo pipefail

WORKSPACE="/workspace"
PARLER_DIR="${WORKSPACE}/parler-tts"

echo "==> RunPod setup for Parler-TTS fine-tuning"
echo "    Persistent volume: ${WORKSPACE}"

cd "${WORKSPACE}"

if [ ! -d "${PARLER_DIR}" ]; then
    echo "==> Cloning parler-tts..."
    git clone https://github.com/huggingface/parler-tts.git
else
    echo "==> parler-tts already exists, skipping clone"
fi

echo "==> Installing parler-tts with training extras..."
cd "${PARLER_DIR}"
pip install -e ".[train]"

echo "==> Installing runtime deps (pinned — do NOT blindly --upgrade)..."
# NOTE: never `--upgrade transformers`. parler-tts 0.2.2 requires transformers==4.46.1
# exactly; upgrading pulls 5.x and breaks training on import.
# NOTE: pin datasets <4.0. datasets 4.x decodes audio via torchcodec, which needs a
# CUDA runtime lib (libnvrtc.so.13) that doesn't match the pod's torch -> load crash.
# datasets 3.x decodes with soundfile (no torchcodec, no CUDA lib needed).
pip install --upgrade accelerate soundfile wandb
pip install "transformers==4.46.1" "tokenizers==0.20.3" "datasets==3.6.0"
# hf_transfer: RunPod templates set HF_HUB_ENABLE_HF_TRANSFER=1 (fast Hub downloads);
# without this package the base-model download crashes at startup.
pip install hf_transfer

echo "==> Patching parler-tts DAC encode (drop unsupported 'bandwidth' kwarg)..."
# parler-tts main passes bandwidth= to DacModel.encode(), which transformers 4.46.1
# rejects (TypeError). Disable that branch so it uses n_quantizers instead. Idempotent.
sed -i 's/if bandwidth is not None:/if False:  # patched: DAC.encode rejects bandwidth/' \
    "${PARLER_DIR}/training/run_parler_tts_training.py" || true
grep -q "patched: DAC.encode" "${PARLER_DIR}/training/run_parler_tts_training.py" \
    && echo "    DAC patch applied" || echo "    WARN: DAC patch target not found (parler-tts changed?)"

echo "==> Installing ffmpeg..."
apt-get update -qq && apt-get install -y -qq ffmpeg

echo ""
echo "==> Setup complete. Before training, run:"
echo "    huggingface-cli login"
echo "    wandb login"
echo ""
echo "    Also accept gated access at:"
echo "    https://huggingface.co/ai4bharat/indic-parler-tts"
echo ""
echo "==> Then train with:"
echo "    cd ${PARLER_DIR}"
echo "    accelerate launch ./training/run_parler_tts_training.py /workspace/marathi_config.json"
echo ""
echo "==> After training, test with:"
echo "    python /workspace/test_inference.py"
