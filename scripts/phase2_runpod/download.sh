#!/usr/bin/env bash
# Zip the trained model on the pod, then download it (+ test.wav) to your laptop.
# RUN THIS ON YOUR LAPTOP (not inside an ssh session).
#
# Usage:
#   ./download.sh <POD_IP> <POD_PORT> [SSH_KEY]
#
# Example:
#   ./download.sh 213.45.67.89 41123
#
# Get <POD_IP> and <POD_PORT> from the RunPod console -> your pod -> Connect
# -> "SSH over exposed TCP" (the option with a real IP and -p port).
#
# After this succeeds and you've verified the files open, you can TERMINATE the pod.

set -euo pipefail

POD_IP="${1:-}"
POD_PORT="${2:-}"
SSH_KEY="${3:-$HOME/.ssh/id_ed25519}"

if [[ -z "$POD_IP" || -z "$POD_PORT" ]]; then
    echo "Usage: ./download.sh <POD_IP> <POD_PORT> [SSH_KEY]" >&2
    echo "Example: ./download.sh 213.45.67.89 41123" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # repo root; download artifacts land here
cd "$ROOT_DIR"

SSH_TARGET="root@${POD_IP}"

echo "==> Zipping trained model on the pod..."
ssh -p "$POD_PORT" -i "$SSH_KEY" "$SSH_TARGET" \
    'cd /workspace && [ -d marathi_tts_output ] && zip -r -q marathi_tts_model_v2.zip marathi_tts_output/ && echo "    zipped: $(du -h marathi_tts_model_v2.zip | cut -f1)" || { echo "ERROR: /workspace/marathi_tts_output not found on pod" >&2; exit 1; }'

echo "==> Downloading model zip to ${ROOT_DIR}/ ..."
scp -P "$POD_PORT" -i "$SSH_KEY" "${SSH_TARGET}:/workspace/marathi_tts_model_v2.zip" ./

echo "==> Downloading test.wav (if present)..."
scp -P "$POD_PORT" -i "$SSH_KEY" "${SSH_TARGET}:/workspace/test.wav" ./test_v2.wav || \
    echo "    (no test.wav found - run test_inference.py on the pod first if you want it)"

echo ""
echo "==> Done. Downloaded to ${ROOT_DIR}:"
ls -lh marathi_tts_model_v2.zip test_v2.wav 2>/dev/null || true
echo ""
echo "==> VERIFY the zip opens, THEN Terminate the pod in the RunPod console to stop billing."
