#!/usr/bin/env bash
# Upload prepared dataset + training files to the RunPod pod's /workspace volume.
# RUN THIS ON YOUR LAPTOP (not inside an ssh session).
#
# Usage:
#   ./upload.sh <POD_IP> <POD_PORT> [SSH_KEY]
#
# Example:
#   ./upload.sh 213.45.67.89 41123
#   ./upload.sh 213.45.67.89 41123 ~/.ssh/id_ed25519
#
# Get <POD_IP> and <POD_PORT> from the RunPod console -> your pod -> Connect
# -> "SSH over exposed TCP" (the option with a real IP and -p port).

set -euo pipefail

POD_IP="${1:-}"
POD_PORT="${2:-}"
SSH_KEY="${3:-$HOME/.ssh/id_ed25519}"

if [[ -z "$POD_IP" || -z "$POD_PORT" ]]; then
    echo "Usage: ./upload.sh <POD_IP> <POD_PORT> [SSH_KEY]" >&2
    echo "Example: ./upload.sh 213.45.67.89 41123" >&2
    exit 1
fi

# Resolve paths relative to this script so it works from any directory.
# Script lives in scripts/phase2_runpod/; the dataset dir + config stay at repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# The built HF dataset is a DIRECTORY (multi-speaker v3, ~5 GB). We upload it with
# `scp -r` instead of zipping — zipping 5 GB locally would need ~5 GB extra disk.
# Override with DATASET_DIR=... if the dataset name differs.
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/marathi_hindi_parler_ds_v3}"

FILES=(
    "$ROOT_DIR/marathi_config.json"
    "$SCRIPT_DIR/runpod_setup.sh"
    "$SCRIPT_DIR/test_inference.py"
    "$SCRIPT_DIR/push_to_hf.py"
)

echo "==> Checking files exist locally..."
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: dataset dir '$DATASET_DIR' not found (build it first, or set DATASET_DIR=...)" >&2
    exit 1
fi
printf "    %-34s %s\n" "$DATASET_DIR/" "$(du -sh "$DATASET_DIR" | cut -f1)"
for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing '$f'" >&2
        exit 1
    fi
    printf "    %-34s %s\n" "$f" "$(du -h "$f" | cut -f1)"
done

echo "==> Uploading to root@${POD_IP}:/workspace/ (port ${POD_PORT})"
# -r for the dataset directory; loose config/scripts go in the same scp call.
scp -r -P "$POD_PORT" -i "$SSH_KEY" "$DATASET_DIR" "${FILES[@]}" "root@${POD_IP}:/workspace/"

echo ""
echo "==> Upload complete (no unzip needed — dataset is already a directory). SSH in and run:"
echo "    ssh root@${POD_IP} -p ${POD_PORT} -i ${SSH_KEY}"
echo "    cd /workspace && bash runpod_setup.sh"
