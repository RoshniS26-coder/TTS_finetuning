
#!/usr/bin/env bash
# Open an SSH session into the RunPod pod.
# RUN THIS ON YOUR LAPTOP.
#
# Usage:
#   ./connect.sh <POD_IP> <POD_PORT> [SSH_KEY]
#
# Example:
#   ./connect.sh 213.45.67.89 41123
#   ./connect.sh 213.45.67.89 41123 ~/.ssh/id_ed25519
#
# Get <POD_IP> and <POD_PORT> from the RunPod console -> your pod -> Connect
# -> "SSH over exposed TCP" (the option with a real IP and -p port).

set -euo pipefail

POD_IP="${1:-}"
POD_PORT="${2:-}"
SSH_KEY="${3:-$HOME/.ssh/id_ed25519}"

if [[ -z "$POD_IP" || -z "$POD_PORT" ]]; then
    echo "Usage: ./connect.sh <POD_IP> <POD_PORT> [SSH_KEY]" >&2
    echo "Example: ./connect.sh 213.45.67.89 41123" >&2
    exit 1
fi

echo "==> Connecting to root@${POD_IP} (port ${POD_PORT}) ..."
exec ssh "root@${POD_IP}" -p "$POD_PORT" -i "$SSH_KEY"
