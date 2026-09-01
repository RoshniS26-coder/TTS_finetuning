#!/usr/bin/env bash
# A/B the Betacraft TTS Queue endpoint: same text, different settings, one WAV
# each, so the variants can be compared by ear.
#
# Usage:
#   export RUNPOD_ENDPOINT_ID=<your endpoint id>
#   export RUNPOD_API_KEY=<your runpod api key>
#   ./scripts/narrate/ab_test.sh                    # uses the sample text below
#   ./scripts/narrate/ab_test.sh "तुमचा मजकूर..."   # or your own story turn
#   ./scripts/narrate/ab_test.sh "..." hi           # language (default mr)
#
# Writes ab_out/<variant>.wav and prints a per-unit timing table for each.
set -euo pipefail

TEXT="${1:-एक आटपाट नगर होतं. तिथे रामू नावाचा एक छोटा मुलगा राहत होता. तो रोज सकाळी जंगलात फिरायला जायचा. एके दिवशी त्याला एक जखमी पक्षी दिसला. रामूने त्याला घरी आणून त्याची काळजी घेतली.}"
LANG_CODE="${2:-mr}"

: "${RUNPOD_ENDPOINT_ID:?set RUNPOD_ENDPOINT_ID to your Queue endpoint id}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY to your RunPod API key}"

URL="https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}"
OUT_DIR="ab_out"
mkdir -p "$OUT_DIR"

# variant_name : JSON object of overrides merged into the request input
VARIANTS=(
  'baseline:{}'
  'no_packing:{"pack_words":0}'
  'old_sampling:{"repetition_penalty":1.2,"no_repeat_ngram_size":3}'
  'no_loudness_match:{"match_loudness_db":0}'
  'cooler_temp:{"temperature":0.55}'
)

run_variant() {
  local name="$1" overrides="$2"
  local payload response status job_id started elapsed

  payload=$(jq -nc --arg t "$TEXT" --arg l "$LANG_CODE" --argjson o "$overrides" \
    '{input: ({text:$t, language:$l} + $o)}')

  started=$(date +%s)
  response=$(curl -sS -X POST "${URL}/runsync" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$payload")

  # /runsync returns inline when the job finishes fast, but hands back a job id
  # with IN_QUEUE/IN_PROGRESS if it exceeds RunPod's sync window — which is
  # exactly what a cold-starting worker does. Poll in that case.
  status=$(jq -r '.status // "UNKNOWN"' <<<"$response")
  job_id=$(jq -r '.id // empty' <<<"$response")
  while [[ "$status" == "IN_QUEUE" || "$status" == "IN_PROGRESS" ]]; do
    printf '  ... %s (worker may be cold-starting)\n' "$status"
    sleep 3
    response=$(curl -sS "${URL}/status/${job_id}" -H "Authorization: Bearer ${RUNPOD_API_KEY}")
    status=$(jq -r '.status // "UNKNOWN"' <<<"$response")
  done
  elapsed=$(( $(date +%s) - started ))

  if [[ "$status" != "COMPLETED" ]]; then
    printf '  FAILED (%s): %s\n\n' "$status" "$(jq -c '.error // .output.error // .' <<<"$response" | head -c 300)"
    return
  fi

  jq -r '.output.audio_b64' <<<"$response" | base64 -D > "${OUT_DIR}/${name}.wav"

  printf '  %ss wall | %s\n' "$elapsed" "$(jq -r '"\(.output.duration_sec)s audio, \(.output.sentences|length) unit(s)"' <<<"$response")"
  jq -r '.output.sentences[] |
    "    unit \(.index): \(.words)w -> \(.audio_sec)s in \(.gen_sec)s (rtf \(.rtf), \(.tokens)/\(.max_tokens) tok, attempts \(.attempts))\(if .hit_token_cap then "  <-- HIT TOKEN CAP" else "" end)"' \
    <<<"$response"
  printf '  saved %s/%s.wav\n\n' "$OUT_DIR" "$name"
}

printf '\nText: %s\nLanguage: %s\nEndpoint: %s\n\n' "$TEXT" "$LANG_CODE" "$URL"

# Warm the workers first so the first real variant isn't penalised by a cold
# start — otherwise "baseline" looks slow for reasons unrelated to its settings.
printf 'Warming workers...\n'
curl -sS -X POST "${URL}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
  -d '{"input":{"warmup":true}}' | jq -c '.status, .output' | tr '\n' ' '
printf '\n\n'

for entry in "${VARIANTS[@]}"; do
  name="${entry%%:*}"
  overrides="${entry#*:}"
  printf '=== %s  %s\n' "$name" "$overrides"
  run_variant "$name" "$overrides"
done

printf 'Done. Compare by ear:\n'
for entry in "${VARIANTS[@]}"; do printf '  afplay %s/%s.wav\n' "$OUT_DIR" "${entry%%:*}"; done
