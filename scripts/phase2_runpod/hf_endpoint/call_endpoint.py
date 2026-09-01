"""Test client for the betacraft-tts HF Inference Endpoint.

Usage:
  # ENDPOINT_URL = the "Endpoint URL" shown on the endpoint's page
  # token read from .env key `mar-hin-betacraft-tts` (or pass HF_TOKEN env var)
  python scripts/phase2_runpod/hf_endpoint/call_endpoint.py \
      --url https://xxxx.endpoints.huggingface.cloud \
      --text "एक आटपाट नगर होतं. तिथे एक राजा राहत होता." \
      --out test_mr.wav
"""
import argparse, base64, os, re, sys
import urllib.request, json

DEFAULT_DESCRIPTION = (
    "Sunita narrates a children's story in an expressive, warm, animated "
    "storytelling tone with emotional variation, at a moderate pace. "
    "Very clear audio with no background noise."
)


def token_from_env_file(key: str = "mar-hin-betacraft-tts") -> str | None:
    """Read a token from the repo .env (keys use spaces/hyphens -> can't `source`)."""
    for cand in (".env", os.path.join(os.path.dirname(__file__), "../../../.env")):
        if os.path.exists(cand):
            with open(cand) as fh:
                for line in fh:
                    m = re.match(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", line)
                    if m:
                        return m.group(1).strip().strip('"')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Endpoint URL (…endpoints.huggingface.cloud)")
    ap.add_argument("--text", required=True, help="Devanagari text to narrate")
    ap.add_argument("--description", default=DEFAULT_DESCRIPTION,
                    help="Caption; swap Sunita->Divya for Hindi. Keep wording identical.")
    ap.add_argument("--out", default="endpoint_out.wav")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or token_from_env_file()
    if not token:
        sys.exit("No token: set HF_TOKEN or add `mar-hin-betacraft-tts=` to .env")

    payload = json.dumps({
        "inputs": args.text,
        "parameters": {"description": args.description},
    }).encode("utf-8")

    req = urllib.request.Request(
        args.url, data=payload, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())

    if "audio_base64" not in data:
        sys.exit(f"Unexpected response: {data}")
    with open(args.out, "wb") as fh:
        fh.write(base64.b64decode(data["audio_base64"]))
    print(f"wrote {args.out}  (sampling_rate={data.get('sampling_rate')})")


if __name__ == "__main__":
    main()
