#!/usr/bin/env python3
"""Transcribe chunked audio with Sarvam Saaras (codemix) and write metadata.csv.

Mirrors transcribe.py's output contract (pipe-delimited `audio/chunk_xxxxx.wav|text`,
resumable) but calls Sarvam's STT API instead of faster-whisper. `codemix` mode keeps
Marathi in Devanagari and English in Latin script — the format Parler fine-tuning wants.

Reads SARVAM_API_KEY from the environment or a .env file (never hardcode it).

  python transcribe_saaras.py --audio-dir marathi_dataset_v2_test/audio \
      --metadata-csv marathi_dataset_v2_test/metadata.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API_URL = "https://api.sarvam.ai/speech-to-text"
AUDIO_DIR = Path("./marathi_dataset/audio")
METADATA_CSV = Path("./marathi_dataset/metadata.csv")
MODEL = "saaras:v3"
MODE = "codemix"
LANGUAGE = "mr-IN"
WORKERS = 4
MAX_RETRIES = 5
TIMEOUT_S = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


KEY_NAMES = ("SARVAM_API_KEY", "sarvam-api-key", "sarvam_api_key", "SARVAM_KEY")


def load_env_key(env_file: Path) -> str:
    """Return the Sarvam API key from the environment, falling back to a .env file.

    Accepts several name variants (the .env here uses `sarvam-api-key`).
    """
    for name in KEY_NAMES:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in KEY_NAMES:
                return value.strip().strip('"').strip("'")
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--mode", default=MODE, help="saaras:v3 mode: codemix/transcribe/translate/...")
    parser.add_argument("--language", default=LANGUAGE, help="BCP-47 code, or 'unknown' for auto")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="only transcribe first N pending (0 = all)")
    parser.add_argument("--force", action="store_true", help="re-transcribe everything")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args()


def load_existing(metadata_csv: Path) -> dict[str, str]:
    if not metadata_csv.exists():
        return {}
    existing: dict[str, str] = {}
    for line in metadata_csv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        rel_path, _, text = line.partition("|")
        existing[rel_path.strip()] = text.strip()
    return existing


def relative_audio_path(audio_path: Path, audio_dir: Path) -> str:
    return str(audio_path.relative_to(audio_dir.parent))


def write_metadata(metadata_csv: Path, rows: dict[str, str], audio_dir: Path) -> None:
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for wav in sorted(audio_dir.glob("chunk_*.wav")):
        rel = relative_audio_path(wav, audio_dir)
        lines.append(f"{rel}|{rows.get(rel, '')}")
    metadata_csv.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def transcribe_one(wav: Path, key: str, model: str, mode: str, language: str) -> str:
    """Call Sarvam STT for one clip; retry on 429/5xx and transient network errors."""
    headers = {"api-subscription-key": key}
    data = {"model": model, "language_code": language}
    if model.startswith("saaras"):
        data["mode"] = mode  # mode only applies to saaras:v3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with wav.open("rb") as fh:
                files = {"file": (wav.name, fh, "audio/wav")}
                resp = requests.post(API_URL, headers=headers, data=data, files=files, timeout=TIMEOUT_S)
            if resp.status_code == 200:
                return (resp.json().get("transcript") or "").strip()
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2 ** attempt, 30)
                log.warning("  %s -> HTTP %d, retry %d/%d in %ds", wav.name, resp.status_code, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            # Non-retryable (e.g. 400/401/403) — surface and stop retrying this file.
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            wait = min(2 ** attempt, 30)
            log.warning("  %s -> %s, retry %d/%d in %ds", wav.name, type(exc).__name__, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
    raise RuntimeError(f"failed after {MAX_RETRIES} retries")


def main() -> None:
    args = parse_args()

    key = load_env_key(args.env_file)
    if not key:
        log.error("SARVAM_API_KEY not found in env or %s", args.env_file)
        sys.exit(1)

    if not args.audio_dir.is_dir():
        log.error("Audio dir not found: %s — run segment_audio.py first", args.audio_dir)
        sys.exit(1)

    wav_files = sorted(args.audio_dir.glob("chunk_*.wav"))
    if not wav_files:
        log.error("No chunk_*.wav in %s", args.audio_dir)
        sys.exit(1)

    rows = {} if args.force else load_existing(args.metadata_csv)
    pending = [w for w in wav_files if not rows.get(relative_audio_path(w, args.audio_dir))]
    if args.limit:
        pending = pending[: args.limit]

    log.info("Found %d clips (%d to transcribe) | model=%s mode=%s lang=%s workers=%d",
             len(wav_files), len(pending), args.model, args.mode, args.language, args.workers)
    if not pending:
        log.info("Nothing to do (use --force to redo).")
        write_metadata(args.metadata_csv, rows, args.audio_dir)
        return

    lock = threading.Lock()
    done = 0
    failed = 0
    start = time.time()

    def work(wav: Path):
        nonlocal done, failed
        rel = relative_audio_path(wav, args.audio_dir)
        try:
            text = transcribe_one(wav, key, args.model, args.mode, args.language)
        except Exception as exc:
            with lock:
                failed += 1
            log.warning("  FAILED %s: %s", wav.name, exc)
            return
        with lock:
            rows[rel] = text
            done += 1
            n = done + failed
            if n % 20 == 0 or n == len(pending):
                write_metadata(args.metadata_csv, rows, args.audio_dir)
                elapsed = time.time() - start
                rate = n / elapsed if elapsed else 0
                eta = (len(pending) - n) / rate if rate else 0
                log.info("Progress: %d/%d (%d failed, %.0fs elapsed, ~%.0fs left)",
                         n, len(pending), failed, elapsed, eta)
            if not text:
                log.warning("  empty transcript: %s", wav.name)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, w) for w in pending]
        for _ in as_completed(futures):
            pass

    write_metadata(args.metadata_csv, rows, args.audio_dir)
    log.info("Wrote %s (%d transcribed, %d failed)", args.metadata_csv.resolve(), done, failed)
    if failed:
        log.info("Re-run to retry the %d failed clips (resumable).", failed)
    log.info("IMPORTANT: review metadata.csv (esp. Latin-script English) before prepare_dataset.py")


if __name__ == "__main__":
    main()
