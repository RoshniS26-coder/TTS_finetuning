#!/usr/bin/env python3
"""Transcribe chunked audio with faster-whisper (CPU) and write metadata.csv."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

AUDIO_DIR = Path("./marathi_dataset/audio")
METADATA_CSV = Path("./marathi_dataset/metadata.csv")
MODEL_SIZE = "large-v3"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANGUAGE = "mr"
PROGRESS_EVERY = 5
DEFAULT_INITIAL_PROMPT = "उंदीर मामा, माकड, मांजर, वाघ, सिंह, हत्ती, ससा, हरीण, खीर, साखर, दूध, शेवया, जंगल, कावळा, चिमणी, बुडबुड, घागर, राजा, भिकारी."

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--model-size", default=MODEL_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--compute-type", default=COMPUTE_TYPE)
    parser.add_argument("--force", action="store_true", help="Re-transcribe all files")
    parser.add_argument("--initial-prompt", default=DEFAULT_INITIAL_PROMPT)
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


def transcribe_file(model: WhisperModel, audio_path: Path, initial_prompt: str = "") -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        task="transcribe",
        vad_filter=True,
        initial_prompt=initial_prompt or None,
    )
    return "".join(seg.text for seg in segments).strip()


def write_metadata(metadata_csv: Path, rows: dict[str, str], audio_dir: Path) -> None:
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(audio_dir.glob("chunk_*.wav"))
    lines = []
    for wav in wav_files:
        rel = relative_audio_path(wav, audio_dir)
        text = rows.get(rel, "")
        lines.append(f"{rel}|{text}")

    metadata_csv.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.audio_dir.is_dir():
        log.error("Audio directory does not exist: %s — run segment_audio.py first", args.audio_dir)
        sys.exit(1)

    wav_files = sorted(args.audio_dir.glob("chunk_*.wav"))
    if not wav_files:
        log.error("No chunk_*.wav files in %s", args.audio_dir)
        sys.exit(1)

    rows = {} if args.force else load_existing(args.metadata_csv)
    pending = [
        wav for wav in wav_files
        if relative_audio_path(wav, args.audio_dir) not in rows or not rows[relative_audio_path(wav, args.audio_dir)]
    ]

    log.info("Found %d audio files (%d to transcribe)", len(wav_files), len(pending))
    if not pending:
        log.info("All files already transcribed. Use --force to redo.")
        write_metadata(args.metadata_csv, rows, args.audio_dir)
        return

    log.info("Loading Whisper model %s (%s, %s)...", args.model_size, args.device, args.compute_type)
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)

    start = time.time()
    for i, wav in enumerate(pending, start=1):
        rel = relative_audio_path(wav, args.audio_dir)
        log.info("[%d/%d] %s", i, len(pending), wav.name)

        try:
            text = transcribe_file(model, wav, args.initial_prompt)
        except Exception:
            log.exception("Failed to transcribe %s", wav.name)
            sys.exit(1)

        if not text:
            log.warning("Empty transcript for %s", wav.name)

        rows[rel] = text
        write_metadata(args.metadata_csv, rows, args.audio_dir)

        if i % PROGRESS_EVERY == 0 or i == len(pending):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(pending) - i) / rate if rate > 0 else 0
            log.info("Progress: %d/%d (%.0fs elapsed, ~%.0fs remaining)", i, len(pending), elapsed, remaining)

    log.info("Wrote %s (%d rows)", args.metadata_csv.resolve(), len(wav_files))
    log.info("IMPORTANT: Review and correct metadata.csv before running prepare_dataset.py")


if __name__ == "__main__":
    main()
