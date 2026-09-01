#!/usr/bin/env python3
"""Build Hugging Face dataset from reviewed metadata.csv for Parler-TTS training."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict

METADATA_CSV = Path("./marathi_dataset_v2/metadata.csv")
AUDIO_BASE = Path("./marathi_dataset_v2")
HF_DATASET_DIR = Path("./marathi_parler_ds_v2")
TARGET_SR = 44100
EVAL_FRAC = 0.02   # held-out fraction for a real eval/overfitting signal (~80 clips on v2)
SEED = 42
DESCRIPTION = (
    "Sunita narrates in an expressive, warm storytelling tone at a moderate pace. "
    "Very clear audio with no background noise."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--audio-base", type=Path, default=AUDIO_BASE)
    parser.add_argument("--output-dir", type=Path, default=HF_DATASET_DIR)
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    parser.add_argument("--description", default=DESCRIPTION)
    parser.add_argument("--eval-frac", type=float, default=EVAL_FRAC,
                        help="held-out test fraction (0 = no eval split, train only)")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def parse_metadata(metadata_csv: Path, audio_base: Path, description: str) -> list[dict]:
    if not metadata_csv.exists():
        log.error("Metadata not found: %s — run transcribe.py first", metadata_csv)
        sys.exit(1)

    rows: list[dict] = []
    for line_no, line in enumerate(metadata_csv.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            log.warning("Line %d: malformed (no pipe delimiter), skipping", line_no)
            continue

        rel_path, _, text = line.partition("|")
        rel_path = rel_path.strip()
        text = text.strip()

        if not rel_path or not text:
            log.warning("Line %d: empty path or text, skipping", line_no)
            continue

        audio_path = audio_base / rel_path
        if not audio_path.exists():
            log.warning("Line %d: audio not found: %s, skipping", line_no, audio_path)
            continue

        rows.append({
            "audio": str(audio_path.resolve()),
            "text": text,
            "description": description,
        })

    if not rows:
        log.error("No valid rows in metadata.csv")
        sys.exit(1)

    return rows


def main() -> None:
    args = parse_args()
    rows = parse_metadata(args.metadata_csv, args.audio_base, args.description)

    log.info("Building dataset from %d rows", len(rows))
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("audio", Audio(sampling_rate=args.target_sr))

    # Hold out a small eval split so training has a real overfitting signal (watch eval/loss).
    if args.eval_frac and args.eval_frac > 0:
        split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
        dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})
    else:
        dataset_dict = DatasetDict({"train": ds})

    if args.output_dir.exists():
        log.info("Removing existing output: %s", args.output_dir)

    shutil.rmtree(args.output_dir, ignore_errors=True)

    dataset_dict.save_to_disk(str(args.output_dir))

    # Duration from source file headers (soundfile) — avoids decoding via the Audio feature,
    # which would pull in torch/torchcodec just for this log line.
    total_sec = sum(sf.info(r["audio"]).duration for r in rows)
    n_train = len(dataset_dict["train"])
    n_test = len(dataset_dict["test"]) if "test" in dataset_dict else 0
    log.info("Saved %s", args.output_dir.resolve())
    log.info("Rows: %d (train %d / test %d) | Total audio: %.1f s (%.1f min / %.2f hr)",
             len(ds), n_train, n_test, total_sec, total_sec / 60, total_sec / 3600)
    log.info("Columns: audio, text, description")
    log.info("Next: zip %s/ and upload to RunPod /workspace/", args.output_dir.name)


if __name__ == "__main__":
    main()
