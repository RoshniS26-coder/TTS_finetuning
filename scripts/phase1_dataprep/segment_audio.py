#!/usr/bin/env python3
"""Slice raw narration in ./audio_data/ into 2–20s mono WAV chunks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence

INPUT_DIR = Path("./audio_data")
OUTPUT_AUDIO_DIR = Path("./marathi_dataset/audio")
TARGET_SR = 44100
MIN_CHUNK_MS = 2_000
MAX_CHUNK_MS = 20_000
MIN_SILENCE_LEN_MS = 500
SILENCE_THRESH_DBFS = -40
KEEP_SILENCE_MS = 250
EDGE_FADE_MS = 10

AUDIO_EXTENSIONS = {".wav", ".mp3", ".WAV", ".MP3"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_AUDIO_DIR)
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    parser.add_argument("--silence-thresh", type=int, default=SILENCE_THRESH_DBFS)
    return parser.parse_args()


def find_audio_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix in AUDIO_EXTENSIONS
    )
    if not files:
        log.error("No WAV/MP3 files found in %s", input_dir)
        sys.exit(1)
    return files


def load_mono(audio_path: Path, target_sr: int) -> AudioSegment:
    segment = AudioSegment.from_file(audio_path)
    return segment.set_channels(1).set_frame_rate(target_sr)


def split_fragments(segment: AudioSegment, silence_thresh: int = SILENCE_THRESH_DBFS) -> list[AudioSegment]:
    return split_on_silence(
        segment,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=silence_thresh,
        keep_silence=KEEP_SILENCE_MS,
    )


def merge_to_target_length(fragments: list[AudioSegment]) -> tuple[list[AudioSegment], int]:
    """Merge adjacent fragments into 2–20s chunks; return chunks and discard count."""
    chunks: list[AudioSegment] = []
    discarded = 0
    buffer = AudioSegment.empty()

    for fragment in fragments:
        if len(fragment) < MIN_CHUNK_MS:
            if buffer:
                buffer += fragment
            else:
                discarded += 1
            continue

        if not buffer:
            buffer = fragment
        else:
            buffer += fragment

        while len(buffer) >= MIN_CHUNK_MS:
            if len(buffer) <= MAX_CHUNK_MS:
                chunks.append(buffer)
                buffer = AudioSegment.empty()
                break

            chunks.append(buffer[:MAX_CHUNK_MS])
            buffer = buffer[MAX_CHUNK_MS:]

    if len(buffer) >= MIN_CHUNK_MS:
        if len(buffer) <= MAX_CHUNK_MS:
            chunks.append(buffer)
        else:
            while len(buffer) >= MIN_CHUNK_MS:
                chunks.append(buffer[:MAX_CHUNK_MS])
                buffer = buffer[MAX_CHUNK_MS:]
            if len(buffer) >= MIN_CHUNK_MS:
                chunks.append(buffer)
            elif len(buffer) > 0:
                discarded += 1
    elif len(buffer) > 0:
        discarded += 1

    return chunks, discarded


def hard_split_oversized(chunks: list[AudioSegment]) -> list[AudioSegment]:
    """Split any chunk still over MAX_CHUNK_MS at fixed intervals."""
    result: list[AudioSegment] = []
    for chunk in chunks:
        if len(chunk) <= MAX_CHUNK_MS:
            result.append(chunk)
            continue
        start = 0
        while start < len(chunk):
            end = min(start + MAX_CHUNK_MS, len(chunk))
            piece = chunk[start:end]
            if len(piece) >= MIN_CHUNK_MS:
                result.append(piece)
            start = end
    return result


def export_chunks(chunks: list[AudioSegment], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("chunk_*.wav"):
        existing.unlink()

    for idx, chunk in enumerate(chunks, start=1):
        # Short fades so every clip starts/ends at zero amplitude. Hard 20s
        # splits land mid-word; without this the model trains on abrupt edges
        # and reproduces clicks/static at clip boundaries during inference.
        chunk = chunk.fade_in(EDGE_FADE_MS).fade_out(EDGE_FADE_MS)
        out_path = output_dir / f"chunk_{idx:05d}.wav"
        chunk.export(out_path, format="wav")

    return len(chunks)


def main() -> None:
    args = parse_args()
    audio_files = find_audio_files(args.input_dir)

    all_chunks: list[AudioSegment] = []
    total_discarded = 0

    for audio_path in audio_files:
        log.info("Processing %s", audio_path.name)
        segment = load_mono(audio_path, args.target_sr)
        fragments = split_fragments(segment, args.silence_thresh)
        if not fragments:
            log.warning("No silence splits for %s — treating as single segment", audio_path.name)
            fragments = [segment]

        chunks, discarded = merge_to_target_length(fragments)
        chunks = hard_split_oversized(chunks)
        all_chunks.extend(chunks)
        total_discarded += discarded
        log.info("  → %d chunks from this file (%d fragments discarded)", len(chunks), discarded)

    count = export_chunks(all_chunks, args.output_dir)
    total_ms = sum(len(c) for c in all_chunks)
    total_sec = total_ms / 1000.0

    log.info("Done: %d chunks, %.1f s total (%.1f min)", count, total_sec, total_sec / 60)
    log.info("Fragments discarded (<2s): %d", total_discarded)
    log.info("Output: %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
