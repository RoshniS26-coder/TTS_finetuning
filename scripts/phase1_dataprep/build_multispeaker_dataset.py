#!/usr/bin/env python3
"""Build a combined multi-speaker Parler-TTS dataset from several metadata.csv files.

Each speaker gets its OWN name in the caption (the only thing that differs between
speakers) and is split into train/test individually, so the held-out eval covers
every voice. The per-speaker train parts are concatenated, the test parts are
concatenated, and the result is saved as one DatasetDict({"train","test"}).

Speaker mapping (edit SPEAKERS below to change names/data):
  Maya   <- marathi_panchtantra_dataset   (Marathi narrator, ~11 h)
  Usha   <- marathi_dataset_v2            (Marathi narrator, ~4.25 h)
  Sarita <- hindi_panchtantra_dataset     (Hindi narrator,  ~2.5 h)

Language is set by the transcript text, NOT the caption — so all captions share the
same storytelling style words and differ only by name. Run from repo root:

  python scripts/phase1_dataprep/build_multispeaker_dataset.py \
      --output-dir marathi_hindi_parler_ds_v3 --eval-frac 0.02
"""
from __future__ import annotations

import argparse, logging, random, re, shutil, sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET_SR = 44100

# Shared storytelling style; {name} and {pace} vary per speaker. Pace should HONESTLY
# match the audio — labelling fast narration as "moderate" muddies pace control AND
# weakens the same caption->acoustic mapping that drives pronunciation fidelity.
CAPTION = ("{name} narrates a children's story in an expressive, warm, animated "
           "storytelling tone with emotional variation, at a {pace} pace. "
           "Very clear audio with no background noise.")
DEFAULT_PACE = "moderate"

SPEAKERS = [
    # Maya = panchtantra; audio cleaned: denoised (0.5) + slowed 10% (atempo 0.90)
    # + 94 male-voice outro clips removed -> now MODERATE pace (uses DEFAULT_PACE).
    # "sample": random-subset to N clips at BUILD time (source metadata untouched) so the
    # large panchtantra set doesn't dominate the shared decoder. Seeded by --seed.
    {"name": "Maya",   "metadata": Path("marathi_panchtantra_dataset/metadata.csv"),
     "audio_base": Path("marathi_panchtantra_dataset"),
     "sample": {"n": 3000, "by": "random"}},
    {"name": "Usha",   "metadata": Path("marathi_dataset_v2/metadata.csv"),
     "audio_base": Path("marathi_dataset_v2")},
    {"name": "Sarita", "metadata": Path("hindi_panchtantra_dataset/metadata.csv"),
     "audio_base": Path("hindi_panchtantra_dataset")},
    # v6: Hindi panchtantra added as "Divya" (Hindi voice) alongside Sunita.
    # Build the v6 2-speaker set with: --only Sunita,Divya
    {"name": "Divya", "metadata": Path("hindi_panchtantra_dataset/metadata.csv"),
     "audio_base": Path("hindi_panchtantra_dataset")},
    # v1 (the loved timbre, ~16 min) added as "Sunita". In a multi-speaker run the
    # SHARED decoder learns prosody/pronunciation from all ~17.7 h, while "Sunita"
    # just anchors v1's timbre -> v1 voice with better narration. Note: metadata_v1.csv.
    # Sunita is the TARGET narration voice but the smallest set (232 clips ~3%). It's
    # also a built-in Indic-Parler Marathi speaker, so this ADAPTS a stable voice. To
    # keep the low-resource speaker from being under-anchored vs the 3000+3000 dominant
    # sets, oversample her TRAIN rows 2x (test split left as-is -> no eval leakage).
    {"name": "Sunita", "metadata": Path("marathi_dataset_v1/metadata_v1.csv"),
     "audio_base": Path("marathi_dataset_v1"), "train_repeat": 2},
    # v6 single-speaker run: v2 (same narrator as v1) ALSO mapped to "Sunita" so a
    # `--only Sunita` build merges v1+v2 into one Sunita voice (Maya/Usha/Sarita excluded).
    # v2 restored to 3376 (english-heavy added back, disfluent stay removed). No cap.
    {"name": "Sunita", "metadata": Path("marathi_dataset_v2/metadata.csv"),
     "audio_base": Path("marathi_dataset_v2")},
]

# v7 SYNTHETIC (Gemini-TTS / Sulafat) 2-speaker set — copyright-safe distillation.
# Sunita = Marathi, Divya = Hindi, both from tts_data/ (synthesized from the exact
# input text -> perfect transcript alignment). Captions are byte-identical to v6 via
# the shared CAPTION template + DEFAULT_PACE="moderate". Build with --v7.
# Proceeding as-is (no train_repeat): 1770 Mr / 1510 Hi is ~1.17:1 by clips, more
# balanced than v6, so no oversampling needed for a fair v6-vs-v7 comparison.
SPEAKERS_V7 = [
    {"name": "Sunita", "metadata": Path("tts_data/marathi/metadata.csv"),
     "audio_base": Path("tts_data/marathi")},
    {"name": "Divya",  "metadata": Path("tts_data/hindi/metadata.csv"),
     "audio_base": Path("tts_data/hindi")},
]


def parse_metadata(metadata_csv: Path, audio_base: Path, description: str) -> list[dict]:
    if not metadata_csv.exists():
        log.error("Metadata not found: %s — transcribe this speaker first", metadata_csv)
        sys.exit(1)
    rows, skipped = [], 0
    for line_no, line in enumerate(metadata_csv.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or "|" not in line:
            continue
        rel_path, _, text = line.partition("|")
        rel_path, text = rel_path.strip(), text.strip()
        if not rel_path or not text:
            skipped += 1
            continue
        audio_path = audio_base / rel_path
        if not audio_path.exists():
            skipped += 1
            continue
        rows.append({"audio": str(audio_path.resolve()), "text": text, "description": description})
    if not rows:
        log.error("No valid rows in %s", metadata_csv)
        sys.exit(1)
    if skipped:
        log.info("  %s: skipped %d empty/missing rows", metadata_csv, skipped)
    return rows


def embed_audio_bytes(ds, target_sr: int):
    """Read each audio file's BYTES into the dataset so the parquet is self-contained.

    Built from local paths, the Audio cells are {"bytes": None, "path": "/local/..."};
    to_parquet writes those paths verbatim (32 KB file) and training FAILS on the pod
    because the local paths don't exist there. Cast to decode=False to expose the raw
    cell, read the file bytes in, then cast back so the feature is a normal Audio column.
    """
    ds = ds.cast_column("audio", Audio(decode=False))

    def _read(ex):
        path = ex["audio"]["path"]
        with open(path, "rb") as fh:
            return {"audio": {"bytes": fh.read(), "path": Path(path).name}}

    ds = ds.map(_read, desc="embedding audio bytes")
    return ds.cast_column("audio", Audio(sampling_rate=target_sr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("marathi_hindi_parler_ds_v3"))
    ap.add_argument("--eval-frac", type=float, default=0.02, help="per-speaker held-out test fraction")
    ap.add_argument("--target-sr", type=int, default=TARGET_SR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None,
                    help="comma-separated speaker names to build (default: all). "
                         "e.g. --only Sunita builds just the v1 add-on parquet.")
    ap.add_argument("--v7", action="store_true",
                    help="use the v7 synthetic 2-speaker set (Sunita=Marathi, Divya=Hindi from tts_data/).")
    args = ap.parse_args()

    speakers = SPEAKERS_V7 if args.v7 else SPEAKERS
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        speakers = [s for s in SPEAKERS if s["name"] in want]
        if not speakers:
            sys.exit(f"--only matched none; known: {[s['name'] for s in SPEAKERS]}")

    train_parts, test_parts, summary = [], [], []
    for sp in speakers:
        desc = CAPTION.format(name=sp["name"], pace=sp.get("pace", DEFAULT_PACE))
        rows = parse_metadata(sp["metadata"], sp["audio_base"], desc)
        spec = sp.get("sample")
        if spec and len(rows) > spec["n"]:
            orig_n, n = len(rows), spec["n"]
            if spec.get("by") == "english":
                # keep the MOST english-heavy n (count of latin-script word tokens)
                rows = sorted(rows, key=lambda r: len(re.findall(r"[A-Za-z]+", r["text"])), reverse=True)[:n]
            else:  # random (seeded -> reproducible)
                rows = random.Random(args.seed).sample(rows, n)
            log.info("  %s: subsampled %d -> %d (by=%s, seed=%d)", sp["name"], orig_n, n, spec.get("by", "random"), args.seed)
        ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=args.target_sr))
        rep = sp.get("train_repeat", 1)
        if args.eval_frac and args.eval_frac > 0 and len(ds) > 50:
            split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
            tr = split["train"]
            if rep > 1:  # oversample TRAIN only (test untouched -> no eval leakage)
                tr = concatenate_datasets([tr] * rep)
                log.info("  %s: oversampled train x%d (%d -> %d rows)", sp["name"], rep, len(split["train"]), len(tr))
            train_parts.append(tr); test_parts.append(split["test"])
            n_tr, n_te = len(tr), len(split["test"])
        else:
            tr = concatenate_datasets([ds] * rep) if rep > 1 else ds
            train_parts.append(tr); n_tr, n_te = len(tr), 0
        secs = sum(sf.info(r["audio"]).duration for r in rows)
        summary.append((sp["name"], len(rows), n_tr, n_te, secs / 3600))
        log.info("Speaker %-7s: %d clips (%.2f h) | caption: %s", sp["name"], len(rows), secs / 3600, desc)

    train = concatenate_datasets(train_parts).shuffle(seed=args.seed)
    test = concatenate_datasets(test_parts) if test_parts else None
    dataset_dict = DatasetDict({"train": train, **({"test": test} if test is not None else {})})

    # Write PARQUET (not save_to_disk): parler-tts loads with datasets.load_dataset(),
    # which can't read a save_to_disk dir (mixed .arrow + .json -> "Couldn't infer format").
    # <split>.parquet in a clean dir IS load_dataset-ready and embeds the Audio bytes.
    # -> no separate convert_to_parquet step on the pod.
    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, ds in dataset_dict.items():
        ds = embed_audio_bytes(ds, args.target_sr)
        ds.to_parquet(str(args.output_dir / f"{split}.parquet"))

    log.info("=" * 70)
    log.info("Saved %s (parquet, load_dataset-ready)", args.output_dir.resolve())
    total_h = sum(s[4] for s in summary)
    for name, n, n_tr, n_te, h in summary:
        log.info("  %-7s %5d clips  %5.2f h  (train %d / test %d)", name, n, h, n_tr, n_te)
    log.info("  TOTAL   %5d clips  %5.2f h  | train %d / test %d",
             sum(s[1] for s in summary), total_h, len(train), len(test) if test is not None else 0)
    log.info("Columns: audio, text, description (per-speaker name). Next: upload %s/ to RunPod.", args.output_dir.name)


if __name__ == "__main__":
    main()
