#!/usr/bin/env python3
"""Push the built parquet dataset to the Hugging Face Hub (PRIVATE dataset repo).

RUN THIS ON YOUR LAPTOP — the dataset already lives here; it does NOT need the pod
and is unaffected by terminating RunPod.

Prereqs:
  huggingface-cli login          # HF token with WRITE scope
Usage:
  venv/bin/python scripts/phase1_dataprep/push_dataset_to_hf.py
  # or override:
  venv/bin/python scripts/phase1_dataprep/push_dataset_to_hf.py \
      --dataset-dir mar_hin_betacraft_ds \
      --repo roshni-sorigin/mar-hin-betacraft-tts-dataset
"""
import argparse
from pathlib import Path
from datasets import load_dataset, Audio

TARGET_SR = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="mar_hin_betacraft_ds",
                    help="dir with train.parquet + test.parquet")
    ap.add_argument("--repo", default="roshni-sorigin/mar-hin-betacraft-tts-dataset",
                    help="target HF dataset repo id")
    ap.add_argument("--public", action="store_true",
                    help="make the dataset public (default: PRIVATE)")
    ap.add_argument("--sr", type=int, default=TARGET_SR)
    args = ap.parse_args()

    d = Path(args.dataset_dir)
    train_pq, test_pq = d / "train.parquet", d / "test.parquet"
    for p in (train_pq, test_pq):
        if not p.is_file():
            raise SystemExit(f"ERROR: missing {p}")

    print(f"==> Loading {train_pq} + {test_pq}")
    ds = load_dataset("parquet", data_files={
        "train": str(train_pq), "test": str(test_pq)})
    # Restore the Audio feature so the HF dataset viewer can play clips.
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sr))
    print(ds)

    private = not args.public
    print(f"==> Pushing to {args.repo}  (private={private})")
    ds.push_to_hub(args.repo, private=private)
    print(f"==> DONE. https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
