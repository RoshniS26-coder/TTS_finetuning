#!/usr/bin/env python3
"""Convert a save_to_disk DatasetDict to Parquet files that load_dataset can read.

parler-tts loads training data with datasets.load_dataset(), which mis-infers the
format of a save_to_disk() directory (mixed .arrow + .json -> "Couldn't infer the
same data file format for all splits"). Re-serializing each split to a single
<split>.parquet in a clean directory fixes that: load_dataset(dir, split=...) then
reads parquet unambiguously and restores the embedded Audio feature.

Run ON THE POD (after the dataset dir has been uploaded):
    python /workspace/convert_to_parquet.py
Override paths via env:
    SRC=/workspace/marathi_hindi_parler_ds_v3  OUT=/workspace/ds_pq  python ...
"""
import os
from datasets import load_from_disk

SRC = os.environ.get("SRC", "/workspace/marathi_hindi_parler_ds_v3")
OUT = os.environ.get("OUT", "/workspace/ds_pq")

ds = load_from_disk(SRC)
os.makedirs(OUT, exist_ok=True)
for split in ds:
    path = os.path.join(OUT, f"{split}.parquet")
    ds[split].to_parquet(path)
    print(f"wrote {path}  ({ds[split].num_rows} rows)")
print("done:", {k: ds[k].num_rows for k in ds})
