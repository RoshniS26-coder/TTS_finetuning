#!/usr/bin/env python3
"""Rewrite Maya's pace in an existing parquet dataset (on the pod) WITHOUT
re-uploading audio. Maya (panchtantra) narrates fast, but ds_pq captioned her
"at a moderate pace" -> mislabel. This flips ONLY Maya's description text from
"moderate" to "fairly fast"; the embedded audio bytes are copied through untouched.

  python fix_maya_caption.py /workspace/ds_pq /workspace/ds_pq_fixed

Then point the training config at the *_fixed dir (e.g. ds_pq_fixed+v1_pq).
"""
import os
import sys
from datasets import load_dataset

SRC = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ds_pq"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/ds_pq_fixed"
OLD, NEW = "at a moderate pace", "at a fairly fast pace"

ds = load_dataset(SRC)  # reads train.parquet / test.parquet (audio bytes embedded)


def fix(ex):
    d = ex["description"]
    if d.startswith("Maya narrates") and OLD in d:
        ex["description"] = d.replace(OLD, NEW)
    return ex


os.makedirs(OUT, exist_ok=True)
for split in ds:
    fixed = ds[split].map(fix, desc=f"fixing Maya pace ({split})")
    n_maya_fixed = sum(
        1 for d in fixed["description"]
        if d.startswith("Maya narrates") and NEW in d
    )
    fixed.to_parquet(f"{OUT}/{split}.parquet")
    print(f"{split}: {len(fixed)} rows | Maya rows now 'fairly fast' = {n_maya_fixed}")

print("wrote", OUT, "-> set config dataset to", f"{OUT}+/workspace/v1_pq")
