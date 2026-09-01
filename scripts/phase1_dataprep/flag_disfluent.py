#!/usr/bin/env python3
"""Flag likely self-correction / disfluent clips by audio-vs-text mismatch.

A self-correction ("the s.. the sun") makes the AUDIO longer than the transcript
implies -> abnormally LOW characters-per-second. We score every clip and surface
the worst outliers so you review ~dozens, not thousands.

  python scripts/phase1_dataprep/flag_disfluent.py --dataset marathi_dataset_v2 --metadata metadata.csv

Outputs a sorted CSV of suspects (lowest chars/sec first) + a summary. Review the
top suspects by ear; delete the truly disfluent rows from metadata before building.
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="dataset root (holds audio/ + metadata)")
    ap.add_argument("--metadata", default="metadata.csv", help="metadata filename inside dataset")
    ap.add_argument("--flag-frac", type=float, default=0.6, help="flag clips below frac*median chars/sec")
    args = ap.parse_args()

    meta = args.dataset / args.metadata
    rows = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        rel, _, text = line.partition("|")
        rel, text = rel.strip(), text.strip()
        ap_ = args.dataset / rel
        if not ap_.exists() or not text:
            continue
        dur = sf.info(str(ap_)).duration
        chars = len(text.replace(" ", ""))          # devanagari char count, spaces excluded
        cps = chars / dur if dur > 0 else 0
        rows.append((cps, dur, chars, rel, text))

    if not rows:
        raise SystemExit(f"no valid rows in {meta}")
    cps_vals = [r[0] for r in rows]
    median = st.median(cps_vals)
    thresh = median * args.flag_frac
    flagged = sorted([r for r in rows if r[0] < thresh])

    out = args.dataset / "disfluent_suspects.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("chars_per_sec,duration_s,chars,path,text\n")
        for cps, dur, chars, rel, text in flagged:
            f.write(f"{cps:.2f},{dur:.2f},{chars},{rel},{text}\n")

    print(f"clips: {len(rows)} | median chars/sec: {median:.1f} | flag threshold (<{args.flag_frac}x): {thresh:.1f}")
    print(f"FLAGGED suspects (low chars/sec = audio longer than text -> likely disfluent): {len(flagged)} "
          f"({100*len(flagged)/len(rows):.1f}%)")
    print(f"-> {out}")
    print("worst 12 (review these by ear first):")
    for cps, dur, chars, rel, text in flagged[:12]:
        print(f"  {cps:4.1f} cps | {dur:4.1f}s | {rel} | {text[:48]}")


if __name__ == "__main__":
    main()
