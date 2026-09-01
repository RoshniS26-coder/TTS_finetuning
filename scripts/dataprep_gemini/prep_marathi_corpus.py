#!/usr/bin/env python3
"""Build a clean, merged Marathi text corpus for TTS synthesis (Sulafat -> v7 dataset).

Sources (balanced): (a) Gemini stories in generate_stories_gemini/marathi/ (all, per-file),
(b) a subset of v2 (Usha) transcripts, (c) a subset of panchtantra (Maya) transcripts.
Filters out code-mix (>=2 English words), converts digits to Marathi words, and merges
sentences into 2-20s units (min ~2s). Output = one merged unit per line, TSV: idx, source, text.

  venv/bin/python scripts/dataprep_gemini/prep_marathi_corpus.py \
      --v2-hours 0.8 --panch-hours 1.1 --out tts_corpus/marathi_units.tsv
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WPS = 2.3  # Marathi words/sec at storytelling pace (for duration estimate)
DIGIT_MR = {"0":"शून्य","1":"एक","2":"दोन","3":"तीन","4":"चार","5":"पाच","6":"सहा","7":"सात","8":"आठ","9":"नऊ"}
DIGIT_HI = {"0":"शून्य","1":"एक","2":"दो","3":"तीन","4":"चार","5":"पाँच","6":"छह","7":"सात","8":"आठ","9":"नौ"}
for _d, _a in zip("०१२३४५६७८९", "0123456789"):  # Devanagari digits map same as ASCII
    DIGIT_MR[_d] = DIGIT_MR[_a]; DIGIT_HI[_d] = DIGIT_HI[_a]

def norm_digits(t: str, dmap) -> str:
    return re.sub(r"(?<!\d)[0-9०-९](?!\d)", lambda m: dmap.get(m.group(), m.group()), t)

def eng_words(t: str) -> int:
    return len(re.findall(r"[A-Za-z]+", t))

def split_sents(t: str):
    parts = re.split(r"(?<=[।.!?])\s+", re.sub(r"\s+", " ", t.strip()))
    return [s.strip() for s in parts if s.strip() and "�" not in s]

def pack(sents, target=24, hard=34, min_w=5):
    units, cur = [], []
    wc = lambda x: len(x.split())
    for s in sents:
        if wc(s) > hard:  # long single sentence -> split on commas
            if cur: units.append(" ".join(cur)); cur=[]
            chunk=[]
            for piece in re.split(r"(?<=[,;])\s+", s):
                if sum(wc(p) for p in chunk)+wc(piece) > hard and chunk:
                    units.append(" ".join(chunk)); chunk=[]
                chunk.append(piece)
            if chunk: units.append(" ".join(chunk))
            continue
        if not cur or sum(wc(x) for x in cur)+wc(s) <= target:
            cur.append(s)
        else:
            units.append(" ".join(cur)); cur=[s]
    if cur: units.append(" ".join(cur))
    # forward-merge any sub-min unit into the next (no clips < ~2s)
    out=[]
    for u in units:
        if out and len(out[-1].split()) < min_w:
            out[-1] = out[-1] + " " + u
        else:
            out.append(u)
    if len(out) >= 2 and len(out[-1].split()) < min_w:
        out[-2] = out[-2] + " " + out.pop()
    return out

def csv_lines(path: Path):
    return [l.split("|",1)[1].strip() for l in path.read_text(encoding="utf-8").splitlines()
            if "|" in l and l.split("|",1)[1].strip()]

def take_hours(lines, hours, max_english, dmap):
    """Clean-filter + accumulate lines to ~hours, then merge. Each CSV line is already a
    sentence-chunk, so pack the LINES directly — join+resplit breaks when transcripts lack
    end punctuation (e.g. Hindi panchtantra) — and ensure each line ends with a danda."""
    budget = hours * 3600 * WPS
    picked, used = [], 0
    for t in lines:
        if eng_words(t) > max_english:
            continue
        t = norm_digits(t, dmap).strip()
        if t and t[-1] not in "।.!?":
            t += "।"
        picked.append(t); used += len(t.split())
        if used >= budget:
            break
    return pack(picked)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini-dir", type=Path, default=ROOT/"generate_stories_gemini"/"marathi")
    ap.add_argument("--v2-csv", type=Path, default=ROOT/"marathi_dataset_v2"/"metadata.csv")
    ap.add_argument("--panch-csv", type=Path, default=ROOT/"marathi_panchtantra_dataset"/"metadata.csv")
    ap.add_argument("--v2-hours", type=float, default=0.8)
    ap.add_argument("--panch-hours", type=float, default=1.1)
    ap.add_argument("--max-english", type=int, default=1)
    ap.add_argument("--lang", choices=["mr", "hi"], default="mr", help="digit->word language (mr/hi)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="override CSV sources as path:hours pairs, e.g. hindi_panchtantra_dataset/metadata.csv:2.6")
    ap.add_argument("--out", type=Path, default=ROOT/"tts_corpus"/"marathi_units.tsv")
    args = ap.parse_args()

    from collections import Counter
    dmap = DIGIT_HI if args.lang == "hi" else DIGIT_MR
    rows, counts = [], Counter()
    # (a) Gemini stories — per file (coherent), all of them
    for f in sorted(args.gemini_dir.glob("*.txt")):
        for u in pack(split_sents(norm_digits(f.read_text(encoding="utf-8"), dmap))):
            rows.append(("gemini", u)); counts["gemini"] += 1
    # (b) CSV sources (default = v2 + panchtantra; override with --sources path:hours ...)
    if args.sources:
        srcs = [(Path(s.rpartition(":")[0]), float(s.rpartition(":")[2])) for s in args.sources]
    else:
        srcs = [(args.v2_csv, args.v2_hours), (args.panch_csv, args.panch_hours)]
    for csv, hrs in srcs:
        label = csv.parent.name[:14]
        for u in take_hours(csv_lines(csv), hrs, args.max_english, dmap):
            rows.append((label, u)); counts[label] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("idx\tsource\ttext\n")
        for i, (src, txt) in enumerate(rows, 1):
            fh.write(f"{i}\t{src}\t{txt}\n")

    words = sum(len(t.split()) for _, t in rows)
    dur = words / WPS
    dists = [len(t.split()) for _, t in rows]
    print(f"Wrote {len(rows)} merged units -> {args.out}")
    print("  sources: " + " | ".join(f"{k} {n}" for k, n in counts.items()))
    print(f"  words {words} | est audio ~{dur/3600:.2f} h ({dur/60:.0f} min)")
    print(f"  unit length: min {min(dists)}w  avg {words/len(rows):.1f}w  max {max(dists)}w  (~{min(dists)/WPS:.1f}-{max(dists)/WPS:.1f}s)")
    print(f"  under 2s (<5w): {sum(1 for d in dists if d<5)} | over 20s (>46w): {sum(1 for d in dists if d>46)}")
    print("\n  sample units (one per source):")
    seen = set()
    for src, txt in rows:
        if src not in seen:
            seen.add(src); print(f"    [{src}] {txt[:80]}")

if __name__ == "__main__":
    main()
