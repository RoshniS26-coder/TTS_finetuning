#!/usr/bin/env python3
"""Audio-in-the-loop review to turn numeric tokens in metadata.csv into spoken words.

Sarvam Saaras (codemix) writes spoken numbers back as ASCII digits (`1`, `7:00`, `2-3`,
`1st`). For Parler-TTS the transcript must spell numbers EXACTLY as the narrator says them
(एक / सात / one / seven — the audio mixes languages), so every flagged clip is reviewed by
ear. See CLAUDE.md Step 3: "Numbers -> spoken form, never the digit; transcript MUST match
audio exactly."

Workflow (fully resumable via a side worklist):

  python review_numbers.py extract   # build numbers_review.csv from metadata.csv
  # listen to each clip, then in the CSV's `reviewed` column:
  python review_numbers.py apply     # merge corrections back into metadata.csv (+ .bak)

`reviewed` semantics (default = yes, so you only mark the exceptions):
  yes  -> accept the Marathi suggestion in corrected_text (auto: cardinals + times).
  no   -> the audio is ENGLISH; apply writes the English-word form (seven, fifteen, first).
  else -> skipped, left unchanged.
Hand-edit corrected_text for anything irregular and keep that row 'yes'.
(`review` is an optional interactive player; CSV editing in a spreadsheet is the main path.)

The metadata.csv contract (pipe-delimited `audio/chunk_xxxxx.wav|text`, UTF-8, headerless,
trailing newline) mirrors transcribe_saaras.py::write_metadata so prepare_dataset.py is
unaffected.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path("marathi_dataset_v2")
METADATA_CSV = DATASET_DIR / "metadata.csv"
WORKLIST_CSV = DATASET_DIR / "numbers_review.csv"

FIELDS = ["audio_path", "detected", "current_text", "suggested_text", "corrected_text", "reviewed"]

# Marathi cardinals used only to pre-fill a default the reviewer can accept with Enter.
# The human still hears the clip and overrides for English / time / ordinal cases.
MARATHI_CARDINAL = {
    0: "शून्य", 1: "एक", 2: "दोन", 3: "तीन", 4: "चार", 5: "पाच", 6: "सहा", 7: "सात",
    8: "आठ", 9: "नऊ", 10: "दहा", 11: "अकरा", 12: "बारा", 13: "तेरा", 14: "चौदा",
    15: "पंधरा", 16: "सोळा", 17: "सतरा", 18: "अठरा", 19: "एकोणीस", 20: "वीस",
    21: "एकवीस", 24: "चोवीस", 25: "पंचवीस", 30: "तीस", 40: "चाळीस", 50: "पन्नास",
    100: "शंभर",
}

# English words — used to convert a row whose audio is English (reviewed=no).
ENGLISH_CARDINAL = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty", 24: "twenty four", 25: "twenty five",
    30: "thirty", 40: "forty", 50: "fifty", 100: "hundred",
}
ENGLISH_ORDINAL = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
    7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}
# Marathi clock readings — on-the-hour and half-past are the only forms in this data.
# 1:30/2:30 are irregular (दीड/अडीच); X:30 for X>=3 is साडे+hour. :15/:20/:45 -> manual.
MARATHI_HALF = {1: "दीड", 2: "अडीच"}

# A whole-number token, NOT part of a time (adjacent ':') and NOT an ordinal ('1st').
# Ranges like 2-3 are allowed through so each side can be suggested (दोन-तीन).
DIGIT_TOKEN = re.compile(r"\d+(?::\d+)?(?:st|nd|rd|th)?")
SUGGEST_TOKEN = re.compile(r"(?<![\d:])(\d{1,3})(?![\d:])(?!st|nd|rd|th)")
TIME_TOKEN = re.compile(r"(\d{1,2}):(\d{2})")
ORDINAL_TOKEN = re.compile(r"(\d{1,3})(?:st|nd|rd|th)")


def detected_tokens(text: str) -> str:
    return " ".join(DIGIT_TOKEN.findall(text)) or "-"


def _suggest_time(m: "re.Match") -> str:
    h, mm = int(m.group(1)), int(m.group(2))
    if mm == 0:                                   # 7:00 -> सात ("वाजता" usually follows)
        w = MARATHI_CARDINAL.get(h)
    elif mm == 30:                                # 10:30 -> साडेदहा ; 1:30/2:30 irregular
        w = MARATHI_HALF.get(h) or ("साडे" + MARATHI_CARDINAL[h] if h in MARATHI_CARDINAL else None)
    else:
        w = None                                  # :15/:20/:45 -> leave for manual
    return w if w else m.group(0)


def suggest(text: str) -> str:
    """Default substitution -> Marathi words: times first, then standalone/range cardinals."""
    text = TIME_TOKEN.sub(_suggest_time, text)
    def repl(m: "re.Match") -> str:
        word = MARATHI_CARDINAL.get(int(m.group(1)))
        return word if word is not None else m.group(0)
    return SUGGEST_TOKEN.sub(repl, text)


def english_convert(text: str) -> str:
    """Convert numeric tokens to English words — used when the audio is English (reviewed=no)."""
    text = ORDINAL_TOKEN.sub(lambda m: ENGLISH_ORDINAL.get(int(m.group(1)), m.group(0)), text)
    def repl(m: "re.Match") -> str:
        word = ENGLISH_CARDINAL.get(int(m.group(1)))
        return word if word is not None else m.group(0)
    return SUGGEST_TOKEN.sub(repl, text)


# --- metadata.csv contract (mirrors transcribe_saaras.py) ---------------------------------

def load_metadata(path: Path) -> "list[tuple[str, str]]":
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        rel, _, text = line.partition("|")
        rows.append((rel.strip(), text.strip()))
    return rows


def write_metadata(path: Path, rows: "dict[str, str]", audio_dir: Path) -> None:
    lines = []
    for wav in sorted(audio_dir.glob("chunk_*.wav")):
        rel = str(wav.relative_to(audio_dir.parent))
        lines.append(f"{rel}|{rows.get(rel, '')}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --- worklist -----------------------------------------------------------------------------

def load_worklist(path: Path) -> "list[dict]":
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def save_worklist(path: Path, items: "list[dict]") -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(items)


# --- subcommands --------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> None:
    meta = load_metadata(args.metadata_csv)
    prior = {it["audio_path"]: it for it in load_worklist(args.worklist)}
    items = []
    for rel, text in meta:
        if not re.search(r"\d", text):
            continue
        keep = prior.get(rel)
        if keep:  # preserve the user's reviewed flag + any hand-edits; refresh detection
            keep["current_text"] = text
            keep["detected"] = detected_tokens(text)
            items.append(keep)
            continue
        s = suggest(text)
        items.append({
            "audio_path": rel,
            "detected": detected_tokens(text),
            "current_text": text,
            "suggested_text": s,
            "corrected_text": s,   # pre-filled with the Marathi suggestion
            # Default 'yes' = accept the suggestion. Flip to 'no' ONLY when the audio is
            # English (apply then writes the English words instead of the Marathi suggestion).
            "reviewed": "yes",
        })
    save_worklist(args.worklist, items)
    eng = sum(1 for it in items if (it.get("reviewed") or "").strip().lower() == "no")
    print(f"Wrote {args.worklist} — {len(items)} flagged rows "
          f"(default yes=Marathi suggestion; {eng} marked no=English). "
          f"Listen and flip English-audio rows to 'no'.")


def _play(wav: Path) -> None:
    if not wav.exists():
        print(f"  ! audio missing: {wav}")
        return
    try:
        subprocess.run(["afplay", str(wav)], check=False)
    except FileNotFoundError:
        print("  ! afplay not found (macOS only) — open the file manually.")


def cmd_review(args: argparse.Namespace) -> None:
    items = load_worklist(args.worklist)
    if not items:
        print(f"No worklist at {args.worklist} — run `extract` first.")
        return
    audio_dir = args.metadata_csv.parent
    todo = [it for it in items if it["reviewed"] != "yes"]
    print(f"{len(todo)} of {len(items)} rows left. Keys: [Enter]=accept suggestion · "
          f"type=custom · r=replay · s=skip · q=save&quit\n")
    for n, it in enumerate(todo, 1):
        wav = audio_dir / it["audio_path"]
        print(f"[{n}/{len(todo)}] {it['audio_path']}   (numbers: {it['detected']})")
        print(f"  current  : {it['current_text']}")
        print(f"  suggested: {it['suggested_text']}")
        _play(wav)
        while True:
            ans = input("  > corrected (Enter=suggested): ")
            if ans == "r":
                _play(wav)
                continue
            if ans == "s":
                print("  skipped.\n")
                break
            if ans == "q":
                save_worklist(args.worklist, items)
                print("Saved. Resume anytime with `review`.")
                return
            it["corrected_text"] = it["suggested_text"] if ans == "" else ans
            it["reviewed"] = "yes"
            save_worklist(args.worklist, items)  # crash-safe: persist after every row
            print(f"  saved -> {it['corrected_text']}\n")
            break
    save_worklist(args.worklist, items)
    left = sum(1 for it in items if it["reviewed"] != "yes")
    print(f"Done this pass. {left} rows still unreviewed.")


def cmd_apply(args: argparse.Namespace) -> None:
    items = load_worklist(args.worklist)
    # reviewed=yes -> write corrected_text (Marathi suggestion, or your hand-edit).
    # reviewed=no  -> the audio is English; write the English-word form of current_text.
    # anything else -> skip (left unchanged, reported).
    applied = {}
    n_yes = n_no = n_skip = 0
    for it in items:
        rv = (it.get("reviewed") or "").strip().lower()
        if rv == "yes":
            applied[it["audio_path"]] = (it.get("corrected_text") or "").strip() \
                or (it.get("suggested_text") or "").strip()
            n_yes += 1
        elif rv == "no":
            applied[it["audio_path"]] = english_convert(it.get("current_text") or "")
            n_no += 1
        else:
            n_skip += 1
    if not applied:
        print("No yes/no rows to apply — run `extract` first.")
        return
    meta = load_metadata(args.metadata_csv)
    rows = {rel: text for rel, text in meta}
    changed = 0
    for rel, corrected in applied.items():
        if rel in rows and rows[rel] != corrected:
            rows[rel] = corrected
            changed += 1
    backup = args.metadata_csv.with_suffix(args.metadata_csv.suffix + ".bak")
    shutil.copy2(args.metadata_csv, backup)
    write_metadata(args.metadata_csv, rows, args.metadata_csv.parent / "audio")
    residual = sum(1 for _, t in load_metadata(args.metadata_csv) if re.search(r"\d", t))
    print(f"Backed up -> {backup}")
    print(f"Applied {changed} changes to {args.metadata_csv} "
          f"(yes/Marathi={n_yes}, no/English={n_no}, skipped={n_skip}).")
    print(f"Rows still containing a digit in the transcript: {residual} "
          f"(should be only intentionally-kept numerics).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    p.add_argument("--worklist", type=Path, default=WORKLIST_CSV)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract", help="build the review worklist from metadata.csv")
    sub.add_parser("review", help="interactively review flagged clips by ear")
    sub.add_parser("apply", help="merge corrections back into metadata.csv (+ .bak)")
    args = p.parse_args()
    {"extract": cmd_extract, "review": cmd_review, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
