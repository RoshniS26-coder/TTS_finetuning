#!/usr/bin/env python3
"""Score the GPU (bf16) and Mac (fp32) probe clips BLIND, side by side.

WHY THIS EXISTS. The open question on 2026-09-08 is whether word-dropping happens
only on the GPU. If it does, bfloat16's coarser rounding flipping the end-of-speech
decision is the leading explanation, and running production in float32 becomes a
real candidate fix. If it does not, precision is eliminated and the Mac/GPU gap
stops mattering.

Two things make that question hard to answer honestly:

  * THE LOCAL DETECTOR CANNOT SEE IT. local_repro's SHORT_RATIO of 0.60 only flags
    a clip that lost 40%+ of its duration. Dropping the final two words of eleven
    leaves ~82%, which sails through. So `short_rate: 0.0` across the local runs is
    not evidence of absence — nobody has listened.
  * EXPECTATION BIASES THE EAR. Anyone who believes "it never happens on Mac" will
    hear it that way. So playback is BLIND by default: the pair is shuffled, you
    score both, and only then is the platform revealed.

The two probe sets pair exactly — same 19 sentences, same seeds (42, 1042), 38
clips each — so every judgement is one variable: bf16 vs fp32.

USAGE
    venv/bin/python scripts/narrate/pair_listen.py            # blind, resumable
    venv/bin/python scripts/narrate/pair_listen.py --report   # print the table only
    venv/bin/python scripts/narrate/pair_listen.py --no-blind # reveal as you go

Verdicts are written after every keypress, so quitting mid-way loses nothing.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPU_RUN = ROOT / "quality_runs" / "20260903-150327"
MAC_RUN = ROOT / "local_repro" / "20260904-152523"
VERDICTS = ROOT / "quality_runs" / "paired_listen.json"

# One keypress each. "dropped" and "stalled" are kept apart deliberately: they are
# the two halves of the same end-of-speech defect, and only ONE of them (dropped)
# is what this run is trying to attribute to precision.
CHOICES = {
    "g": "good",
    "d": "dropped",   # a word or words never spoken
    "s": "stalled",   # a held/stretched sound, or a sung tail
    "b": "bad",       # wrong in some other way
}


def load_pairs() -> list[dict]:
    """Every (sentence, seed) present in BOTH runs, with each side's wav path."""
    gpu = json.loads((GPU_RUN / "report.json").read_text())
    mac = json.loads((MAC_RUN / "report.json").read_text())

    gpu_rows, gpu_seed_verdict = {}, {}
    for lang, block in gpu["languages"].items():
        for r in block["rows"]:
            if not r.get("wav"):
                continue
            gpu_rows[(r["text"], r["seed"])] = (lang, r["wav"], r.get("check"))
            if r.get("verdict"):
                gpu_seed_verdict[(r["text"], r["seed"])] = r["verdict"]

    mac_rows = {}
    for case in mac["cases"]:
        text = case["summary"]["text"]
        for run in case["runs"]:
            mac_rows[(text, run["seed"])] = run["wav"]

    pairs = []
    for key in sorted(set(gpu_rows) & set(mac_rows), key=lambda k: (gpu_rows[k][0], k[1], k[0])):
        lang, gpu_wav, check = gpu_rows[key]
        pairs.append({
            "text": key[0], "seed": key[1], "lang": lang, "check": check,
            "gpu_wav": str(GPU_RUN / gpu_wav), "mac_wav": str(MAC_RUN / mac_rows[key]),
            # Carry the one verdict already recorded by ear so it isn't re-scored.
            "prior_gpu": gpu_seed_verdict.get(key),
        })
    return pairs


def play(path: str) -> None:
    if not Path(path).exists():
        print(f"    !! missing file: {path}")
        return
    subprocess.run(["afplay", path], check=False)


def ask(label: str, path: str) -> str | None:
    """Play one clip and take a verdict. None means quit."""
    while True:
        play(path)
        print(f"    {label}  [g]ood  [d]ropped words  [s]talled/stretched  [b]ad other  "
              f"[r]epeat  [q]uit: ", end="")
        ans = input().strip().lower()
        if ans == "q":
            return None
        if ans == "r":
            continue
        if ans in CHOICES:
            return CHOICES[ans]
        print("    ? use g / d / s / b / r / q")


def summarise(store: dict, pairs: list[dict]) -> None:
    """Defect rates per platform per language — the whole point of the exercise."""
    langs = sorted({p["lang"] for p in pairs})
    print("\n" + "=" * 78)
    print("PAIRED RESULT — same text, same seed, bf16 (GPU) vs fp32 (Mac)")
    print("=" * 78)
    header = f"{'':<6}{'scored':>8}{'good':>8}{'DROPPED':>10}{'stalled':>9}{'bad':>6}{'defect rate':>13}"
    for platform in ("gpu", "mac"):
        print(f"\n{platform.upper()}")
        print(header)
        for lang in langs + ["ALL"]:
            rows = [p for p in pairs if lang in (p["lang"], "ALL")]
            vs = [store.get(f"{p['text']}||{p['seed']}||{platform}") for p in rows]
            vs = [v for v in vs if v]
            if not vs:
                continue
            n = len(vs)
            counts = {k: sum(1 for v in vs if v == k) for k in ("good", "dropped", "stalled", "bad")}
            rate = (n - counts["good"]) / n * 100
            print(f"{lang:<6}{n:>8}{counts['good']:>8}{counts['dropped']:>10}"
                  f"{counts['stalled']:>9}{counts['bad']:>6}{rate:>12.0f}%")

    # The single question this run exists to answer.
    gpu_drop = sum(1 for p in pairs if store.get(f"{p['text']}||{p['seed']}||gpu") == "dropped")
    mac_drop = sum(1 for p in pairs if store.get(f"{p['text']}||{p['seed']}||mac") == "dropped")
    scored = sum(1 for p in pairs
                 if store.get(f"{p['text']}||{p['seed']}||gpu")
                 and store.get(f"{p['text']}||{p['seed']}||mac"))
    print("\n" + "-" * 78)
    print(f"Pairs scored on BOTH platforms: {scored}/{len(pairs)}")
    print(f"Clips with DROPPED words —  GPU (bf16): {gpu_drop}    Mac (fp32): {mac_drop}")
    if scored >= 10:
        if gpu_drop and not mac_drop:
            print("\n  => Dropping is GPU-ONLY across these pairs. bfloat16 rounding flipping the")
            print("     end-of-speech decision is now the leading explanation, and running the")
            print("     endpoint in float32 is the experiment worth paying for.")
        elif gpu_drop and mac_drop:
            print("\n  => Dropping occurs on BOTH platforms. Precision is NOT the cause; it is a")
            print("     property of the model. Stop pursuing the Mac/GPU gap.")
        elif not gpu_drop and not mac_drop:
            print("\n  => No dropping in either set. This probe set does not reproduce the defect —")
            print("     score the failing app sentences instead of these.")
    else:
        print("  (score at least 10 pairs before reading anything into this)")
    print(f"\nVerdicts: {VERDICTS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print the table and exit")
    ap.add_argument("--no-blind", action="store_true", help="reveal the platform while scoring")
    a = ap.parse_args()

    if sys.platform != "darwin":
        sys.exit("playback uses afplay — run this on the Mac")

    pairs = load_pairs()
    store = json.loads(VERDICTS.read_text()) if VERDICTS.exists() else {}
    # Fold in the verdict already recorded directly in the GPU report.
    for p in pairs:
        if p["prior_gpu"]:
            store.setdefault(f"{p['text']}||{p['seed']}||gpu", p["prior_gpu"])

    if a.report:
        summarise(store, pairs)
        return

    todo = [p for p in pairs
            if not (store.get(f"{p['text']}||{p['seed']}||gpu")
                    and store.get(f"{p['text']}||{p['seed']}||mac"))]
    print(f"\n{len(pairs)} pairs, {len(pairs) - len(todo)} already scored, {len(todo)} to go.")
    print("Each pair is the SAME sentence at the SAME seed on two machines.")
    if not a.no_blind:
        print("Order is shuffled and the platform is hidden until you have scored both.")
    print("Listen for: is every word actually spoken?\n")

    for n, p in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {p['lang']}  seed {p['seed']}")
        print(f"    {p['text']}")
        if p.get("check"):
            print(f"    check: {p['check']}")

        sides = [("gpu", p["gpu_wav"]), ("mac", p["mac_wav"])]
        if not a.no_blind:
            random.shuffle(sides)

        results = {}
        for i, (platform, wav) in enumerate(sides):
            key = f"{p['text']}||{p['seed']}||{platform}"
            if key in store:
                results[platform] = store[key]
                continue
            label = f"{platform.upper()} :" if a.no_blind else f"clip {'AB'[i]}  :"
            verdict = ask(label, wav)
            if verdict is None:
                summarise(store, pairs)
                return
            store[key] = verdict
            results[platform] = verdict
            VERDICTS.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")

        if not a.no_blind:
            order = " / ".join(f"{'AB'[i]}={pl.upper()}" for i, (pl, _) in enumerate(sides))
            print(f"    -> {order}   GPU={results['gpu']}  Mac={results['mac']}\n")
        else:
            print()

    summarise(store, pairs)


main()
