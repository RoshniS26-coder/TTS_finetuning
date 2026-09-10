"""Reproduce a TTS defect LOCALLY, many times, and measure its RATE. No RunPod.

WHY THIS EXISTS. Every hypothesis about the word-skipping defect (2026-09-03/04)
was tested against ONE audio clip and then contradicted by the next one:
segmentation, a hardcoded danda, unseen proper nouns, quotation marks, premature
EOS, and clean_edges over-trimming were each proposed and each eliminated. The
common flaw was the method, not the theories — a single clip from a model that
samples randomly proves almost nothing.

The defect is stochastic: the SAME sentence at seed 42 produced "ढप्प" correctly
once and dropped it another time. So the only honest measurement is a FAILURE
RATE over many generations, and the only affordable way to get one is locally.

This runs the real betacraft_core generation path on the Mac (MPS or CPU) using
the model already in the HF cache. It costs nothing, so a rate can be measured
before and after any change instead of guessing from one listen.

CAVEATS, so the numbers are not over-read:
  * MPS runs fp32 where CUDA runs bf16 (see _load_model). Audio should be
    equivalent but TIMING here is not comparable to production.
  * It is slow — expect roughly 30-60s per short sentence versus ~6s on an L4.
    Fine for 20 generations of one sentence; impractical for a whole story.

USAGE
    # measure the rate for one sentence over 20 seeds
    venv/bin/python scripts/narrate/local_repro.py \
        --text "रोहनला स्वतःहून हळूहळू चालत जायला आवडेल का?" --lang mr --runs 20

    # the four sentences reported as dropping words, 10 seeds each
    venv/bin/python scripts/narrate/local_repro.py --known-failures --runs 10

    # test a change: set any generation knob and compare the rate
    venv/bin/python scripts/narrate/local_repro.py --known-failures --runs 10 \
        --temperature 0.55
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The sentences actually reported as losing words, with what went missing, so a
# run can be scored against the real complaint rather than a synthetic one.
KNOWN_FAILURES = [
    ("mr", "रोहनला स्वतःहून हळूहळू चालत जायला आवडेल का?", "आवडेल का?"),
    ("mr", "आरवला खिडकीतून पृथ्वी एका छोट्या निळ्या चेंडूसारखी दिसू लागली.", "पृथ्वी एका छोट्या निळ्या चेंडूसारखी दिसू लागली"),
    ("hi", "और उसे लगा कि अकेले जाने में उतना मज़ा नहीं है।", "उतना मज़ा नहीं है।"),
    ("en", "The berries look so juicy and yummy to eat right now.", "right now"),
]

# A generation this far below its expected duration has almost certainly lost
# words. Deliberately generous: EXPECTED_WORDS_PER_SEC is Marathi-calibrated and
# English legitimately runs at ~0.68-0.84x of it (measured, production logs
# 2026-09-04), so a tighter bound would flag normal English as defective.
SHORT_RATIO = 0.60


def save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    w = wave.open(str(path), "w")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--lang", default="mr", choices=["mr", "hi", "en"])
    ap.add_argument("--known-failures", action="store_true")
    ap.add_argument("--probes", action="store_true",
                    help="the 19-sentence probe set from score_quality.py, at the SAME "
                         "seeds (42, 1042) the 2026-09-03 RunPod run used — so local fp32 "
                         "output is directly comparable, clip for clip, with the bf16 clips "
                         "already in quality_runs/")
    ap.add_argument("--seeds", help="comma-separated seeds; overrides --runs")
    ap.add_argument("--runs", type=int, default=20, help="generations per sentence, each a different seed")
    ap.add_argument("--model", default="betacraft", choices=["betacraft", "base"])
    ap.add_argument("--temperature", type=float, default=None, help="None = server default")
    ap.add_argument("--failing-from", type=Path,
                    help="re-run only the sentences marked bad BY EAR in a previous run's "
                         "report.json. Reads the verdicts rather than hardcoding a list, so it "
                         "always reflects the latest scoring.")
    ap.add_argument("--strip-commas", action="store_true",
                    help="remove commas from the text before generating. Tests whether comma-chopped "
                         "fragments (countdowns, comma-fenced names) are what triggers the dropping "
                         "and trailing defects — commas do NOT affect unit splitting, so this is a "
                         "single-variable A/B against the same sentence with commas.")
    ap.add_argument("--pack-words", type=int,
                    help="group sentences into units of ~N words (0 = one sentence per unit, the "
                         "current default). Training clips held a MEDIAN OF 2 SENTENCES and ~20 words, "
                         "so 0 under-fills relative to what the model saw. Packing was A/B'd on "
                         "2026-08-31 and lost - but _merge_short was comma-joining short sentences "
                         "then, so every packed unit carried the comma runs we have since proven cause "
                         "hallucination. Worth re-testing now that it joins with punctuation intact.")
    ap.add_argument("--no-cap", action="store_true",
                    help="give every unit the model's own default budget (~29.8s) instead of a "
                         "per-sentence one. Tests what happens with no meaningful runaway stop: the "
                         "'hit the cap' signal is what currently triggers a reseed, so a unit that "
                         "never emits EOS will simply run on and ship.")
    ap.add_argument("--hard-cap", type=int,
                    help="override PACK_HARD_CAP_WORDS. A unit longer than this is split into "
                         "separate generations. Set high (e.g. 99) to keep long sentences WHOLE.")
    ap.add_argument("--texts-file", type=Path,
                    help="TSV of 'lang<TAB>text', one case per line.")
    ap.add_argument("--ellipsis", action="store_true",
                    help="replace commas with '...' — the form the training corpus actually uses "
                         "for counting (9 Marathi units, all ellipsis, 0 with commas). Keeps the "
                         "dramatic pause that --strip-commas throws away.")
    ap.add_argument("--commas-only", action="store_true",
                    help="keep only cases whose text contains a comma. Pointless to regenerate the "
                         "rest: --strip-commas would leave them byte-identical.")
    ap.add_argument("--greedy", action="store_true",
                    help="do_sample=False. Removes the random early-stop draw that makes the same "
                         "sentence drop words at one seed and not another. Output is deterministic, "
                         "so seed and temperature have NO effect and one run per sentence is enough.")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "local_repro")
    a = ap.parse_args()

    if not (a.text or a.known_failures or a.probes or a.failing_from or a.texts_file):
        ap.error("give --text, --known-failures, --probes, --failing-from or --texts-file")
    # MUST precede the betacraft_core import: pack_sentences binds hard_cap as a
    # default argument, which Python evaluates once at def time.
    if a.hard_cap:
        os.environ["BETACRAFT_PACK_HARD_CAP"] = str(a.hard_cap)
    if a.no_cap:
        # Every unit then clamps to MAX_TOKENS_CEILING regardless of its length.
        os.environ["BETACRAFT_MAX_DURATION_MULT"] = "99"
    # --probes reproduces the RunPod run's exact seeds so the two sets pair up.
    seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds
             else ([42, 1042] if a.probes else [42 + i * 101 for i in range(a.runs)]))

    # Imported late: this pulls in torch + parler_tts (multi-GB) and loads the
    # model, so argument errors above should fail fast instead of after a
    # 30-second import.
    import betacraft_core as core

    if a.probes:
        from score_quality import PROBES
        cases = [(lang, text, what) for lang in ("mr", "hi", "en")
                 for text, what in PROBES[lang]]
    elif a.texts_file:
        cases = [(ln.split("\t")[0].strip(), ln.split("\t")[1].strip(), None)
                 for ln in a.texts_file.read_text(encoding="utf-8").splitlines() if "\t" in ln]
    elif a.known_failures:
        cases = KNOWN_FAILURES
    else:
        cases = [(a.lang, a.text, None)]
    if a.failing_from:
        rep = json.loads(a.failing_from.read_text())
        seen, picked = set(), []
        for case in rep.get("cases", []):
            text, lang = case["summary"]["text"], case["summary"]["lang"]
            if text in seen:
                continue
            for run in case["runs"]:
                if run.get("verdict") == "bad":
                    picked.append((lang, text, run.get("note")))
                    seen.add(text)
                    break
        if not picked:
            sys.exit(f"no clip has verdict 'bad' in {a.failing_from}")
        cases = picked
    if a.commas_only:
        cases = [c for c in cases if "," in c[1]]
        if not cases:
            sys.exit("no case contains a comma")
    if a.ellipsis:
        cases = [(lang, text.replace(",", "..."), what) for lang, text, what in cases]
    if a.strip_commas:
        # Commas only; sentence-final marks (. ? ! ।) must stay or split_sentences
        # stops finding unit boundaries and the comparison is no longer one variable.
        cases = [(lang, text.replace(",", "").replace("، ", " "), what) for lang, text, what in cases]

    out = a.out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {a.model} …")
    t0 = time.time()
    bundle = core.get_bundle(a.model)
    print(f"  ready in {time.time()-t0:.0f}s on {bundle['device']}, sr={bundle['sr']}\n")

    report: dict = {"when": out.name, "model": a.model, "runs": len(seeds), "seeds": seeds,
                    "temperature": None if a.greedy else a.temperature,
                    "decoding": "greedy" if a.greedy else "sampling",
                    "commas_stripped": bool(a.strip_commas),
                    "commas_to_ellipsis": bool(a.ellipsis),
                    "hard_cap": a.hard_cap,
                    "pack_words": a.pack_words,
                    "device": str(bundle["device"]),
                    "short_ratio_threshold": SHORT_RATIO, "cases": []}

    for ci, (lang, text, expect_missing) in enumerate(cases):
        words = len(text.split())
        # Language-aware: English narrates at ~2.08 w/s against Marathi's 1.75, so
        # a single global rate made every English ratio read ~19% low.
        expected = core._expected_sec_for(words, lang)
        print(f"[{ci}] {lang}  {words} words, expect ~{expected:.2f}s")
        print(f"    {text}")
        if expect_missing:
            print(f"    reported missing: {expect_missing}")
        rows = []
        for run, seed in enumerate(seeds):
            t1 = time.time()
            try:
                wav, sr, meta = core.synthesize(
                    bundle, text=text, language=lang, seed=seed,
                    temperature=None if a.greedy else a.temperature,
                    gen_overrides={"do_sample": False} if a.greedy else None,
                    **({"pack_words": a.pack_words} if a.pack_words is not None else {}),
                )
            except Exception as exc:                       # noqa: BLE001
                print(f"    run {run:2d} seed={seed:<5} FAILED: {exc}")
                rows.append({"seed": seed, "error": str(exc)})
                continue
            dur = len(wav) / sr
            ratio = dur / expected
            short = ratio < SHORT_RATIO
            flux = core._tail_flux(wav, sr)
            stalled = flux is not None and flux < core.TAIL_FLUX_SUSPECT
            name = f"c{ci}_{lang}_s{seed}.wav"
            save_wav(out / name, wav, sr)
            rows.append({"seed": seed, "audio_sec": round(dur, 2), "ratio": round(ratio, 2),
                         "short": short, "tail_flux": None if flux is None else round(flux, 4),
                         "stalled": stalled, "wav": name,
                         "gen_sec": round(time.time() - t1, 1),
                         "attempts": meta[0].get("attempts") if meta else None})
            flag = "SHORT" if short else ("STALL" if stalled else "ok")
            print(f"    run {run:2d} seed={seed:<5} {dur:5.2f}s  ratio {ratio:4.2f}  "
                  f"flux {('n/a' if flux is None else f'{flux:.3f}'):>6}  {flag}")
        scored = [r for r in rows if "ratio" in r]
        n_short = len([r for r in scored if r["short"]])
        n_stall = len([r for r in scored if r["stalled"]])
        summary = {
            "lang": lang, "text": text, "words": words,
            "expected_sec": round(expected, 2), "generations": len(scored),
            "short": n_short, "stalled": n_stall,
            "short_rate": round(n_short / len(scored), 3) if scored else None,
            "stall_rate": round(n_stall / len(scored), 3) if scored else None,
            "median_ratio": round(float(np.median([r["ratio"] for r in scored])), 2) if scored else None,
        }
        report["cases"].append({"summary": summary, "runs": rows})
        print(f"    -> short {n_short}/{len(scored)}   stalled {n_stall}/{len(scored)}   "
              f"median ratio {summary['median_ratio']}\n")

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print(f"{'case':5} {'lang':5} {'words':>5} {'gens':>5} {'short':>6} {'stalled':>8} {'med ratio':>10}")
    for i, c in enumerate(report["cases"]):
        s = c["summary"]
        # median_ratio is None when every attempt for this case failed (an MPS
        # out-of-memory, say), so it must not be format-specified as a number —
        # otherwise one dead case takes down the summary for all the good ones.
        med = "—" if s.get("median_ratio") is None else f"{s['median_ratio']:.2f}"
        print(f"{i:<5} {s['lang']:5} {s['words']:>5} {s['generations']:>5} "
              f"{s['short']:>6} {s['stalled']:>8} {med:>10}")
    print(f"\nwavs + report.json -> {out}")
    print("A 'SHORT' flag is a HINT, not proof — listen to those clips and confirm "
          "words are actually missing before trusting the rate.")


if __name__ == "__main__":
    main()
