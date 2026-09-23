#!/usr/bin/env python3
"""A/B one language's SPEAKER NAME in the caption, locally. No RunPod, no rebuild.

WHY THIS EXISTS. English output improved markedly when its caption speaker went
from "Sunita" (a MARATHI-only voice on the Indic Parler model card) to "Mary" (a
recommended ENGLISH one) — see betacraft_core.MODELS. That raised the obvious
question: would Marathi improve with Isha/Radha, or Hindi with Rani/Aman?

The two cases are NOT symmetric, which is the thing this script is for measuring
rather than assuming:

  * English was never fine-tuned, so the base model's English speaker conditioning
    survived intact. Asking for Mary invoked knowledge that was already there.
  * Marathi and Hindi WERE fine-tuned, on exactly two names (CLAUDE.md v7:
    "2 speakers: Sunita=Marathi, Divya=Hindi", captions byte-identical
    train<->inference). Isha/Radha/Rani/Aman appeared in ZERO training examples,
    so asking for them is asking a fine-tune for an identity it was trained away
    from. Expect neutral-to-worse — but measure it.

Speaker names are hardcoded in betacraft_core.MODELS with no env var or request
field, so trying one normally means a docker build + push + RunPod release. This
swaps the caption on an already-built bundle instead: everything else — the same
splitting, packing, retry ladder and edge-trimming — runs untouched, so only the
name differs.

CAVEAT, so the numbers are not over-read: this runs on MPS/CPU in fp32 where
production runs CUDA in bf16 (see _dtype_for_device). Audio should be equivalent
but TIMING is not comparable, and the defect rate may not be either.

USAGE
    # Marathi: the trained name vs an untrained one, 3 seeds each
    venv/bin/python scripts/narrate/speaker_ab.py \
        --text "अचानक रॉकेट वेगाने वर जाऊ लागले आणि रुद्राने खिडकीतून पाहिले." \
        --lang mr --speakers Sunita,Isha --seeds 648,42,1042

    # Hindi
    venv/bin/python scripts/narrate/speaker_ab.py \
        --text "क्या रोशनी को गोलू को बताना चाहिए?" \
        --lang hi --speakers Divya,Rani --seeds 648,42,1042

Then LISTEN to the wavs it writes. Duration ratio is printed because it is free,
but score_quality.py's header already established that duration cannot see a
dropped word — the ear is the ground truth here too.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from betacraft_core import (  # noqa: E402
    CAPTION_TEMPLATE,
    EXPECTED_WORDS_PER_SEC,
    WORDS_PER_SEC_BY_LANG,
    get_bundle,
    synthesize,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="mr", choices=["mr", "hi", "en"])
    ap.add_argument(
        "--speakers",
        help="Comma-separated caption NAMES to compare, e.g. Sunita,Isha. Each is "
             "substituted into CAPTION_TEMPLATE. Mutually exclusive with --caption.",
    )
    ap.add_argument(
        "--caption",
        action="append",
        metavar="LABEL::TEXT",
        help="Full description to use VERBATIM, repeatable to compare several. "
             "Format 'label::caption text'; the label names the output wav. "
             "Use this for --model base, which is caption-STEERABLE by design. "
             "For the fine-tune the caption is an identity token trained "
             "byte-identical (see betacraft_core.CAPTION_TEMPLATE), so anything "
             "but the template moves it off-distribution — expect worse, not better.",
    )
    ap.add_argument("--seeds", default="648,42,1042", help="Comma-separated seeds")
    ap.add_argument("--model", default="betacraft", choices=["betacraft", "base"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument(
        "--top-p",
        type=float,
        default=None,
        dest="top_p",
        help="Nucleus sampling. Omit to use betacraft_core's DEFAULT_TOP_P; pass "
             "1.0 for the library default (no truncation). NOTE: the duration "
             "thresholds that drive the reseed-retry ladder are module constants "
             "read from the environment at IMPORT time, so to vary them set "
             "BETACRAFT_MAX_DURATION_MULT / BETACRAFT_SOFT_DURATION_MULT before "
             "launching — they cannot change between arms inside one process.",
    )
    ap.add_argument("--out", default="speaker_ab")
    args = ap.parse_args()

    if bool(args.speakers) == bool(args.caption):
        ap.error("pass exactly one of --speakers or --caption")

    # Both modes reduce to the same thing: an ordered list of (label, caption).
    if args.caption:
        arms = []
        for i, raw in enumerate(args.caption, 1):
            label, sep, text = raw.partition("::")
            if not sep:
                label, text = f"cap{i}", raw
            arms.append((label.strip(), text.strip()))
    else:
        arms = [(n.strip(), CAPTION_TEMPLATE.format(name=n.strip()))
                for n in args.speakers.split(",") if n.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out) / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = get_bundle(args.model)
    # The description tokenizer is NOT kept on the bundle (get_bundle encodes the
    # captions once and keeps only the ids/mask), so rebuild it from the same
    # place get_bundle does: the text encoder named in the model's own config.
    desc_tok = AutoTokenizer.from_pretrained(
        bundle["model"].config.text_encoder._name_or_path
    )

    rate = WORDS_PER_SEC_BY_LANG.get(args.lang, EXPECTED_WORDS_PER_SEC)
    expected_sec = len(args.text.split()) / rate
    print(f"\n{args.model} | lang={args.lang} | {len(args.text.split())} words "
          f"| expected ~{expected_sec:.2f}s at {rate} w/s")
    print(f"writing to {out_dir}\n")

    # Record what each label actually meant, so a wav is traceable to its caption
    # after the fact — the log line inside synthesize() prints the HARDCODED
    # MODELS speaker name and cannot see the override applied here.
    (out_dir / "captions.txt").write_text(
        f"model={args.model} lang={args.lang} seeds={seeds}\n"
        f"temperature={args.temperature if args.temperature is not None else 'DEFAULT'} "
        f"top_p={args.top_p if args.top_p is not None else 'DEFAULT'}\n"
        f"BETACRAFT_MAX_DURATION_MULT={os.environ.get('BETACRAFT_MAX_DURATION_MULT','unset')} "
        f"BETACRAFT_SOFT_DURATION_MULT={os.environ.get('BETACRAFT_SOFT_DURATION_MULT','unset')}\n"
        f"text={args.text}\n\n"
        + "\n".join(f"{label}\n    {caption}\n" for label, caption in arms),
        encoding="utf-8",
    )

    rows = []
    for name, caption in arms:
        print(f"  [{name}] {caption}")
        enc = desc_tok(caption, return_tensors="pt").to(bundle["device"])
        # Swap THIS language's caption on the live bundle. Everything downstream —
        # splitting, packing, the reseed-retry ladder, edge trimming — is the real
        # path, so the caption is the only variable.
        bundle["desc_ids_by_lang"][args.lang] = enc.input_ids
        bundle["desc_mask_by_lang"][args.lang] = enc.attention_mask

        for seed in seeds:
            wav, sr, meta = synthesize(
                bundle,
                args.text,
                language=args.lang,
                seed=seed,
                temperature=args.temperature,
                gen_overrides=({"top_p": args.top_p}
                               if args.top_p is not None else None),
            )
            path = out_dir / f"{name}_s{seed}.wav"
            sf.write(path, wav, sr)
            audio_sec = len(wav) / sr
            attempts = sum(m.get("attempts", 1) for m in meta if not m.get("failed"))
            suspect = sum(1 for m in meta if m.get("suspect"))
            rows.append((name, seed, audio_sec, audio_sec / expected_sec, attempts, suspect))
            print(f"  {name:10} seed={seed:<6} {audio_sec:5.2f}s  "
                  f"{audio_sec / expected_sec:4.2f}x  attempts={attempts}  "
                  f"suspect={suspect}  -> {path.name}")

    print("\n--- summary (LISTEN before trusting any of this) ---")
    for name, _caption in arms:
        mine = [r for r in rows if r[0] == name]
        if not mine:
            continue
        ratios = sorted(r[3] for r in mine)
        median = ratios[len(ratios) // 2]
        print(f"  {name:10} n={len(mine)}  median {median:4.2f}x  "
              f"retries={sum(r[4] for r in mine) - len(mine)}  "
              f"suspect={sum(r[5] for r in mine)}")
    print(f"\nwavs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
