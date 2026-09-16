#!/usr/bin/env python3
"""Model loading + generation for the fine-tuned Betacraft Parler-TTS model.

This is the SHARED core, deliberately free of any web-framework or RunPod
imports, so the exact same generation code backs both entrypoints:

  * api_server.py  — FastAPI, used for local dev on the Mac (and for the old
                     RunPod Load Balancer endpoint, which serves plain HTTP).
  * rp_handler.py  — RunPod Serverless QUEUE handler, which has no HTTP server
                     at all: it polls RunPod's job queue and returns JSON.

Queue endpoints are what let us (a) escape the Load Balancer's ~30-40s gateway
timeout that forced every /synthesize call to be split into tiny 2-sentence
batches client-side, and (b) get real job distribution across workers, which
the Load Balancer's request-count autoscaler would not do for small bursts.

Keeping generation here (rather than in either entrypoint) means a change to
sampling params or the hallucination-retry logic can never drift between the
local dev path and the deployed path.
"""
from __future__ import annotations

import io
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# narrate_story.py lives next to this file and isn't a package (no __init__.py) —
# add its directory to sys.path so it can be imported directly. This also reuses
# its DAC padding_mask monkeypatch, which runs at import time.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from narrate_story import (  # noqa: E402
    clean_edges,
    split_sentences,
)
from parler_tts import ParlerTTSForConditionalGeneration  # noqa: E402
from transformers import AutoTokenizer, set_seed  # noqa: E402

MODEL_DIR = os.environ.get("BETACRAFT_MODEL_DIR", "roshni-sorigin/mar-hin-betacraft-tts")
BASE_MODEL_DIR = os.environ.get("BETACRAFT_BASE_MODEL_DIR", "ai4bharat/indic-parler-tts")
DEVICE = os.environ.get("BETACRAFT_DEVICE")  # None -> auto-detect

# Byte-identical to the v7 training caption (CLAUDE.md "v7 SYNTHETIC DATASET"),
# only the speaker name differs. For the FINE-TUNED model this string must not
# be edited — voice identity is caption-driven and the model was trained on
# exactly these bytes, so any change moves it off-distribution.
CAPTION_TEMPLATE = (
    "{name} narrates a children's story in an expressive, warm, animated "
    "storytelling tone with emotional variation, at a moderate pace. "
    "Very clear audio with no background noise."
)

# --- Model registry ---------------------------------------------------------
# Two selectable models, so the product can offer (and compare) a cheaper
# off-the-shelf voice against the fine-tune:
#
#   "betacraft" — the fine-tune. Trained ONLY on Marathi + Hindi. English runs
#                 off-label through the Marathi speaker; usable in practice
#                 (the Indic Parler base includes Indian-accented English, so
#                 the fine-tune inherits some of it) but it is not a trained
#                 language and should be judged on its own.
#
#   "base"      — stock ai4bharat/indic-parler-tts, which DOES support English
#                 as a first-class language. Speakers below are the model
#                 card's recommended ones for each language: Mary (English),
#                 Divya (Hindi), Sunita (Marathi). Divya/Sunita deliberately
#                 match the fine-tune's speaker names so an A/B compares the
#                 training, not two unrelated voices.
MODELS = {
    "betacraft": {
        "dir": MODEL_DIR,
        # en: "Mary" UNDER TEST from 2026-09-10, was "Sunita". The Indic-Parler
        # model card lists Sunita for MARATHI ONLY; its English speakers are
        # Thoma and Mary. So English was asking a Marathi-only speaker name to
        # narrate English — off-distribution on top of a fine-tune that contains
        # no English at all. The caption is name-parameterised, so this changes
        # the name and nothing else. UNVERIFIED: the fine-tune saw only Marathi
        # and Hindi, so whether it still responds to "Mary" is exactly what the
        # A/B against the seed-648 English clips is for. Revert to "Sunita" if
        # Mary sounds worse.
        "speakers": {"mr": "Sunita", "hi": "Divya", "en": "Mary"},
    },
    "base": {
        "dir": BASE_MODEL_DIR,
        "speakers": {"mr": "Sunita", "hi": "Divya", "en": "Mary"},
    },
}
DEFAULT_MODEL = "betacraft"

# Superset of languages any model serves; per-model support is in MODELS.
LANGUAGE_SPEAKERS = {"mr": "Sunita", "hi": "Divya", "en": "Mary"}

log = logging.getLogger("betacraft_core")

# Duration-based hallucination heuristic. See _max_new_tokens_for.
#
# MEASURED 2026-08-31 on the A5000 from clean single-attempt generations
# (25w->14.56s, 10w->5.41s, 7w->3.99s): the model narrates at ~1.75 words/sec.
# This was 2.2 (inherited, never verified against real output), which made
# every token budget ~25% too small and truncated legitimate speech — the
# truncation then looked like a runaway and triggered a pointless reseeded
# retry, which is itself a source of drift. Do not raise this without
# re-measuring: too high silently re-creates that failure.
EXPECTED_WORDS_PER_SEC = 1.75
# PER-LANGUAGE as of 2026-09-07. The 1.75 above was measured on MARATHI output
# and then applied to every language, which made the English token budget ~19%
# too generous: production logs (2026-09-04, 75 units) show English narrates at
# ~2.08 words/sec, so normal English landed at 0.68-0.84x "expected" and an
# English runaway had to reach ~3x its true length before the cap stopped it.
# Confirmed by quality_runs/20260903-150327 en_02_r0: 10 words of English ran
# 14.71s (3.06x the 4.81s it should take) and was only caught because it hit the
# ceiling. Hindi shares Marathi's rate until it is separately measured.
# HINDI RAISED 1.75 -> 2.1 on 2026-09-10. It had been borrowing Marathi's rate,
# which is the same mistake English carried until 2026-09-07. Measured on the
# seed-648 probe run (local_repro/20260910-100810), 6 of 6 Hindi clips came in
# SHORT of their expected length and none was defective by ear:
#     0.83  0.80  0.77  0.82  0.90  0.88   median 0.825
# Systematic, not noise. 1.75 / 0.825 = 2.12, so Hindi genuinely narrates at
# ~2.1 words/sec. Under the old rate every Hindi token budget was ~17% too
# generous, giving a runaway that much extra room before the cap stopped it.
#
# English 2.08 was measured on SUNITA's English. English now uses Mary, who
# narrates ~2.5 w/s (5 clips, median ratio 0.82). Left at 2.08 deliberately:
# raising it TIGHTENS the cap, and five samples is too thin to risk truncating
# legitimate speech. Extra headroom is harmless; re-measure with more clips.
WORDS_PER_SEC_BY_LANG = {"mr": 1.75, "hi": 2.1, "en": 2.08}
# A unit running this much longer than expected is a runaway. Env-tunable so the
# cap can be loosened, or effectively removed, without a rebuild: set it high
# (e.g. 99) and every unit clamps to MAX_TOKENS_CEILING (~29.8s), which is
# essentially the model's own generation_config default of max_length 2610.
MAX_DURATION_MULTIPLIER = float(os.environ.get("BETACRAFT_MAX_DURATION_MULT") or 2.5)
MAX_CHUNK_ATTEMPTS = 3  # original attempt + up to 2 retries with a different seed

# --- Unit packing -----------------------------------------------------------
# Match the TRAINING distribution. The v7 corpus was built by sentence-packing
# to a 24-word target / 34-word hard cap / 5-word minimum, yielding 2-20s clips
# (CLAUDE.md "v7 SYNTHETIC DATASET", prep_marathi_corpus.py). Generating one
# SENTENCE at a time instead put ~4-8 word, ~2.5s units through a model that
# essentially never saw anything that short, which is a plausible driver of
# both the cross-sentence prosody drift AND the short-text hallucination that
# lib/text-segments.ts' mergeShortSentences was added to paper over.
#
# Packing also cuts the number of generation boundaries roughly in half, and
# drift can only occur AT a boundary: sentences inside one packed unit are a
# single autoregressive sequence, so their prosody stays self-consistent.
# DISABLED BY DEFAULT 2026-08-31 after a listening A/B said so.
#
# The theory was sound on paper: the v7 corpus was sentence-PACKED (24-word
# target, 2-20s clips averaging ~8.45s), so generating one ~4-8 word sentence
# at a time looked off-distribution, and fewer generation boundaries should
# mean less cross-sentence drift.
#
# The ear disagreed. In a same-text/same-seed A/B, per-sentence output
# (pack_words=0) was judged clearly better — more expressive and no worse for
# drift — while packed units were flatter, and packing measurably raised the
# retry rate by pushing units toward the top of the training length range
# where the model is least stable. Packing also bought nothing on speed
# (~26.5s vs ~27s for the same text).
#
# Kept as an option rather than deleted, because it's one request field away
# and worth re-testing if the content model starts producing much shorter
# sentences. But 0 is the default: generate one sentence at a time.
#
# RE-TESTED 2026-09-16 and CONFIRMED WORSE, this time with a reason to expect
# the opposite. Measuring the v7 training set showed production units sit well
# BELOW what the model was trained on (mr: median 20 words / 10.9s, p10 13
# words; production units are 4-13), which predicted that packing toward 20
# would help. It did not:
#     pack_words=0    49 units, median  8 words/unit,  2 retries (4%)
#     pack_words=20   27 units, median 15 words/unit,  9 retries (33%)
#   Same 8 story turns, same seed (648), temperature 0.65, RunPod.
# 4.5x the retry rate, and the only suspect unit in the set. Sentence boundaries
# don't divide evenly, so pack_words=20 actually lands near 15 — longer than what
# works, still short of the training median, and each unit now carries more text
# that a single alignment slip can damage. Distribution-matching from THIS side
# does not work; closing that gap would mean retraining with short clips.
PACK_TARGET_WORDS = 0
# LOWERED from 22 on 2026-09-02 after a trailing-hallucination report in Hindi.
# A/B on the SAME text and seed (42), warm worker, via the RunPod queue endpoint:
#   A  "...लग गया और वह और भी थोड़ा फट गया।"   21 words, 1 unit  -> 11.12s audio
#   B  "...लग गया। और वह और भी थोड़ा फट गया।"  same words, 2 units ->  9.52s audio
# A emitted 1.6s MORE audio for identical words, and the extra is an elongated
# final vowel ("...gayaaaaa"). Measured on the 50ms RMS envelope:
#   envelope flux over the last 1.3s : A 0.016  vs  B 0.184   (11x smoother)
#   monotonically-decaying tail frames: A 21/25 vs  B 13/25
# i.e. A ends in a long smooth fade (a sustained vowel) where B has the rapid
# phoneme-to-silence alternation of real speech. Whole-clip flux is depressed too
# (A 0.146 vs B 0.210), so the degradation builds through the generation rather
# than appearing only at the end — matching the mechanism described above.
#
# clean_edges() is NOT the fix: both clips trimmed an identical 0.42s after the
# last frame >=7% of peak. The elongated vowel is voiced and stays ABOVE that
# threshold, so the trimmer correctly leaves it alone. The cure is to stop
# generating it, by keeping units shorter via _split_overlong (which prefers
# comma boundaries before falling back to a hard word split).
#
# 22 let a 21-word sentence through by one word. 18, not 16: production logs
# (2026-09-02, 58 units) show only 22% of units exceed 16 words and their rtf
# (1.30) is no worse than shorter units (1.34) — so splitting more buys no speed,
# while every extra split adds a join boundary and a chance of prosody drift.
# 18 still catches the 21-word case that triggered this.
#
# MIRRORED CLIENT-SIDE in jaanteho-fresh lib/text-segments.ts (MAX_SENTENCE_WORDS,
# also 18). Whichever splits FIRST wins, so the client value governs app traffic
# and this one is the backstop for direct callers (ab_test.sh, curl). Keep in sync.
# Env-tunable so the cap can be A/B'd without an image rebuild. Its 18 rests on
# a SINGLE Hindi A/B that changed two things at once (shorter units AND an added
# sentence boundary), and it sits BELOW the training median: 67% of Marathi
# units and 52% of Hindi units exceed 18 words, the modal bucket being 18-23.
PACK_HARD_CAP_WORDS = int(os.environ.get("BETACRAFT_PACK_HARD_CAP") or 18)

# Bound the cost of a runaway generation. This used to be one flat number
# (1200 tokens, ~14s), which no longer works once units vary from 4 to 34
# words: 1200 is far too loose for a 4-word unit and actually too TIGHT for a
# 34-word one (~34/2.2 = 15.5s of legitimate speech ≈ 1350 tokens, i.e. the old
# ceiling would have truncated correct audio). So derive it per unit instead.
#
# Because the cap is set at exactly MAX_DURATION_MULTIPLIER x expected length,
# a runaway can no longer merely be *detected* after the fact — it is stopped
# at the boundary, and "hit the cap" IS the hallucination signal.
MAX_TOKENS_FLOOR = 400  # ~4.6s: headroom for the shortest units
# RAISED from 1800 (~20.7s) on 2026-08-31. Setting the ceiling at the longest
# TRAINING clip was wrong: it is a runaway backstop, not a length target, and
# at 1800 it bound *before* MAX_DURATION_MULTIPLIER for any unit over ~20
# words — collapsing the safety margin to ~1.2x expected exactly where it was
# needed most. Observed effect: 30-word units generating 20.81s of legitimate
# speech were truncated at 20.65s, marked as runaways, and reseeded. The
# ceiling must sit well ABOVE the longest legitimate unit; PACK_HARD_CAP_WORDS
# is what keeps units inside the training distribution.
MAX_TOKENS_CEILING = 2600  # ~29.9s
TOKEN_CAP_SUSPECT_RATIO = 0.97

# Digit -> word, language-aware. prep_marathi_corpus.py applied exactly this to
# every training transcript, so raw digits are OFF-DISTRIBUTION at inference —
# and LLM-written story text does contain them. Same single-digit scope as
# training (multi-digit runs were avoided at the story-generation prompt level,
# per CLAUDE.md "numbers-as-words"); those are warned about instead.
DIGIT_MR = {"0": "शून्य", "1": "एक", "2": "दोन", "3": "तीन", "4": "चार",
            "5": "पाच", "6": "सहा", "7": "सात", "8": "आठ", "9": "नऊ"}
DIGIT_HI = {"0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
            "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ"}
for _d, _a in zip("०१२३४५६७८९", "0123456789"):  # Devanagari digits map same as ASCII
    DIGIT_MR[_d] = DIGIT_MR[_a]
    DIGIT_HI[_d] = DIGIT_HI[_a]
DIGIT_MAPS = {"mr": DIGIT_MR, "hi": DIGIT_HI, "en": DIGIT_MR}

_SINGLE_DIGIT_RE = re.compile(r"(?<!\d)[0-9०-९](?!\d)")
_MULTI_DIGIT_RE = re.compile(r"[0-9०-९]{2,}")

# Loudness matching. Perceived "drift" between units is partly just energy
# variation, which is cheap to remove without touching the model. Deliberately
# gentle: units are nudged toward the median unit's RMS, and no unit is moved
# more than this many dB, so genuine dynamics (a whisper, a shout) survive
# while gross level jumps between adjacent units do not.
LOUDNESS_MAX_GAIN_DB = 3.0


def _env_opt_float(name: str, default: float | None) -> float | None:
    """Read an optional float tuning knob from the environment.

    Explicitly supports DISABLING a sampling parameter via "none"/"off"/"" so
    the two suspect logits processors below can be A/B tested by flipping an
    endpoint env var, with no image rebuild.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in ("", "none", "off", "null"):
        return None
    return float(raw)


# --- Sampling defaults -------------------------------------------------------
# repetition_penalty and no_repeat_ngram_size are DISABLED by default as of
# 2026-08-31. They were added to suppress the looping/"singing" hallucination.
#
# CONFIRMED ON A5000, 2026-08-31 (scripts/narrate/ab_test.sh, same text/seed):
# turning them back on made the output AUDIBLY HALLUCINATED while the default
# (both off) was clean — i.e. they were manufacturing the very defect they were
# added to prevent — and cost 28% throughput on top (46 tok/s vs 64 tok/s;
# run-to-run noise measured at +/-5%, so the gap is real). The slowdown appears
# only on LONG units (25w: rtf 1.44 -> 1.88; 7w: no measurable change), which
# matches the mechanism below.
#
# They are text-generation heuristics applied to DAC *audio* tokens, and both
# do net harm here:
#
#   * repetition_penalty penalises every codebook entry already used in the
#     sequence. Over a 400-1200 token generation a large share of the 1024-entry
#     codebook gets used, so the penalty progressively pushes the model toward
#     tokens it hasn't used yet — i.e. increasingly wrong ones — the FURTHER
#     INTO a generation it gets. That matches the reported symptoms closely:
#     degradation toward the end of sentences, trailing artefacts, and markedly
#     worse behaviour on long stories.
#   * no_repeat_ngram_size bans repeated 3-grams, but silence and sustained
#     vowels legitimately repeat in audio-token space. It also costs real speed:
#     HF's NoRepeatNGramLogitsProcessor rebuilds its n-gram map in Python on
#     every decode step, across all 9 codebook rows, stalling the GPU each step.
#
# They were never A/B'd against a clean baseline, so rather than deleting them
# outright they're kept reachable and switchable at runtime (env var, or a
# per-request override) — set BETACRAFT_REPETITION_PENALTY=1.2 and
# BETACRAFT_NO_REPEAT_NGRAM=3 to restore the old behaviour for comparison.
# TEMPERATURE BRACKETED 2026-09-16 — 0.65 is a measured optimum, not a guess.
# Both directions were tried on real story text and both are WORSE:
#   1.0 (library default) — prosody varies sentence to sentence, as if the
#                           narrator changed between units. Judged by ear.
#   0.65                  — kept.
#   0.5                   — LOUD BUZZING on staging, in both mr and hi, across
#                           two story sessions. No buzz was reported at 0.65.
#   ~0 (greedy)           — 2026-09-08 smoke test: never emitted a stop token,
#                           621 tokens, ~90% of it stripped by clean_edges.
# So do not "just lower it" to chase dropped words. Under-constrained sampling
# loses voice consistency; over-constrained sampling degenerates into held
# vowels and buzz. This value sits in the trough between the two.
DEFAULT_TEMPERATURE = _env_opt_float("BETACRAFT_TEMPERATURE", 0.65) or 0.65
# TOP_P: the fallback below is 0.9, but PRODUCTION RUNS 1.0 via BETACRAFT_TOP_P
# on the RunPod endpoint. lib/tts.ts deliberately does NOT send top_p (see the
# note above its request payload), so this env var is what actually decides it.
#
# MEASURED 2026-09-16 on RunPod, temperature held at 0.65:
#   * mr, "...आवाज आला धूम!", seed 648: at 0.9 the end of the sentence was
#     DROPPED (0.52x expected length, confirmed by ear); at 1.0, SAME seed, the
#     whole sentence was spoken (1.10x).
#   * hi, full 6-sentence turn, 5 seeds: worst single-unit length ratio fell
#     from 1.46x to 1.15x — better on 5 of 5 seeds.
#   * mr, 5 known-bad sentences x 5 seeds: better on 3, unchanged on 1, slightly
#     worse on 1.
# MECHANISM: 0.9 keeps only the top 90% of probability mass at each step. Where
# this fine-tune is undertrained — short units, which sit BELOW the 10th
# percentile of its own training clips (mr median 20 words, p10 13) — that
# distribution is diffuse, so the trim can discard the correct continuation and
# leave the decoder choosing among wrong ones. Heard as a held vowel, buzz, or a
# skipped span. Same family as the temperature findings above.
# NOT A FIX, A RATE REDUCTION: defects still occur at 1.0, just less often.
# Raising this default to 1.0 would make a fresh endpoint safe without the env
# var, but changes behaviour on the next rebuild — decide that deliberately.
DEFAULT_TOP_P = _env_opt_float("BETACRAFT_TOP_P", 0.9)

# Seed. set_seed() is re-applied before EVERY unit (see synthesize), so this is
# the voice anchor for a whole narration, not just a randomness source: change
# it and the timbre changes. 648 was picked from the local seed sweep on
# 2026-09-04 (local_repro/20260904-*/report.json). Env-tunable so a different
# anchor can be tried from the RunPod console without an image rebuild.
DEFAULT_SEED = int(os.environ.get("BETACRAFT_SEED") or 648)

# The BLIND SPOT between "normal" and "runaway", closed 2026-09-07.
#
# MAX_DURATION_MULTIPLIER (2.5x) is a hard token cap: it TRUNCATES. That makes it
# a backstop for catastrophes only, and everything below it shipped unchecked —
# which is where the real damage was. MEASURED, English, seed 648, 10 words
# (quality_runs/20260907-seed648-check): 10.53s against 4.81s expected = 2.19x,
# a stalled countdown that never tripped the 14.29s cap and so was never retried.
# Seed 42 on the SAME sentence ran 14.71s, hit the cap, and WAS retried — i.e.
# the worse-sounding clip was handled and the merely-bad one was not.
#
# So flag "too long" independently of the cap: a unit past this multiple of its
# expected length is SUSPECT (reseed and retry, keeping the better take) but is
# never truncated, which is what separates this from the min_new_tokens floor
# rejected on 2026-09-04 — that one FORCED generation to continue and so
# manufactured trailing stalls. This only rejects a take that already exists.
#
# 1.8 clears the longest legitimate output observed (Marathi 1.68x in
# local_repro/20260904-152523) while catching the 2.19x case above. Env-tunable
# because it is a calibration threshold: it will need moving once the probe set
# is scored by ear, and that must not require an image rebuild.
#
# MEASURED 2026-09-16 over 113 Marathi production units (cap-hits excluded):
# median 1.00x, p90 1.22x, MAX 1.60x. So at 1.8 this detector fired ZERO times —
# it is switched on and catching nothing; every Marathi flag in those logs came
# from the 2.5x hard cap instead. Lowering it would cost retries:
#     1.8 -> 0/113 (0%)     1.37 -> 4/113 (4%)     1.24 -> 11/113 (10%)
# 1.35 would have caught the one reported unit that ran long ("अचानक रॉकेट...",
# 1.37x) for ~4% more retries. NOT ATTEMPTED YET — and note it only ever sees
# units that run LONG: the other reported failure that turn was 1.07x, and the
# khoop stall was mid-clip, which TAIL_ANALYSIS_SEC (1.3s, end only) cannot see.
SOFT_DURATION_MULTIPLIER = _env_opt_float("BETACRAFT_SOFT_DURATION_MULT", 1.8) or 1.8

DEFAULT_REPETITION_PENALTY = _env_opt_float("BETACRAFT_REPETITION_PENALTY", None)
DEFAULT_NO_REPEAT_NGRAM = _env_opt_float("BETACRAFT_NO_REPEAT_NGRAM", None)


# --- Enumeration punctuation ------------------------------------------------
# MEASURED 2026-09-08, same seed (648), same temperature, one variable
# (local_repro/20260908-163739 vs -164351 vs -172322, all ear-scored):
#
#     "…केली, तीन, दोन, एक, झूम!"    1.51x expected — hallucinates at the commas
#     "…केली... तीन... दोन... एक..." 2.57x, tail flux 0.0093 — WORSE, breaks even earlier
#     "…केली तीन दोन एक झूम!"        1.01x — CORRECT
#
# Why: a comma is a place to pause, and this model has no mechanism forcing it to
# cover the text before stopping — so a run of one-word fragments gives it several
# consecutive chances to mistake a pause for an ending. It then either stops (the
# words are never spoken) or fails to stop (a held vowel). Ellipsis is worse still:
# it reads as "trail off", and the model obliges indefinitely.
#
# TRAINING FREQUENCY is what decides the scope. Of 1,772 Marathi units:
#     comma-separated counting            0      <- what we generate, never seen
#     runs of >=2 short comma fragments   47     (2.65%)
#     a SINGLE short comma fragment      101     (5.7%, mostly "आई," / "मुलांनो,")
# So only RUNS are collapsed. A lone "ढप्प," or a form of address is left alone —
# stripping those would move AWAY from the training distribution, not toward it.
_ENUM_SEP_RE = re.compile(r"(\s*(?:\.\.\.|[,\u060C])\s*)")
ENUM_SHORT_WORDS = 2      # a fragment this short or shorter counts toward a run
ENUM_MIN_RUN = 2          # this many consecutive short fragments makes it a run


def normalize_enumerations(text: str) -> str:
    """Collapse runs of short comma/ellipsis-fenced fragments into plain words."""
    parts = _ENUM_SEP_RE.split(text)
    if len(parts) < 3:
        return text
    frags, seps = parts[0::2], parts[1::2]
    short = [0 < len(f.split()) <= ENUM_SHORT_WORDS for f in frags]

    drop = set()                                  # indices of separators to replace
    i = 0
    while i < len(frags):
        if not short[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(frags) and short[j + 1]:
            j += 1
        # A run of the SAME word repeated ("हा... हा... हा...", "घोर... घोर...")
        # IS in the corpus — onomatopoeia and laughter are written exactly that
        # way — so leave it alone. Only runs of DISTINCT short fragments are the
        # unseen pattern: comma-separated counting appears 0 times in 3,283 units.
        # Strip quotes too: a run inside dialogue makes the first fragment '"हा'
        # and the rest 'हा', which would compare as different words and defeat the
        # exemption. Empty fragments (a lone closing quote after the last "...")
        # are dropped rather than counted as a distinct word.
        _EDGE = " \u0964.!?\u0965\"'\u201c\u201d\u2018\u2019-"
        run_frags = [f.strip(_EDGE) for f in frags[i:j + 1]]
        run_frags = [f for f in run_frags if f]
        repeated = len(set(run_frags)) <= 1
        if j - i + 1 >= ENUM_MIN_RUN and not repeated:
            # Separators INSIDE the run, plus the ones on BOTH sides joining it to
            # its neighbours. Dropping only the leading one leaves a comma sitting
            # immediately after the enumeration — precisely where the model is most
            # likely to mistake a pause for an ending — and would not reproduce the
            # text that was ear-confirmed correct (local_repro/20260908-164351).
            drop.update(range(i, j))
            if i > 0:
                drop.add(i - 1)
            if j < len(seps):
                drop.add(j)
        i = j + 1
    if not drop:
        return text

    out = [frags[0]]
    for k, sep in enumerate(seps):
        out.append(" " if k in drop else sep)
        out.append(frags[k + 1])
    return "".join(out)


def normalize_digits(text: str, language: str) -> str:
    """Spell single digits out, matching what training transcripts contain."""
    multi = _MULTI_DIGIT_RE.findall(text)
    if multi:
        log.warning(
            "Text contains multi-digit number(s) %s — training data had these "
            "spelled out, so pronunciation here is off-distribution.", multi[:5],
        )
    dmap = DIGIT_MAPS.get(language, DIGIT_MR)
    return _SINGLE_DIGIT_RE.sub(lambda m: dmap.get(m.group(), m.group()), text)


# A standalone 1-3 word sentence ("पप पप!", "फुस्स फुस्स!", "धपाप!") is a known
# Parler failure mode: too little text to predict a clean stop, so it emits a
# runaway — often a LOUD hallucinated tail that clean_edges cannot trim (its
# docstring: "a tail as loud as real speech won't be caught by this").
#
# lib/text-segments.ts does this merge client-side, but the server re-splits
# whatever text it receives, so it cannot rely on that having happened — and
# anything hitting /synthesize directly (ab_test.sh, curl, a future caller)
# bypasses the client entirely. Enforce the floor here too.
MIN_UNIT_WORDS = 4


def _merge_short(sentences: list[str], min_words: int = MIN_UNIT_WORDS) -> list[str]:
    """Fold too-short sentences into a neighbour so none is generated alone.

    Attaches to the PREVIOUS sentence where possible (reads more naturally than
    leading the next one); a short fragment at the very start is held and
    prefixed onto the following sentence instead.
    """
    merged: list[str] = []
    pending: str | None = None
    for sentence in sentences:
        if pending:
            # Join with a SPACE, keeping the fragment's own punctuation. This used
            # to strip the ender and insert ", ", which turned ordinary short
            # sentences ("ढप्प. तो घाबरला. सगळे हसले.") into exactly the comma run
            # that breaks generation. Training units were sentence-PACKED with
            # their punctuation intact, so preserving it matches the corpus.
            sentence = f"{pending} {sentence}"
            pending = None
        if len(sentence.split()) < min_words:
            if merged:
                # Keep the previous sentence's own ending punctuation and join with
                # a SPACE. Stripping the ender and inserting ", " turned ordinary
                # short sentences into the comma run that measurably breaks
                # generation — and children's stories are full of short sentences,
                # so this fired constantly. Training units were sentence-PACKED
                # with punctuation intact, which is exactly what this now produces.
                merged[-1] = f"{merged[-1]} {sentence}"
            else:
                pending = sentence
        else:
            merged.append(sentence)
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]} {pending}"
        else:
            merged.append(pending)  # whole input was one short fragment
    return merged


# Clause conjunctions used as fallback break points when a long sentence has no
# comma. These carry the same prosodic role a comma does — a natural pause the
# narrator would take — so breaking BEFORE one yields two units that each stand
# alone as speech, instead of the mid-clause cut a blind word-count split gives.
#
# DELIBERATELY EXCLUDES पर and तो, which are ambiguous in Hindi: पर is far more
# often the postposition "on" (नक्शे पर = "on the map") than the conjunction
# "but", and तो is usually emphatic rather than clausal. Including पर was tried
# on 2026-09-02 and split "...फटे हुए नक्शे | पर लग गया..." straight through a
# noun phrase. When in doubt leave a word out: a missed split falls through to
# the hard-split warning path, whereas a wrong split is silently unnatural.
CLAUSE_CONJUNCTIONS = ("और", "लेकिन", "क्योंकि", "जब", "तब", "फिर", "या", "इसलिए")
# Both sides of a conjunction break must be at least this many words, so we never
# strand a 1-2 word fragment (which _merge_short would then fold back anyway).
CONJ_MIN_SIDE_WORDS = 4


def _split_at_conjunction(text: str, hard_cap: int, _depth: int = 0) -> list[str] | None:
    """Break `text` before a clause conjunction, nearest the midpoint.

    Returns None when there is no usable conjunction, so the caller can fall
    back to a hard word split. Recurses (bounded) so a very long sentence with
    several conjunctions is reduced until every piece fits under hard_cap.

    The left piece is closed with the SOURCE sentence's own terminal punctuation,
    never a hardcoded one. An earlier version appended a Devanagari danda "।"
    unconditionally, which is right for Hindi (1422 of 1511 training units end in
    one) but WRONG for Marathi: the danda occurs ZERO times in all 1772 Marathi
    training units, which end in a plain period. That fed the model a character it
    had never seen in training, on exactly the path meant to make long sentences
    safer. Reusing the writer's own ender cannot drift out of distribution.
    """
    words = text.split()
    candidates = [
        i for i, w in enumerate(words)
        if w in CLAUSE_CONJUNCTIONS
        and CONJ_MIN_SIDE_WORDS <= i <= len(words) - CONJ_MIN_SIDE_WORDS
    ]
    if not candidates:
        return None
    i = min(candidates, key=lambda i: abs(i - len(words) / 2))
    ender = text.rstrip()[-1] if text.rstrip()[-1:] in "।.!?" else "."
    left = " ".join(words[:i]).rstrip("।.!?,").rstrip() + ender
    right = " ".join(words[i:])

    out: list[str] = []
    for piece in (left, right):
        if len(piece.split()) > hard_cap and _depth < 3:
            deeper = _split_at_conjunction(piece, hard_cap, _depth + 1)
            if deeper is not None:
                out.extend(deeper)
                continue
        out.append(piece)
    return out


def _split_overlong(sentence: str, hard_cap: int) -> list[str]:
    """Break a single sentence that exceeds hard_cap into speakable pieces.

    Needed because the per-unit token budget tops out at MAX_TOKENS_CEILING
    (~20.7s, the longest training clip). A sentence needing more than that
    would be TRUNCATED mid-speech, silently losing text — worse than an extra
    pause. Prefer comma boundaries (already prosodic pause points, and
    preserved in the training transcripts); fall back to a hard word split only
    when there are no commas to use.
    """
    words = sentence.split()
    if len(words) <= hard_cap:
        return [sentence]

    pieces, current = [], []
    for part in (p.strip() for p in sentence.split(",")):
        if not part:
            continue
        if current and len(" ".join(current).split()) + len(part.split()) > hard_cap:
            pieces.append(", ".join(current))
            current = []
        current.append(part)
    if current:
        pieces.append(", ".join(current))

    out: list[str] = []
    for piece in pieces:
        piece_words = piece.split()
        if len(piece_words) <= hard_cap:
            out.append(piece)
            continue
        conj = _split_at_conjunction(piece, hard_cap)
        if conj is not None:
            out.extend(conj)
            continue
        log.warning(
            "Sentence fragment of %d words has no comma or clause conjunction to break "
            "on — hard-splitting at %d words, which may land mid-clause.",
            len(piece_words), hard_cap,
        )
        out.extend(
            " ".join(piece_words[i : i + hard_cap]) for i in range(0, len(piece_words), hard_cap)
        )
    return out


def pack_sentences(
    sentences: list[str],
    target_words: int = PACK_TARGET_WORDS,
    hard_cap: int = PACK_HARD_CAP_WORDS,
) -> list[str]:
    """Greedily group sentences into ~target_words units, never exceeding hard_cap.

    Mirrors the sentence-packing that built the training corpus. Sentences
    longer than hard_cap on their own are broken up first (see _split_overlong)
    so no unit can exceed the token budget and get truncated.
    """
    # Over-long sentences are split regardless of whether GROUPING is enabled:
    # a single sentence longer than hard_cap would blow past the per-unit token
    # budget and get truncated mid-speech. That protection is independent of
    # packing and must not be skipped when target_words is 0.
    expanded: list[str] = []
    for sentence in _merge_short(sentences):
        expanded.extend(_split_overlong(sentence, hard_cap))

    if target_words <= 0:  # grouping disabled -> one unit per sentence
        return expanded

    units: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in expanded:
        words = len(sentence.split())
        # Close BEFORE appending, not after. The previous version appended and
        # then checked `>= target`, so the last sentence always overshot — a
        # 24-word target produced observed 30-31 word units. Closing first
        # bounds every unit at hard_cap.
        #
        # Two independent reasons to close: the unit has already reached its
        # target, or adding this sentence would exceed the hard cap. Keeping
        # both means short sentences still accumulate toward the target
        # (rather than each becoming its own undersized unit) while no unit
        # can ever run past hard_cap.
        if current and (current_words >= target_words or current_words + words > hard_cap):
            units.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += words
    if current:
        units.append(" ".join(current))
    return units


# --- Trailing-stall detection ------------------------------------------------
# The failure the token cap CANNOT see. A unit whose last word decays into a held
# vowel ("...फट गया" -> "gayaaaaa", "the moon" -> "moonnnnnn", "locking" ->
# "lockinggggg") adds only ~1-2s, far inside MAX_DURATION_MULTIPLIER, so it never
# trips the runaway check. clean_edges() also leaves it alone by design: the vowel
# is VOICED and sits above its energy floor ("a tail as loud as real speech won't
# be caught by this").
#
# MEASURED 2026-09-02, same text and seed, 50ms RMS envelope over the last ~1.3s
# of signal:
#     stalled tail : flux 0.016, 21/25 frames decaying monotonically
#     clean speech : flux 0.184, 13/25
# an 11x separation. Real speech alternates rapidly between phonemes and silence;
# a held vowel is a smooth monotonic fade. Flux (mean absolute frame-to-frame
# change) is what tells them apart, and it is independent of total duration, which
# is why it catches what every duration-based guard misses.
#
# OBSERVED 2026-09-03 in all three languages (mr/hi/en) and almost always on the
# FINAL word of a sentence — i.e. the model failing to emit a clean EOS.
TAIL_ANALYSIS_SEC = 1.3      # how much of the end to score
TAIL_FRAME_SEC = 0.05        # RMS frame size the threshold was measured with
# CALIBRATED BY LISTENING 2026-09-03. Every clip to hand, scored by ear:
#     0.017  STALLED  ("...gayaaaa", confirmed)
#     0.043  clean    <- 0.06 flagged this one; listening says it lands cleanly
#     0.096  clean
#     0.121  clean
#     0.163  clean    (confirmed)
#     0.168  clean
# 0.03 sits between the one confirmed stall and the lowest confirmed-clean take,
# and classifies all six correctly. Deliberately biased toward KEEPING audio: a
# false positive forces a reseeded retry, and a seed change alters the voice
# realisation for that unit alone — an audible prosody jump against its
# neighbours (see the retry note below). A missed stall is only as bad as today.
#
# THIN CALIBRATION: one confirmed stalled sample. Re-tune as more are collected —
# `stalled_tail` in the per-unit metadata makes them easy to gather from logs.
TAIL_FLUX_SUSPECT = 0.03
MIN_SEC_FOR_TAIL_CHECK = 2.0 # shorter units have too little tail to score


def _tail_flux(wav: np.ndarray, sr: int) -> float | None:
    """Mean absolute frame-to-frame change of the RMS envelope over the unit's
    final TAIL_ANALYSIS_SEC of NON-SILENT audio.

    Returns None when the unit is too short to judge. Trailing near-silence is
    skipped first so the score reflects the last of the *speech*, not the pad.
    """
    if wav.size < int(sr * MIN_SEC_FOR_TAIL_CHECK):
        return None
    hop = max(1, int(sr * TAIL_FRAME_SEC))
    frames = np.array([
        float(np.sqrt(np.mean(np.square(wav[i:i + hop]))))
        for i in range(0, wav.size - hop, hop)
    ])
    if frames.size < 4:
        return None
    peak = float(frames.max())
    if peak <= 0:
        return None
    norm = frames / peak
    voiced = np.flatnonzero(norm >= 0.02)      # drop the trailing pad
    if voiced.size < 4:
        return None
    end = int(voiced[-1]) + 1
    start = max(0, end - int(TAIL_ANALYSIS_SEC / TAIL_FRAME_SEC))
    tail = norm[start:end]
    if tail.size < 4:
        return None
    return float(np.mean(np.abs(np.diff(tail))))


def _expected_sec_for(word_count: int, language: str) -> float:
    """How long `word_count` words SHOULD take to speak in `language`."""
    rate = WORDS_PER_SEC_BY_LANG.get(language, EXPECTED_WORDS_PER_SEC)
    return max(word_count, 1) / rate


def _max_new_tokens_for(word_count: int, sr: int, hop_length: int, language: str = "mr") -> int:
    """Token budget for one unit: MAX_DURATION_MULTIPLIER x its expected length."""
    expected_sec = _expected_sec_for(word_count, language)
    tokens = expected_sec * MAX_DURATION_MULTIPLIER * sr / hop_length
    return int(min(MAX_TOKENS_CEILING, max(MAX_TOKENS_FLOOR, tokens)))


# TRIED AND REJECTED 2026-09-04: a `min_new_tokens` FLOOR (forbid EOS before ~0.6x
# the unit's expected duration), intended to stop the model dropping a unit's last
# words. Rejected for two reasons, both from measurement:
#
#   1. IT WOULD MANUFACTURE THE OPPOSITE DEFECT. A floor forces generation to
#      continue past a genuine ending. _merge_short normally keeps units at
#      MIN_UNIT_WORDS or more, but a hard-split fragment or a direct API call can
#      still produce a 2-3 word unit, and forcing those to keep generating is
#      precisely how a trailing hallucination is created. Trading a skip for a
#      stall is not a fix.
#   2. THE CALIBRATION WAS UNSAFE FOR ENGLISH. Production logs (2026-09-04, 75
#      units) show English narrates at ~2.08 words/sec against the 1.75 assumed by
#      EXPECTED_WORDS_PER_SEC, so normal English lands at 0.68-0.84x "expected".
#      A 0.6 floor leaves ~12% headroom on legitimate fast English.
#
# It also would not have caught the reported cases: dropping the final 2 of 11
# words still yields 82% of expected duration, far above any floor safe enough to
# ship. The floor only caught severe truncation (one unit lost 78% of its text).
#
# The real finding from that log is separate and still open: EXPECTED_WORDS_PER_SEC
# is MARATHI-calibrated, so the English token CEILING is ~40% too generous — which
# is why English runaways reached 15.8s and 27.4s before the cap stopped them.
# Making that rate language-aware is the change worth making here.


def match_loudness(units: list[np.ndarray], max_gain_db: float = LOUDNESS_MAX_GAIN_DB) -> list[np.ndarray]:
    """Nudge each unit toward the median unit's RMS, capped at +/- max_gain_db.

    Targets the MEDIAN rather than the mean so one hallucinated or unusually
    loud unit can't drag the whole story's level with it.
    """
    if len(units) < 2 or max_gain_db <= 0:
        return units
    rms = np.array([float(np.sqrt(np.mean(u.astype(np.float64) ** 2))) for u in units])
    usable = rms > 1e-6
    if usable.sum() < 2:
        return units
    target = float(np.median(rms[usable]))
    limit = 10.0 ** (max_gain_db / 20.0)
    out = []
    for unit, unit_rms in zip(units, rms):
        if unit_rms <= 1e-6:
            out.append(unit)
            continue
        gain = float(np.clip(target / unit_rms, 1.0 / limit, limit))
        adjusted = unit * gain
        # Guard against clipping introduced by a boost.
        peak = float(np.max(np.abs(adjusted))) if adjusted.size else 0.0
        if peak > 1.0:
            adjusted = adjusted / peak
        out.append(adjusted.astype(np.float32))
    return out


def _pick_device() -> str:
    if DEVICE:
        return DEVICE
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype | None:
    # bf16 matches training precision (CLAUDE.md: bf16=true during training) and
    # runs ~2x faster on Ampere+ CUDA tensor cores. MPS's bf16 support is less
    # mature, so keep float32 there — this only changes CUDA behaviour.
    return torch.bfloat16 if device.startswith("cuda") else None


def _load_model(model_dir: str, device: str):
    """Load a model, preferring Flash Attention 2 but falling back to the
    default attention implementation if it isn't installed/available — a
    missing/broken flash-attn package must not crash the whole worker."""
    dtype = _dtype_for_device(device)
    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype, attn_implementation="flash_attention_2",
        ).to(device)
        log.info("Loaded %s with attn_implementation=flash_attention_2", model_dir)
    except Exception as exc:
        log.warning("flash_attention_2 unavailable (%s) — falling back to default attention", exc)
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype).to(device)
    return model


def _warmup(bundle: dict) -> None:
    """Run one tiny throwaway generation right after load. Loading the weights
    only gets them into memory — the FIRST model.generate() call also pays a
    one-time kernel-compile + memory-pool-allocation tax on top of that. Running
    it here means that tax lands on cold start (where RunPod's own worker-init
    accounting expects it) rather than on the first real user's request."""
    start = time.time()
    lang = next(iter(bundle["desc_ids_by_lang"]))
    desc_ids = bundle["desc_ids_by_lang"][lang]
    desc_mask = bundle["desc_mask_by_lang"][lang]
    # Mirror the real generate() signature exactly: this call exists to pay the
    # kernel-compile tax on cold start, and a different signature can compile a
    # different path, leaving the tax for the first real request after all.
    prompt_enc = bundle["prompt_tok"]("नमस्ते", return_tensors="pt").to(bundle["device"])
    with torch.no_grad():
        bundle["model"].generate(
            input_ids=desc_ids, attention_mask=desc_mask,
            prompt_input_ids=prompt_enc.input_ids,
            prompt_attention_mask=prompt_enc.attention_mask,
            max_new_tokens=32, do_sample=True, temperature=0.7,
        )
    log.info("Warmup generation (lang=%s) done in %.1fs", lang, time.time() - start)


def load_bundle(model_name: str = DEFAULT_MODEL) -> dict:
    """Load one model + its tokenizers + pre-tokenized captions, then warm it up.

    See get_bundle() for the caching wrapper callers should normally use.
    """
    if model_name not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    spec = MODELS[model_name]
    model_dir = spec["dir"]

    device = _pick_device()
    log.info("Loading model '%s' from %s on %s", model_name, model_dir, device)
    load_start = time.time()
    model = _load_model(model_dir, device)
    model.eval()
    prompt_tok = AutoTokenizer.from_pretrained(model_dir)
    if prompt_tok.pad_token is None:
        prompt_tok.pad_token = prompt_tok.eos_token
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    # Same caption wording for both models, only the speaker name differs. For
    # "betacraft" this string is load-bearing (it must match the training
    # caption byte-for-byte); for "base" it is simply a good description in the
    # style the Indic Parler model card recommends, kept identical so an A/B
    # between the two isolates the fine-tuning rather than the prompt.
    # Tokenize once per language and keep the ATTENTION MASK too. Every unit is
    # a single unpadded sequence, so the mask is all-ones and passing it changes
    # nothing numerically today — but HF cannot INFER that (pad_token == eos_token
    # here, which is what the "attention mask is not set" warning is about), so it
    # falls back to a default and warns on every generate. Passing it explicitly
    # silences that and, more importantly, makes the call correct by construction
    # if units are ever batched again: batching pads, and an inferred mask over
    # pad-that-equals-eos is exactly how padded sequences produced the audible
    # trailing buzz that got batching rolled back on 2026-08-29.
    desc_enc_by_lang = {
        lang: desc_tok(CAPTION_TEMPLATE.format(name=name), return_tensors="pt").to(device)
        for lang, name in spec["speakers"].items()
    }
    desc_ids_by_lang = {lang: enc.input_ids for lang, enc in desc_enc_by_lang.items()}
    desc_mask_by_lang = {lang: enc.attention_mask for lang, enc in desc_enc_by_lang.items()}
    # Samples of audio per DAC token. Used to convert an audio length back into
    # a token count, so we can tell whether generation stopped on EOS or was cut
    # off at the max_new_tokens ceiling (see TOKEN_CAP_SUSPECT_RATIO).
    audio_cfg = model.audio_encoder.config
    hop_length = max(1, int(audio_cfg.sampling_rate // audio_cfg.frame_rate))

    bundle = {
        "name": model_name,
        "model": model,
        "prompt_tok": prompt_tok,
        "desc_ids_by_lang": desc_ids_by_lang,
        "desc_mask_by_lang": desc_mask_by_lang,
        "speakers": spec["speakers"],
        "sr": model.config.sampling_rate,
        "device": device,
        "hop_length": hop_length,
    }
    _warmup(bundle)
    log.info(
        "Model '%s' ready in %.1fs (sr=%d, hop=%d, device=%s)",
        model_name, time.time() - load_start, bundle["sr"], hop_length, device,
    )
    return bundle


# Loaded models, keyed by name. Populated LAZILY rather than at startup: cold
# start is the dominant latency problem on RunPod, and eagerly loading both
# models would roughly double it for every worker — including the majority of
# workers that only ever serve the default model. The cost is that the first
# request for the *other* model pays its load time once per worker; a warmup
# job can pre-pay that deliberately (rp_handler accepts {"warmup": true,
# "model": "base"}).
_BUNDLES: dict[str, dict] = {}


def get_bundle(model_name: str = DEFAULT_MODEL) -> dict:
    """Return a loaded bundle for `model_name`, loading it on first use."""
    if model_name not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    if model_name not in _BUNDLES:
        _BUNDLES[model_name] = load_bundle(model_name)
    return _BUNDLES[model_name]


def build_gen_kwargs(
    temperature: float | None = None,
    top_p: float | None = -1.0,
    repetition_penalty: float | None = -1.0,
    no_repeat_ngram_size: float | None = -1.0,
    do_sample: bool | None = None,
) -> dict:
    """Assemble generate() kwargs, letting a caller override any sampling knob.

    Sentinel -1.0 means "not specified by the caller, use the configured
    default"; an explicit None means "disable this parameter entirely". That
    distinction is what allows a single request to turn repetition_penalty /
    no_repeat_ngram_size back on (or off) for an A/B without a redeploy.
    """
    top_p = DEFAULT_TOP_P if top_p == -1.0 else top_p
    repetition_penalty = DEFAULT_REPETITION_PENALTY if repetition_penalty == -1.0 else repetition_penalty
    no_repeat_ngram_size = DEFAULT_NO_REPEAT_NGRAM if no_repeat_ngram_size == -1.0 else no_repeat_ngram_size

    # GREEDY (do_sample=False) exists to test the dropped-word defect at its
    # mechanism. The model stops when it emits an end-of-audio token, and with
    # sampling on, that token only needs a small probability to be drawn early —
    # which is why the SAME sentence drops words at one seed and reads cleanly at
    # another. Greedy takes the argmax instead, so an early stop can only happen
    # if it is genuinely the most likely continuation. Output becomes fully
    # deterministic: seed and temperature stop having any effect.
    #
    # Not a free win — expect flatter prosody, which matters for children's
    # narration. It is a trade to measure, not a fix to assume.
    sampling = True if do_sample is None else bool(do_sample)
    kwargs: dict = {"do_sample": sampling}
    if not sampling:
        # temperature/top_p are sampling-only; passing them with greedy decoding
        # makes transformers warn about unused generation flags on every call.
        return kwargs
    kwargs["temperature"] = temperature if temperature is not None else DEFAULT_TEMPERATURE
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = float(repetition_penalty)
    if no_repeat_ngram_size is not None:
        kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    return kwargs


def synthesize(
    bundle: dict,
    text: str,
    language: str = "mr",
    temperature: float | None = None,
    seed: int = DEFAULT_SEED,
    gap_ms: int = 350,
    gen_overrides: dict | None = None,
    pack_words: int = PACK_TARGET_WORDS,
    match_loudness_db: float = LOUDNESS_MAX_GAIN_DB,
) -> tuple[np.ndarray, int, list[dict]]:
    """Generate narration for `text`, one packed unit at a time.

    Returns (waveform float32 mono, sample_rate, per_unit_metadata).

    Units are sentence-PACKED to ~pack_words words (see PACK_TARGET_WORDS) so
    they resemble the 2-20s clips the model was fine-tuned on, rather than the
    ~4-8 word single sentences it was previously asked for. Pass pack_words=0
    to fall back to one unit per sentence for an A/B.

    Batched multi-unit generation (several units in ONE generate() call) was
    tried 2026-08-29 and rolled back: sequences decode in LOCKSTEP, so one slow
    outlier drags the rest down (a 3-chunk batch measured 80s instead of
    ~16-20s), and padding shorter sequences produced audible trailing buzz that
    clean_edges does not catch (its own docstring: "a tail as loud as real
    speech won't be caught by this"). Note that packing is NOT that: a packed
    unit is a single unpadded sequence, so it has neither failure mode.
    Parallelism happens ACROSS RunPod workers instead.
    """
    # Validate against THIS bundle's speakers, not the global superset: the two
    # models don't necessarily cover the same languages, and routing a request
    # to a model that has no speaker for it would KeyError below.
    speakers = bundle.get("speakers", LANGUAGE_SPEAKERS)
    if language not in speakers:
        raise ValueError(
            f"model '{bundle.get('name', '?')}' supports {list(speakers)}, not '{language}'"
        )

    sentences = split_sentences(normalize_enumerations(normalize_digits(text, language)))
    if not sentences:
        raise ValueError("no speakable text after cleaning")
    chunks = pack_sentences(sentences, target_words=pack_words)

    model = bundle["model"]
    prompt_tok = bundle["prompt_tok"]
    device = bundle["device"]
    sr = bundle["sr"]
    hop_length = bundle["hop_length"]
    desc_ids = bundle["desc_ids_by_lang"][language]
    desc_mask = bundle["desc_mask_by_lang"][language]

    gen_kwargs = build_gen_kwargs(temperature=temperature, **(gen_overrides or {}))
    log.info(
        "Generating %d unit(s) from %d sentence(s), model=%s, lang=%s, speaker=%s, "
        "pack_words=%d, gen_kwargs=%s",
        len(chunks), len(sentences), bundle.get("name", "?"), language,
        speakers[language], pack_words, gen_kwargs,
    )

    def generate_one(chunk: str, chunk_seed: int, max_new_tokens: int) -> tuple[np.ndarray | None, float]:
        set_seed(chunk_seed)
        prompt_enc = prompt_tok(chunk, return_tensors="pt").to(device)
        started = time.time()
        with torch.no_grad():
            audio = model.generate(
                input_ids=desc_ids,
                attention_mask=desc_mask,
                prompt_input_ids=prompt_enc.input_ids,
                prompt_attention_mask=prompt_enc.attention_mask,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
        elapsed = time.time() - started
        # NumPy has no bfloat16 dtype at all — .numpy() on a bf16 tensor always
        # raises "unsupported ScalarType BFloat16" regardless of any later
        # .astype(np.float32), since that only runs AFTER .numpy() failed. Must
        # upcast while still on the torch side, before touching NumPy.
        wav = audio.float().cpu().numpy().squeeze().astype(np.float32)
        if wav.ndim != 1 or wav.size == 0:
            return None, elapsed
        return wav, elapsed

    meta: list[dict] = []
    rendered: list[np.ndarray] = []
    gap = np.zeros(int(sr * gap_ms / 1000.0), dtype=np.float32)

    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        # Scaled to THIS unit's expected length rather than one flat ceiling,
        # so a 4-word unit can't quietly run for 14s and a 34-word unit isn't
        # truncated mid-sentence. See _max_new_tokens_for.
        max_new_tokens = _max_new_tokens_for(word_count, sr, hop_length, language)
        # The SOFT limit: suspect-but-not-truncated. Sits below the token cap,
        # so it catches the stretched takes the cap structurally cannot see.
        expected_sec = _expected_sec_for(word_count, language)
        soft_limit_sec = expected_sec * SOFT_DURATION_MULTIPLIER
        best: np.ndarray | None = None
        best_info: dict = {}
        total_gen_sec = 0.0

        for attempt in range(MAX_CHUNK_ATTEMPTS):
            chunk_seed = seed if attempt == 0 else seed + attempt
            wav, elapsed = generate_one(chunk, chunk_seed, max_new_tokens)
            total_gen_sec += elapsed
            if wav is None:
                log.warning("  [%d/%d] attempt %d produced no audio", i, len(chunks), attempt + 1)
                continue

            # Token count is derived from the RAW (pre-trim) audio length: DAC
            # emits one frame per hop_length samples, so this recovers how many
            # decode steps actually ran without depending on generate()'s
            # return signature (which yields a waveform, not token ids).
            tokens = int(wav.size // hop_length)
            # The cap is set at MAX_DURATION_MULTIPLIER x this unit's expected
            # length, so reaching it means generation never emitted EOS within
            # 3x the time the text should take to speak — i.e. a runaway, now
            # stopped AT the boundary rather than detected after the fact.
            hit_cap = tokens >= int(max_new_tokens * TOKEN_CAP_SUSPECT_RATIO)
            duration_sec = wav.size / sr
            flux = _tail_flux(wav, sr)
            stalled_tail = flux is not None and flux < TAIL_FLUX_SUSPECT
            # Overlong WITHOUT hitting the cap: generation did emit EOS, just far
            # too late. A held vowel short enough to stay under the cap, a repeated
            # phrase, or a sung tail all land here — and none of them move tail
            # flux if the very end of the clip happens to decay normally.
            overlong = duration_sec > soft_limit_sec
            suspect = hit_cap or stalled_tail or overlong

            # Keep the best attempt rather than simply the last one: if every
            # attempt is suspect we still have to ship something. Rank CLEAN above
            # suspect first — with overlong now a suspect reason, ranking on flux
            # alone would happily keep a 2.2x-length take over a clean one, since a
            # stretched clip can still end on a normal-looking decay. Within the
            # same class, the highest-flux take is the one closest to real speech.
            # (None scores as -1 so a scoreable take always beats an unscoreable one.)
            score = (0 if suspect else 1, flux if flux is not None else -1.0)
            prev_score = (
                0 if best_info.get("suspect", True) else 1,
                best_info.get("tail_flux") if best_info.get("tail_flux") is not None else -1.0,
            )
            better = best is None or score > prev_score
            if better:
                best = wav
                best_info = {
                    "tokens": tokens,
                    "max_tokens": max_new_tokens,
                    "hit_token_cap": hit_cap,
                    "tail_flux": flux,
                    "stalled_tail": stalled_tail,
                    "overlong": overlong,
                    "expected_sec": round(expected_sec, 2),
                    "length_ratio": round(duration_sec / expected_sec, 2) if expected_sec else None,
                    "audio_sec": round(duration_sec, 2),
                    "gen_sec": round(elapsed, 2),
                    "rtf": round(elapsed / duration_sec, 2) if duration_sec > 0 else None,
                    "attempts": attempt + 1,
                    "seed": chunk_seed,
                    "suspect": suspect,
                }
            # rtf/ratio describe THIS attempt: best_info may hold an earlier one
            # now that a clean take outranks a later suspect take.
            log.info(
                "  [%d/%d] attempt %d seed=%d | %d words -> %.2fs audio in %.2fs "
                "(rtf %.2f, %.2fx expected, %d tokens%s%s) | %s",
                i, len(chunks), attempt + 1, chunk_seed, word_count, duration_sec, elapsed,
                elapsed / duration_sec if duration_sec > 0 else 0.0,
                duration_sec / expected_sec if expected_sec else 0.0, tokens,
                ", HIT TOKEN CAP" if hit_cap else "",
                ", OVERLONG" if overlong and not hit_cap else "",
                chunk[:60],
            )
            if not suspect:
                break
            # NOTE: a retry changes the seed, which changes the voice
            # realisation for this unit only — an audible prosody jump against
            # its neighbours. "attempts > 1" in the metadata is therefore also
            # a marker for where drift was self-inflicted.
            if hit_cap:
                reason = "hit max_new_tokens (%d) without EOS" % max_new_tokens
            elif stalled_tail:
                reason = ("trailing stall (tail flux %.3f < %.3f — a held vowel, not speech)"
                          % (flux if flux is not None else -1.0, TAIL_FLUX_SUSPECT))
            else:
                reason = ("overlong (%.2fs = %.2fx the %.2fs this text should take in %s, "
                          "limit %.1fx) — EOS came far too late"
                          % (duration_sec, duration_sec / expected_sec, expected_sec,
                             language, SOFT_DURATION_MULTIPLIER))
            log.warning(
                "  [%d/%d] attempt %d %s — retrying with a different seed",
                i, len(chunks), attempt + 1, reason,
            )

        if best is None:
            log.warning("  [%d/%d] skipped — no valid audio after %d attempts", i, len(chunks), MAX_CHUNK_ATTEMPTS)
            meta.append({"index": i, "text": chunk, "words": word_count, "failed": True})
            continue
        if best_info.get("suspect"):
            log.warning("  [%d/%d] still suspect after %d attempts — keeping the last one", i, len(chunks), MAX_CHUNK_ATTEMPTS)

        rendered.append(clean_edges(best, sr))
        meta.append({
            "index": i, "text": chunk, "words": word_count,
            "total_gen_sec": round(total_gen_sec, 2), **best_info,
        })

    if not rendered:
        raise RuntimeError("no audio generated")

    # Level-match BEFORE concatenating: each unit is an independent generation,
    # so their loudness varies, and that variation is a real part of what reads
    # as voice "drift" between sentences. Cheap, model-free, and gentle enough
    # (see LOUDNESS_MAX_GAIN_DB) to leave intentional dynamics alone.
    rendered = match_loudness(rendered, max_gain_db=match_loudness_db)

    pieces: list[np.ndarray] = []
    for unit in rendered:
        pieces.append(unit)
        pieces.append(gap)
    full = np.concatenate(pieces)
    total_gen = sum(m.get("total_gen_sec", 0.0) for m in meta)
    log.info(
        "Done: %.1fs audio from %d unit(s) in %.1fs of generation (overall rtf %.2f)",
        len(full) / sr, len(chunks), total_gen,
        total_gen / (len(full) / sr) if full.size else 0.0,
    )
    return full, sr, meta


def encode_wav(wav: np.ndarray, sr: int, subtype: str = "PCM_16") -> bytes:
    """Serialise a waveform to WAV bytes.

    PCM_16 rather than the model's native float32, for two reasons: it halves
    the payload (which matters a lot now that Queue endpoints return audio
    base64-encoded inside JSON, inflating it a further ~33%), and it removes
    the float32-vs-PCM header mismatch that previously corrupted client-side
    WAV concatenation in lib/tts.ts.
    """
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype=subtype)
    return buf.getvalue()
