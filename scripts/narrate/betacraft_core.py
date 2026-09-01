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
        "speakers": {"mr": "Sunita", "hi": "Divya", "en": "Sunita"},
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
MAX_DURATION_MULTIPLIER = 2.5  # a unit running this much longer than expected is a runaway
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
PACK_TARGET_WORDS = 0
PACK_HARD_CAP_WORDS = 22

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
DEFAULT_TEMPERATURE = _env_opt_float("BETACRAFT_TEMPERATURE", 0.65) or 0.65
DEFAULT_TOP_P = _env_opt_float("BETACRAFT_TOP_P", 0.9)
DEFAULT_REPETITION_PENALTY = _env_opt_float("BETACRAFT_REPETITION_PENALTY", None)
DEFAULT_NO_REPEAT_NGRAM = _env_opt_float("BETACRAFT_NO_REPEAT_NGRAM", None)


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
            sentence = f"{pending.rstrip('।.!?').rstrip()}, {sentence}"
            pending = None
        if len(sentence.split()) < min_words:
            if merged:
                merged[-1] = f"{merged[-1].rstrip('।.!?').rstrip()}, {sentence}"
            else:
                pending = sentence
        else:
            merged.append(sentence)
    if pending:
        if merged:
            merged[-1] = f"{merged[-1].rstrip('।.!?').rstrip()}, {pending}"
        else:
            merged.append(pending)  # whole input was one short fragment
    return merged


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
        log.warning(
            "Sentence fragment of %d words has no comma to break on — hard-splitting "
            "at %d words, which may land mid-clause.", len(piece_words), hard_cap,
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


def _max_new_tokens_for(word_count: int, sr: int, hop_length: int) -> int:
    """Token budget for one unit: MAX_DURATION_MULTIPLIER x its expected length."""
    expected_sec = max(word_count, 1) / EXPECTED_WORDS_PER_SEC
    tokens = expected_sec * MAX_DURATION_MULTIPLIER * sr / hop_length
    return int(min(MAX_TOKENS_CEILING, max(MAX_TOKENS_FLOOR, tokens)))


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
    prompt_ids = bundle["prompt_tok"]("नमस्ते", return_tensors="pt").input_ids.to(bundle["device"])
    with torch.no_grad():
        bundle["model"].generate(
            input_ids=desc_ids, prompt_input_ids=prompt_ids,
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
    desc_ids_by_lang = {
        lang: desc_tok(CAPTION_TEMPLATE.format(name=name), return_tensors="pt").input_ids.to(device)
        for lang, name in spec["speakers"].items()
    }
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

    kwargs: dict = {
        "do_sample": True,
        "temperature": temperature if temperature is not None else DEFAULT_TEMPERATURE,
    }
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
    seed: int = 42,
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

    sentences = split_sentences(normalize_digits(text, language))
    if not sentences:
        raise ValueError("no speakable text after cleaning")
    chunks = pack_sentences(sentences, target_words=pack_words)

    model = bundle["model"]
    prompt_tok = bundle["prompt_tok"]
    device = bundle["device"]
    sr = bundle["sr"]
    hop_length = bundle["hop_length"]
    desc_ids = bundle["desc_ids_by_lang"][language]

    gen_kwargs = build_gen_kwargs(temperature=temperature, **(gen_overrides or {}))
    log.info(
        "Generating %d unit(s) from %d sentence(s), model=%s, lang=%s, speaker=%s, "
        "pack_words=%d, gen_kwargs=%s",
        len(chunks), len(sentences), bundle.get("name", "?"), language,
        speakers[language], pack_words, gen_kwargs,
    )

    def generate_one(chunk: str, chunk_seed: int, max_new_tokens: int) -> tuple[np.ndarray | None, float]:
        set_seed(chunk_seed)
        prompt_ids = prompt_tok(chunk, return_tensors="pt").input_ids.to(device)
        started = time.time()
        with torch.no_grad():
            audio = model.generate(
                input_ids=desc_ids,
                prompt_input_ids=prompt_ids,
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
        max_new_tokens = _max_new_tokens_for(word_count, sr, hop_length)
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
            suspect = hit_cap

            best = wav
            best_info = {
                "tokens": tokens,
                "max_tokens": max_new_tokens,
                "hit_token_cap": hit_cap,
                "audio_sec": round(duration_sec, 2),
                "gen_sec": round(elapsed, 2),
                "rtf": round(elapsed / duration_sec, 2) if duration_sec > 0 else None,
                "attempts": attempt + 1,
                "seed": chunk_seed,
                "suspect": suspect,
            }
            log.info(
                "  [%d/%d] attempt %d seed=%d | %d words -> %.2fs audio in %.2fs "
                "(rtf %.2f, %d tokens%s) | %s",
                i, len(chunks), attempt + 1, chunk_seed, word_count, duration_sec, elapsed,
                best_info["rtf"] or 0.0, tokens,
                ", HIT TOKEN CAP" if hit_cap else "",
                chunk[:60],
            )
            if not suspect:
                break
            # NOTE: a retry changes the seed, which changes the voice
            # realisation for this unit only — an audible prosody jump against
            # its neighbours. "attempts > 1" in the metadata is therefore also
            # a marker for where drift was self-inflicted.
            log.warning(
                "  [%d/%d] attempt %d hit max_new_tokens (%d) without EOS — retrying with a different seed",
                i, len(chunks), attempt + 1, max_new_tokens,
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
