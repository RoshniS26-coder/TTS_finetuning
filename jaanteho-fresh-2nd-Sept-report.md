# Betacraft TTS + Jaanteho — Session Report
**Dates covered:** 1–2 September 2026
**Repos touched:** `TTS_finetuning`, `jaanteho-fresh`
**Endpoint:** RunPod Serverless *queue* `5tk0794i74t8lh` · image `ghcr.io/betacraft/betacraft-tts:v1` (= `:v1.1`, digest `sha256:c30f5cfe…`)

> **How to read this.** Every number is tagged **[MEASURED]**, **[DERIVED]** (arithmetic on measured
> inputs), or **[ESTIMATE]** (assumption stated). Anything not measured is called out as a gap with a
> way to measure it — several beta-readiness questions asked for below are **not yet measured**, and
> saying so is more useful than a confident guess.

---

## 1. TL;DR

**Good news.** Throughput is at baseline (**65.3 tok/s [MEASURED]**), the generation-parameter
config is stable (**1 retry in 59 attempts = 1.7% [MEASURED]**), and two feared problems turned out
not to exist. GPU cost per story is **$0.034 [DERIVED]** and does not get worse at scale.

**The real problem is cold starts, not capacity.** In today's beta session, **63% of GPU spend
produced no audio [MEASURED]** — 4 workers × 215s of model loading versus 507s of actual synthesis.

**Biggest open risk:** long-story (Complete Story) playback has an arithmetic problem that predicts
~2s of silence between every sentence. **Not yet confirmed by listening** — first thing to check.

**Not measured at all:** per-language success rates. Also note **English is not your fine-tune** —
it is served by stock `ai4bharat/indic-parler-tts` (the fine-tune trained only mr/hi).

---

## 2. Everything tried, and what happened

### 2.1 Tried and REJECTED (with the reason, so it is not retried)

| # | Tried | Outcome | Why |
|---|---|---|---|
| 1 | Rebuild on native amd64 to "fix" flash-attn | **Rejected** | flash-attn IS installed. Runtime log: `flash_attention_2 unavailable (T5EncoderModel does not support Flash Attention 2.0 yet) — falling back to default attention`. A **transformers limitation**, not a build failure. Rebuilding anywhere changes nothing. **[MEASURED]** |
| 2 | Chase a text-truncation bug (`…के लिए क�`) | **Rejected — not a real bug** | `position(U&'\FFFD' in text) = 0` on every `StorySegment` row. The `�` was the terminal pager cutting a long line mid-character. **[MEASURED]** |
| 3 | `PACK_HARD_CAP_WORDS = 16` | **Superseded by 18** | Logs show only 22% of units exceed 16 words and their rtf (1.30) is no worse than shorter units (1.34) — extra splitting buys no speed and adds join boundaries. **[MEASURED]** |
| 4 | Naive conjunction split incl. `पर`/`तो` | **Rejected** | `पर` is usually the postposition "on" (`नक्शे पर`). It split a noun phrase. Both words excluded, reason recorded in code. |
| 5 | Synthetic SFX replacing `झूऽऽऽम` (3 rocket variants) | **Rejected on listening** | `झूऽऽऽम` is a **word a narrator performs**, not an environmental sound. Substitution removed the charm and left a dangling referent. |
| 6 | `झूम झूम` (repeated onomatopoeia) | **Rejected on listening** | Repetition triggers list intonation — first rises, second falls — wrong for a continuous sound. |
| 7 | Pentatonic music-box + pad loader music | **Rejected on listening** | "Sounds like yoga music." No pulse, no hook. |
| 8 | Load Balancer endpoint for long stories | **Deferred, not adopted** | Technically better for streaming (queue job responses have a size cap; a 20-min story would exceed it), but progressive parts already solve it. Keep LB as the standby. |
| 9 | `repetition_penalty` / `no_repeat_ngram_size` | **Confirmed OFF** | Verified they never reach `.generate()` (defaults `None`, guarded). Prior A/B: audibly worse **and** 28% slower. **Do not re-enable.** |

### 2.2 Tried and ADOPTED

| # | Change | File | Live? |
|---|---|---|---|
| 1 | Docker recovery: deleted `Docker.raw` was held open by a running VM; quitting Docker reclaimed 487Mi → 45Gi | — | done |
| 2 | Build OOM fix: `snapshot_download` instead of instantiating two 0.9B models in fp32 (~8 GB peak) just to warm the cache | `Dockerfile` | in `:v1` |
| 3 | `.gitignore` hardened: **25 GB → 7.2 MB, 174 files** | `TTS_finetuning/.gitignore` | done |
| 4 | Conjunction-aware sentence splitting (comma → clause conjunction → hard split) | `betacraft_core.py` | **NOT deployed** |
| 5 | `PACK_HARD_CAP_WORDS = 18` | `betacraft_core.py` | **NOT deployed** |
| 6 | Same splitting, client-side (`MAX_SENTENCE_WORDS = 18`) | `lib/text-segments.ts` | **live on next dev/deploy** |
| 7 | Onomatopoeia spelling rules (continuous = once; iterative = repeat; never avagraha) | `lib/story-prompts.ts` | live |
| 8 | Sentence-length ceiling for ages 6-8 and 9-12 (never past 18 words) | `lib/story-prompts.ts` | live |
| 9 | 8 background music beds, generated (no licensing) | `public/audio/music/` + `scripts/generate-music-beds.py` | live |
| 10 | Music bed hook + mute toggle, both players | `lib/use-music-bed.ts`, `CompleteStoryPlayer.tsx`, `StoryPlayer.tsx` | live |
| 11 | Local preview harness (test segmentation/SFX without redeploying) | `scripts/narrate/preview_local.py` | done |

### 2.3 The hallucination finding (the session's main technical result)

A 21-word Hindi sentence ended with an elongated vowel (`फट गया` → `gayaaaaa`). A/B, **same text, same
seed 42, warm worker [MEASURED]**:

| | units | audio | envelope flux (last 1.3s) | monotonic tail frames |
|---|---|---|---|---|
| **A** whole (21w) | 1 | **11.12s** | **0.016** | **21/25** |
| **B** split in two | 2 | 9.52s | 0.184 | 13/25 |

A emitted **1.6s more audio for identical words**, and that extra is an 11×-smoother, monotonically
decaying signal — a held vowel, not speech. Whole-clip flux was also depressed (0.146 vs 0.210), so
degradation **builds through** the generation rather than appearing only at the end.

**It escaped every existing guard**: it did not hit the token cap (11.12s vs ~9.5s expected = 1.17×,
inside `MAX_DURATION_MULTIPLIER`), and `clean_edges` correctly left it alone because a voiced vowel
sits above its energy threshold.

**→ Highest-value unimplemented change:** add an **envelope-flux check** (~15 lines) next to the
existing token-cap test in `betacraft_core.py`. The 0.016 vs 0.184 separation is a clean
discriminator, and it would trigger the seed-retry that already exists instead of shipping the artifact.

---

## 3. Long stories — current state

**Implemented:** `CompleteStoryPlayer.playBetacraftProgressive()` splits the story and plays part 1
while part 2 generates. Reachable at `CompleteStoryPlayer.tsx:368` for `ttsProvider === "betacraft"`.

**⚠️ Predicted defect — CONFIRM BY LISTENING FIRST.** Parts are **1 sentence** each
(`BETACRAFT_FIRST_PART_SENTENCES = 1`, `BETACRAFT_PART_SENTENCES = 1`), with only **one** part buffered
ahead. But:

```
play one sentence : 6.71s   [MEASURED]
generate the next : 8.86s   [MEASURED]  (rtf 1.33 — slower than real time)
                    → ~2.1s of silence after every sentence   [DERIVED]
```

On a 20-sentence story that is ~40s of dead air, and it will not self-correct: the buffer never gets
a chance to build. **The function's own docstring says "~5-sentence parts", which no longer matches
the constants — stale.**

**Fixes, cheapest first:** (1) buffer 2–3 parts ahead instead of 1 — the correct fix; (2) raise
`BETACRAFT_PART_SENTENCES` to 3–4 while keeping the first at 1; (3) the music bed (now live) masks
the gaps regardless, but does not remove them.

**Size limits:** a 15–20 min story cannot be one queue job — `rp_handler.py` already warns near
RunPod's job-response cap, and 20 min of base64 WAV would be >100 MB. Progressive parts avoid this
entirely. **Not a reason to move to Load Balancer.**

**Success rate: NOT MEASURED.** No long story has been run end-to-end and scored this session.

---

## 4. Economics

### 4.1 Measured inputs
| input | value | source |
|---|---|---|
| GPU per unit (sentence) | **8.86s** | median of 50 jobs **[MEASURED]** |
| audio per unit | **6.71s** | median **[MEASURED]** |
| rtf | **1.33** | median **[MEASURED]** |
| tok/s | **65.3** | median of 58 units **[MEASURED]** |
| retry rate | **1.7%** (1/59) | **[MEASURED]** |
| cold start | **215s** (`delayTime 214813ms`) | **[MEASURED]** |
| sentences/turn today | **6.5** (Hindi) | **[MEASURED]** — prompt asks for "4-5" |
| client fan-out | 4 (`BETACRAFT_MAX_CONCURRENCY`) | config |
| L4 / A5000 | $0.43 / $0.26 per hr | RunPod |

### 4.2 Per story *(5 turns; [DERIVED])*
| | GPU | L4 | A5000 |
|---|---|---|---|
| today (6.5 sentences) | 288s | **$0.0344** | $0.0208 |
| with prompt fixed to 4 | 177s | **$0.0212** | $0.0128 |
| **one cold start** | 215s | **$0.0257** | $0.0155 |

**A single cold start costs more than a whole 5-turn story.**

### 4.3 100 users in an evening *([DERIVED]; 1 story each, 6.9 min session)*
| spread | active at once | avg workers | +2.5× burst | GPU cost |
|---|---|---|---|---|
| over 2 hours | **~6** | **4** | 10 | **$3.44** |
| in a 1-hour peak | ~12 | 8 | 20 | $3.44 |
| all within 30 min | ~23 | 16 | 40 | $3.44 |

**100 subscribed users in an evening is not a capacity problem** — ~10 workers with headroom, ~$3.44
of GPU. Compression changes how many workers you must *provision*, never what you *spend*.

*(Earlier "69 workers for 100 users" assumed 100 users all mid-generation at the same instant — a far
bigger product than 100 subscribers using the app across an evening.)*

### 4.4 Scaling to 10k / 100k *([DERIVED], 2-hour evening peak, 1 story per user)*
| users | avg workers | peak (2.5×) | GPU-hr | $ / evening | $ / user |
|---|---|---|---|---|---|
| 100 | 4 | 10 | 8 | $3 | $0.0344 |
| 1,000 | 40 | 100 | 80 | $34 | $0.0344 |
| 10,000 | 400 | 1,000 | 800 | $344 | $0.0344 |
| 100,000 | 4,000 | 10,000 | 7,999 | $3,439 | $0.0344 |

### 4.5 **Does scale reduce cost per user? — the honest answer**

**Partly, once, and then no.**

GPU cost is **linear**: every story needs its own generation, so there is no economy of scale in
compute itself. Doubling users doubles workers and doubles the bill.

The *one* real economy of scale is **cold-start amortisation**, and it is large at your current stage:

| stage | cold-start share of GPU | $ / story |
|---|---|---|
| **beta today** (1 user, 4 cold starts) | **63%** | **~$0.11** |
| 100 users | 6.9% | $0.0370 |
| 1,000 users | 1.8% | $0.0350 |
| 10,000 users | 0.4% | $0.0345 |
| 100,000 users | 0.1% | **$0.0344** |

**So growth cuts your per-story cost ~3× (≈$0.11 → ≈$0.034) and then flattens.** Beyond that, more
traffic means proportionally more workers with no further unit saving. Anyone modelling "it gets
cheaper as we grow" past that point is wrong.

**The levers that *do* keep cutting cost at scale:**
1. **Sentence count 6.5 → 4** — 39% off every story. Free, prompt-only. **Biggest single lever.**
2. **A5000 instead of L4** — 40% cheaper per GPU-hour; measure quality/rtf parity first.
3. **rtf 1.33** — structural. Every 10% improvement is 10% off both bill and worker count.
4. **Committed/reserved capacity** — at 400+ steady workers, on-demand pricing is the wrong contract.
5. **Cache repeated audio** — stock phrases, intros, and any repeated sentence never need regenerating.

**Ceiling to plan for:** 100k users needs ~4,000 average / ~10,000 peak 24GB GPUs in a 2-hour window.
That is a capacity-planning and vendor-negotiation problem long before it is an engineering one.

### 4.6 Background music: free
The beds are **static MP3s played by the browser** — **zero GPU, zero marginal cost**, 704 KB each.
They cover the cold start, the gaps between sentences, and the wait between interactive turns. This
is the only latency mitigation in the system that costs nothing per play.

---

## 5. Beta-readiness scorecard

| area | status | number | confidence |
|---|---|---|---|
| Throughput | ✅ | 65.3 tok/s, rtf 1.33 | **[MEASURED]** |
| Cost / story | ✅ | $0.034 (L4) | **[DERIVED]** |
| Cost / story incl. cold starts, beta volume | ⚠️ | ~$0.11 | **[DERIVED]** |
| Cold start | ⚠️ | 215s, 63% of beta spend | **[MEASURED]** |
| Runaway/retry rate | ✅ | 1.7% (1/59) | **[MEASURED]**, Hindi only |
| Trailing hallucination | ⚠️ | fixed client-side; server needs redeploy; **no detector** | **[MEASURED]** |
| Background music | ✅ | 8 beds, mute toggle, both players | implemented |
| Long-story success rate | ❌ | **not measured**; ~2s gap predicted | **[DERIVED]** |
| Interactive success rate — Hindi | ⚠️ | 1.7% retry proxy only | partial |
| Interactive success rate — Marathi | ❌ | **not measured** | — |
| Interactive success rate — English | ❌ | **not measured** — *and see below* | — |

### ⚠️ English is not your fine-tuned model
`Dockerfile` comment: the fine-tune **trained only mr/hi**; English is served by stock
`ai4bharat/indic-parler-tts`. So "success rate in all three languages with my fine-tuned
betacraft-tts model" cannot be answered for English — it is a different model, and should be measured
and reported separately.

### How to measure what is missing (proposed, ~1 hour)
1. **20 stories per language** (mr / hi / en) via `preview_local.py`, fixed seeds.
2. Score each unit: **runaway** (hit token cap), **trailing artefact** (envelope flux < 0.05 over the
   last 1.3s), **truncation**, **mispronunciation** (by ear, sample).
3. Report per language: units, failure counts by class, median rtf, median tok/s.
4. Repeat for **5 long stories** end-to-end, timing the gap between parts.

This gives real success rates instead of a single-language retry proxy, and the flux threshold falls
out of it as a shippable detector.

---

## 6. Priority list for tomorrow

| # | Action | Why | Effort |
|---|---|---|---|
| 1 | **Listen to a Complete Story** — confirm/deny the ~2s inter-sentence gap | Biggest unknown; blocks long-story claims | 5 min |
| 2 | **Raise the RunPod idle timeout**, keep 1–2 workers warm | Kills 63% of beta spend | 5 min |
| 3 | **Verify the "4-5 sentences" prompt is obeyed**; enforce/trim if not | 39% off every story | 30 min |
| 4 | Buffer 2–3 parts ahead in `playBetacraftProgressive` | Fixes #1 properly | 1 h |
| 5 | Add the **envelope-flux hallucination detector** | Catches the one failure that escapes all guards | 1 h |
| 6 | Rebuild + redeploy image (`PACK_HARD_CAP_WORDS = 18` + conjunction split) | Backstop for direct callers | 40 min |
| 7 | Run the per-language measurement plan (§5) | Turns ❌ rows into numbers | 1 h |
| 8 | Lower Docker disk limit to 32 GB | Prevented twice today by luck | 2 min |
| 9 | Test the music bed with an actual child | Comprehension is the product | — |

---

## 7. Housekeeping

- **`BETACRAFT_API_KEY` was printed in plaintext** in the terminal during debugging. **Rotate it.**
- `:v1` and `:v1.1` point to the same digest. Prefer immutable tags from now on; `:v1` is a rollback target.
- Local Docker still holds ~50 GB (1 image + 21 GB build cache). `docker image prune -a -f && docker builder prune -a -f` reclaims it; the image is safe in GHCR.
- **This file is `*.md`**, which the new `.gitignore` excludes — an explicit exception was added so it
  can be committed. Other new `.md` files will be ignored unless similarly excepted.
- `scripts/narrate/preview_local.py` lets you test segmentation and SFX **without redeploying**: the
  server splits whatever it is handed, so pre-split units render identically to a redeployed image.

---
*Compiled 2 September 2026. Measured values come from `logs-betacraft-tts (2).txt` (50 jobs / 58 units),
live A/B calls to endpoint `5tk0794i74t8lh`, and direct queries against the local Postgres.*
