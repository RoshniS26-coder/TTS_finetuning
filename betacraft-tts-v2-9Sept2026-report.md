# Betacraft TTS v2 — deployment and tuning runbook

**Image:** `ghcr.io/betacraft/betacraft-tts:v2` · **Endpoint:** `5tk0794i74t8lh` ·
**Model:** `roshni-sorigin/mar-hin-betacraft-tts` · **Written:** 9 September 2026

Every figure here was measured on this project between 31 August and 9 September 2026.
A web copy exists at <https://claude.ai/code/artifact/7fa51e15-416e-474a-acfa-7d103c90f7e6>,
but this file is the durable one — it ships with the code.

---

## 1. Build and tag as v2 (so you can roll back)

Building over `:v1` destroys your only known-good image. One character buys a rollback that
takes a minute instead of a 40-minute rebuild while the app is broken.

```bash
cd /Users/roshnisundrani/TTS/TTS_finetuning
export HF_TOKEN=$(venv/bin/python -c "from huggingface_hub import HfFolder; print(HfFolder.get_token())")

DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
  --secret id=hf_token,env=HF_TOKEN \
  -t ghcr.io/betacraft/betacraft-tts:v2 .

docker push ghcr.io/betacraft/betacraft-tts:v2
```

Then on RunPod: **endpoint → Manage → set image tag to `:v2` → New release**.

**To roll back:** set the tag back to `:v1` and release again. `:v1` is a different tag and
remains in GHCR untouched.

Reclaim local disk afterwards: `docker image prune -a -f && docker builder prune -a -f`.

---

## 2. What changed in v2

Two of these have never executed in production — the stall detector and the normaliser. v2 is
the first image where the server does anything about hallucination beyond capping runaway length.

| Change | What it does | Evidence |
|---|---|---|
| `normalize_enumerations` | Collapses runs of short comma/ellipsis fragments (countdowns, comma-fenced names). Exempts repeated words (`हा... हा...`) which the corpus does contain. | Same seed, one variable: 1.51x and hallucinating → 1.01x and correct. Confirmed by ear both ways. |
| `_merge_short` | Joins short sentences with a space, keeping their full stops. Previously replaced them with commas — manufacturing the failing pattern out of clean input. | Fires on any sentence under 4 words, so it hit children's stories constantly. |
| `split_sentences` | Keeps `...` intact instead of treating its final dot as a sentence end. | Ellipses were being silently rewritten as commas before reaching the model. |
| `TAIL_FLUX_SUSPECT` | Detects a held vowel at the end of a unit and retries. **Never run in production.** | Separation measured at 11x (0.016 stalled vs 0.184 clean); threshold 0.03 set by listening. |
| `SOFT_DURATION_MULT` | Flags a unit past 1.8x expected as suspect and retries — without truncating. | Catches the 2.19x buzz case that sat under the 2.5x hard cap and shipped silently. |
| `WORDS_PER_SEC_BY_LANG` | English 2.08 vs Marathi/Hindi 1.75. The English token budget was ~19% too generous. | Measured over 75 production units. |
| `attention_mask` | Passed explicitly. No numerical change today; correctness insurance if units are ever batched. | Silences the warning on every generate call. |
| attempt ranking | A clean take now beats a suspect one. Previously ranked on tail flux alone. | Without it a 2.2x-length take could be kept over a clean one. |

---

## 3. The escalation ladder

Work down this list — each tier costs more than the one above. Most problems this project has
had were solved in tiers 0–2.

### Tier 0 — test one sentence, change nothing (~$0.001/call)

Nothing restarts. The handler reads seed and temperature off the payload, so a warm worker
answers in seconds. Use this to find *which* knob matters before touching any config.

```bash
venv/bin/python scripts/narrate/preview_local.py \
  --text "<the sentence that sounded wrong>" \
  --lang mr --seed 648 --temperature 0.65 --audio /tmp/a.wav
```

Vary `--seed` and `--temperature` between runs. If a different seed fixes it, the defect is a
sampling draw, not your text.

### Tier 1 — RunPod endpoint variables (worker restart + 3–4 min cold start)

Eight knobs, no rebuild. **`BETACRAFT_SEED` and `BETACRAFT_TEMPERATURE` are overridden by the
app**, which sends both on every request — change those in Railway instead.

| Variable | Default | Turn it when |
|---|---|---|
| `BETACRAFT_SOFT_DURATION_MULT` | 1.8 | Stretched clips are shipping. Lower to ~1.4 to retry more aggressively; raise if legitimate long units are being needlessly regenerated. |
| `BETACRAFT_PACK_HARD_CAP` | 18 | Long sentences degrade. Splitting fixed 2 of 2 known failures; training median is 20 words, so 16–20 is the sane range. |
| `BETACRAFT_TOP_P` | 0.9 | Words are dropping. **Untested** — a tighter nucleus (0.8) excludes more of the improbable tail, where an early stop token may live. |
| `BETACRAFT_TEMPERATURE` | 0.65 | Only affects direct callers. Lower sharpens the distribution; same mechanism as top_p. |
| `BETACRAFT_REPETITION_PENALTY` | off | **Leave off.** Confirmed to cause audible hallucination and cost 28% throughput. Listed only so nobody re-enables it by accident. |
| `BETACRAFT_NO_REPEAT_NGRAM` | off | Same — leave off. Silence and sustained vowels legitimately repeat in audio-token space. |
| `BETACRAFT_MODEL_DIR` | v7 fine-tune | Point at a different checkpoint. |
| `BETACRAFT_DEVICE` | `cuda:0` | Rarely. Set only to force a device. |

### Tier 2 — Railway variables (app restart, seconds, free)

The app decides seed and temperature for all real traffic. Fastest lever you have.

```
BETACRAFT_TTS_ENDPOINT=https://api.runpod.ai/v2/5tk0794i74t8lh   # required
BETACRAFT_API_KEY=<runpod key>                                    # required
BETACRAFT_SEED=648
BETACRAFT_TEMPERATURE=0.65
BETACRAFT_MAX_CONCURRENCY=4     # must not exceed the endpoint's max workers
```

Seed is not a quality setting in the usual sense. Because the RNG is reset before *every* unit,
one seed applies the same random trajectory to every sentence — which is why some seeds really
are systematically worse.

| Seed | Defects | Sample |
|---|---|---|
| 42 | 4 / 19 | complete ear-scored baseline |
| 1042 | 1 / 19 | complete ear-scored baseline |
| 648 | 1 / 15 | probe run, both text fixes active |

### Tier 3 — story prompt (frontend deploy)

Highest-leverage lever found, because it stops bad input being generated rather than cleaning it
up afterwards. See section 5.

### Tier 4 — rebuild required (~40 min + push)

Hardcoded in `betacraft_core.py`: `TAIL_FLUX_SUSPECT` 0.03, `MIN_UNIT_WORDS` 4,
`MAX_DURATION_MULTIPLIER` 2.5, `MAX_CHUNK_ATTEMPTS` 3, `ENUM_MIN_RUN` 2, `PACK_TARGET_WORDS` 0,
`MAX_TOKENS_CEILING` 2600, `LOUDNESS_MAX_GAIN_DB` 3.0.

If you find yourself wanting one of these often, make it env-tunable in the next build rather
than rebuilding repeatedly.

---

## 4. Symptom → lever

| What you hear | Most likely cause | Try first |
|---|---|---|
| **Held vowel at the end** — "Bhauuuu", "twoooo" | Comma-fenced fragment, or the model failing to stop | Check the sentence for comma runs. v2's normaliser should handle it; if not, lower `BETACRAFT_SOFT_DURATION_MULT` to 1.4. |
| **Last words missing**, clip otherwise fine | Early stop token. *Invisible to every duration check.* | Try another seed in Railway, then `BETACRAFT_TOP_P=0.8`. This is the residual defect with no reliable detector. |
| **Sentence stops halfway**, clearly truncated | Severe early stop | Detectable by duration, but production has no short check yet — the one real gap in v2. See section 6. |
| **Buzz or garbage** after a few words | Runaway that hit the token cap | Should now be caught and retried. If it still ships, lower `BETACRAFT_PACK_HARD_CAP` to 16. |
| **Extra syllable** inserted mid-sentence | Insertion — a third defect class | Nothing addresses this. Rare (1 of 38 observed). Log it and move on. |
| **English sounds worst** | The fine-tune contains **no English at all** and serves it with the Marathi speaker name | Route English to the base model: send `"model": "base"`, which has a real English speaker. |
| **Every request fails** | Config, not audio | `BETACRAFT_API_KEY` missing, or the endpoint URL lacks the endpoint id. Queue endpoints require the key as a bearer token. |
| **First story takes 4 minutes** | Cold start, working as designed | Not a bug. Raise the RunPod idle timeout — the highest-value setting change still unmade. |

---

## 5. Where the generated stories diverge from the training data

Four measurable divergences, ordered by how far off-distribution they are.

| Pattern | In the corpus | What to do |
|---|---|---|
| Comma-separated counting (`तीन, दोन, एक`) | **0** of 3,283 units | Already in the prompt. Keep it. This was the single confirmed cause of hallucination. |
| English narration | **0** units — the fine-tune is Marathi + Hindi only | Biggest structural gap. Either route English to the base model, or accept it as the weakest language. |
| Unseen sound words (`ढप्प`) | **0** occurrences | Prefer sound words the corpus contains. An unseen word is guessed at every time — punctuation can stop making it worse but cannot make it reliable. |
| Sentence length | Median **20** words; 67% exceed 18 | The prompt caps at 18, *below* the training median. The tension is deliberate: splitting fixed 2 of 2 failures on 19–24 word sentences. Do not raise it without re-testing. |
| Ordinary commas | **45%** of Marathi units | Safe. Do not strip commas generally — only runs of one- and two-word fragments break. |

**One prompt change still worth making.** Sound words are currently comma-fenced
(`ढप्प, तो डबा पडला.`), which drops words. Written as its own exclamation
(`ढप्प! तो डबा पडला.`) it came back clean at both seeds tested. That form only survives the
pipeline because of v2's `_merge_short` fix — before it, the `!` was rewritten as a comma.

---

## 6. If v2 is not enough

The measured residual with both text fixes active was **1 defect in 15 clips**, and that one was
a plain control sentence — no commas, not long — so it is the honest floor of what text-level
fixes can reach.

**1. Add a short-duration check (cheapest).** The clip that failed measured **0.42x** its
expected length — severe truncation a simple duration floor *would* catch. Production has no such
check. This is **not** the `min_new_tokens` floor that was rejected: that forced generation to
continue and manufactured stalls. This only rejects a finished take and reseeds, reusing the
retry ladder that already exists.

**2. Verify against the text with speech-to-text.** Partial drops — two or three words missing
from an otherwise normal clip — are invisible to every measurement available: one measured 1.72x,
one 1.01x, one 0.42x. Only comparing the audio against the input text can see them. Sarvam Saaras
(`saaras:v3`) is already wired into this repo and named in `CLAUDE.md` as the reliable Devanagari
metric. Costs one STT call per unit, which can overlap the next generation.

**3. Training data.** If the residual stays above roughly 5% after both, the limit is the model,
not the pipeline. The corpus is 1,772 Marathi and 1,511 Hindi units of flowing narration with no
English, no counting sequences, and none of your sound words.

### Do not spend more time on these

Settled, with evidence:

- **Greedy decoding** (`do_sample=False`) never terminates — every sentence ran to the token cap, 17.25s generated collapsing to 1.8s after trimming.
- **Ellipsis substitution** is *worse* than commas — 2.57x expected, tail flux 0.0093.
- **Repetition penalty / n-gram blocking** are confirmed harmful — audible hallucination plus 28% throughput loss.
- **Seed hunting** — 648 and 1042 both measure 1 defect in 19; 42 measures 4. Pick 648 and move on.
- **bf16 vs fp32 precision** is *not* the cause — dropping occurs on both platforms at the same rate (~13%).

---

## 7. Reference

- Measurement runs and ear-scored verdicts: `local_repro/*/report.json`, `quality_runs/*/report.json`
- Local rate testing: `scripts/narrate/local_repro.py --probes --seeds 648`
- Endpoint testing: `scripts/narrate/preview_local.py`
- Blind GPU/Mac paired scoring: `scripts/narrate/pair_listen.py`
- Cold start: 215s, $0.026. Per turn: $0.007. Per 5-turn story: $0.034.
- Real-time factor: ~1.33 on the GPU, ~8–11 on the Mac (MPS, fp32).
