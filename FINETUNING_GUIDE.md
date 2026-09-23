# Fine-tuning Indic Parler-TTS — a complete guide

How `roshni-sorigin/mar-hin-betacraft-tts` (the **v7** model) was built, end to end.
Every number here was measured on this project. Follow it to reproduce the run, or
to fine-tune Indic Parler on your own voice data.

**Base model:** `ai4bharat/indic-parler-tts` (Apache-2.0)
**Result:** a 2-speaker Marathi + Hindi children's-story narrator
**Cost:** ~$1–2 of GPU, ~1 hour of training, ~3 hours wall-clock
**Data:** 3,213 clips, ~7.6 hours

---

## 1. How Parler-TTS works (and why that shapes everything)

Parler-TTS takes **two** text inputs:

| Input | Example | What it controls |
|---|---|---|
| **prompt** | `एक आटपाट नगर होतं.` | The words that get spoken |
| **description** (caption) | `Sunita narrates a children's story...` | Voice, pace, expressivity, recording quality |

There is **no speaker embedding and no voice cloning**. Identity lives entirely in
the caption text. That single fact drives the whole recipe:

- A speaker is "created" by training many clips that share one caption.
- At inference you must reproduce that caption **byte-for-byte** or the voice drifts.
- Captions stay in **English** even for Marathi/Hindi audio — the description
  encoder is `flan-t5-large`, an English instruction-tuned model. (Confirmed by
  AI4Bharat: *"our descriptions are still in English, and FlanT5 is instruction
  tuned which means better representations even without training it."*)

---

## 2. Prerequisites

### 2.1 HuggingFace token — what it's for and how to make one

You need **two** capabilities, and it is worth understanding why before creating it:

| Capability | Why |
|---|---|
| **Read** | `ai4bharat/indic-parler-tts` and `google/flan-t5-large` are downloaded at train time. Read access is also required later at *inference* if your fine-tune is a private repo. |
| **Write** | To push the finished model to your own HF repo. This matters more than it sounds — see the one-sitting rule in §8.7. |

**Create it:**

1. Go to <https://huggingface.co/settings/tokens>
2. **Create new token** → type **Write** (write implies read)
3. Name it after its purpose, e.g. `parler-finetune`
4. Copy the value — it is shown **once**, and starts `hf_`

**Store it properly.** Log in rather than pasting it into commands, so it never
lands in `~/.zsh_history`:

```bash
huggingface-cli login      # paste at the prompt; answer "n" to the git-credential question
```

This writes `~/.cache/huggingface/token`, which every HF library reads
automatically. Note: clearing that cache to free disk **deletes the token** and
local scripts then fail with a `401 RepositoryNotFoundError`.

To check what a token can do:

```bash
huggingface-cli whoami
```

> **Gated repos.** Some models (e.g. `ai4bharat/IndicF5`) require accepting terms on
> the model page first; a valid token alone is not enough.

### 2.2 Everything else

```bash
git clone https://github.com/huggingface/parler-tts.git
cd parler-tts && pip install -e .[train]
```

Install **from git, not PyPI** — PyPI's resolver hits `resolution-too-deep`
against an already-installed torch, and the PyPI release predates Indic Parler's
dual-tokenizer setup.

You also need a Gemini API key (for data generation) and optionally a W&B key
(for training metrics).

---

## 3. The data: what, how much, and why synthetic

### 3.1 Why synthetic

The earlier datasets (v1–v6) used scraped children's-story audio from YouTube and
story sites. That carries **copyright risk for a commercial product**, so v7 was
rebuilt from generated audio:

> Gemini writes original stories → **Gemini TTS Pro** (`gemini-2.5-pro-preview-tts`),
> voice **Sulafat** → clean `(wav, text)` pairs → fine-tune Parler.

Expressiveness transfers through the **audio**, not the caption. The caption stays
simple; the teacher model's prosody is what the student learns.

> ⚠️ **Licensing caveat.** Gemini, OpenAI, ElevenLabs and Sarvam all forbid training
> a *competing* model on their output. v7 is research/PoC until terms are reviewed
> or narration is commissioned. Indic Parler itself is Apache-2.0 and unaffected.

### 3.2 How much data

| | Clips | Hours | Median clip | Median words |
|---|---|---|---|---|
| **Sunita** (Marathi) | 1,734 | ~5.2 h | 10.9 s | 20 |
| **Divya** (Hindi) | 1,479 | ~3.2 h | 7.7 s | 19 |
| **Total** | **3,213** | **~7.6 h** | 9.3 s | — |

Measured speaking rate in the training audio: **Marathi 1.84 w/s, Hindi 2.40 w/s.**

Clip length distribution (all, after filtering to 2–20 s):

```
 0-3s:    23  ( 0.7%)
 3-5s:   175  ( 5.4%)
 5-7s:   508  (15.8%)
 7-10s: 1157  (36.0%)
10-15s: 1291  (40.2%)
15-25s:   59  ( 1.8%)
```

**This distribution matters at inference.** See §10 — it is the single biggest
lesson from running the model in production.

### 3.3 Where the text came from

Three sources, merged into one corpus:

| Source | Marathi units | Hindi units |
|---|---|---|
| Gemini-generated stories | 373 | 356 |
| Reused v2 transcripts | 326 | — |
| Reused panchtantra transcripts | 1,072 | 1,154 |
| **Total** | **1,771** | **1,510** |

The reused transcripts came from the **earlier** scraped datasets, which had been
transcribed with **Sarvam Saaras STT** (`scripts/phase1_dataprep/transcribe_saaras.py`).
That text was kept; the audio was not.

> **Important distinction.** No ASR runs in the v7 pipeline itself. The transcript
> for every v7 clip *is the exact text handed to Gemini TTS*, so alignment is
> perfect by construction. Sarvam appears in two other places only: transcribing
> the older scraped corpora (whose text was reused), and **evaluation** (WER/CER
> scoring, `saaras:v3`). Never use English Whisper on Devanagari — it silently
> returns translations.

---

## 4. Step 1 — Generate story text

```bash
python scripts/dataprep_gemini/generate_stories.py
```

Rules the prompt enforces, each learned the hard way:

- **Original characters only** — no copyrighted names.
- **Devanagari output**, no transliteration.
- **Numbers as words** (`तीन`, not `3`) — digits are read unpredictably.
- **No code-mixing** — units with ≥2 English words are dropped later.

---

## 5. Step 2 — Build the unit corpus

```bash
python scripts/dataprep_gemini/prep_marathi_corpus.py --lang mr
python scripts/dataprep_gemini/prep_marathi_corpus.py --lang hi
```

Merges the three text sources and packs sentences into speakable **units**:

| Setting | Value |
|---|---|
| Target | 24 words |
| Hard cap | 34 words |
| Minimum | 5 words |
| Duration window | 2–20 s (sweet spot 4–12 s) |

Also: language-aware digit→word conversion, and code-mix filtering.

Output: `tts_corpus/<lang>_units.tsv`

---

## 6. Step 3 — Synthesize the audio

```bash
python scripts/dataprep_gemini/synthesize_tts.py \
    --units-tsv tts_corpus/marathi_units.tsv \
    --out-dir  tts_data/marathi \
    --voice    Sulafat
```

Produces `tts_data/<lang>/audio/chunk_NNNNN.wav` plus `metadata.csv`.

**metadata.csv format** — pipe-separated, relative path first:

```
audio/chunk_00001.wav|एका गावात एक छोटा ससा राहत होता.
audio/chunk_00002.wav|तो रोज सकाळी लवकर उठायचा.
```

Practical notes for a run of this size (~3,300 clips, ~10 hours):

- **Resumable** — re-running skips wavs already on disk. If it dies, just re-run.
- Retries 429/5xx automatically. Rate ≈ 4.5–5.2 clips/min.
- `metadata.csv` is rebuilt from disk at the end, so mid-run it lags — **the wavs
  are the source of truth**.
- Run detached (`PPID 1`) so closing the terminal doesn't kill it, and use
  `caffeinate -is -t 86400` to keep the machine awake.

Cost for the full v7 corpus: **~$16–18** on Gemini TTS Pro.

---

## 7. Step 4 — Build the Parler dataset

```bash
python scripts/phase1_dataprep/build_multispeaker_dataset.py \
    --v7 --output-dir marathi_hindi_parler_ds_v7 --eval-frac 0.02
```

### 7.1 The caption

```python
CAPTION = ("{name} narrates a children's story in an expressive, warm, animated "
           "storytelling tone with emotional variation, at a moderate pace. "
           "Very clear audio with no background noise.")

SPEAKERS_V7 = [
    {"name": "Sunita", "metadata": "tts_data/marathi/metadata.csv"},
    {"name": "Divya",  "metadata": "tts_data/hindi/metadata.csv"},
]
```

**Only the name differs between speakers.** Language is carried by the transcript
text, not the caption — so both captions share identical style wording. This is
deliberate: it means a later A/B isolates the *training*, not the prompt.

**Write this string down.** You must reproduce it byte-for-byte at inference.

### 7.2 The dataset format

Saved as a `DatasetDict({"train", "test"})` in parquet, with exactly three columns:

| Column | Type | Content |
|---|---|---|
| `audio` | `Audio` | The waveform (resampled to the model's rate) |
| `text` | `string` | The transcript — what gets spoken |
| `description` | `string` | The caption — who speaks and how |

Each speaker is split into train/test **individually**, then concatenated, so the
held-out eval covers every voice. `--eval-frac 0.02` gives ~2% per speaker.

Minimal equivalent if you're building your own:

```python
from datasets import Dataset, Audio, DatasetDict

rows = [{"audio": "audio/chunk_00001.wav",
         "text": "एका गावात एक छोटा ससा राहत होता.",
         "description": CAPTION.format(name="Sunita")}]

ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=44100))
DatasetDict({"train": ds, "test": ds.select(range(2))}).save_to_disk("my_ds")
```

---

## 8. Step 5 — Train on RunPod

### 8.1 Have everything ready BEFORE deploying

**Billing starts the moment the pod reads "Running"**, so don't fumble for these
while paying:

```
[ ] Dataset directory built (train.parquet + test.parquet) — no zip needed
[ ] marathi_config.json, runpod_setup.sh, test_inference.py
[ ] SSH public key added to RunPod → Settings → SSH Public Keys
[ ] HF token ready (§2.1)
[ ] HF gated access ACCEPTED — click "Agree and access" once at
    https://huggingface.co/ai4bharat/indic-parler-tts
[ ] (Optional) W&B key from https://wandb.ai/authorize
    — or set "report_to": ["tensorboard"] in the config and skip it
```

That gated-access click is easy to miss and fails the run 10 minutes in, on a
paid GPU, with a 401.

**SSH key, if you don't have one:**

```bash
cat ~/.ssh/id_ed25519.pub                    # existing?
ssh-keygen -t ed25519 -C "runpod"            # if not — Enter at every prompt
cat ~/.ssh/id_ed25519.pub                    # copy the whole line into RunPod
```

### 8.2 Deploy the pod

| Setting | Value | Why |
|---|---|---|
| GPU | **RTX A5000** (24 GB) | Enough for effective batch 32. L4 also works |
| Cloud | Community | ~$0.26/hr vs L4 ~$0.43/hr |
| Type | **On-Demand** | Not Spot — a mid-run eviction loses everything |
| Template | RunPod PyTorch 2.x | |
| **Volume Disk** | **80 GB at `/workspace`** | 40–50 GB caused "No space left" crashes. Peak usage ~30 GB |
| Container Disk | ~20 GB | |

Deploy → wait for **Running** → **Connect → SSH over exposed TCP** → note the IP
and port.

Only `/workspace` survives a Stop. The container disk does not.

### 8.3 Upload, connect, install

```bash
# [LAPTOP] verify SSH first
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
nvidia-smi        # confirm the GPU, then: exit

# [LAPTOP] upload dataset + files (~3.4 GB, scp -r, no zip)
scp -r -P <PORT> -i ~/.ssh/id_ed25519 \
    marathi_hindi_parler_ds_v7 \
    marathi_config.json \
    scripts/phase2_runpod/runpod_setup.sh \
    scripts/phase2_runpod/test_inference.py \
    root@<IP>:/workspace/

# [POD]
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
cd /workspace
ls marathi_hindi_parler_ds_v7      # expect train.parquet + test.parquet
df -h /workspace                   # confirm free space BEFORE training
bash runpod_setup.sh               # clones parler-tts + deps, ~5-10 min
```

If you're **reusing** a volume, check for leftovers first — `du -sh /workspace/*`
— and delete stale `marathi_tts_output/`, `audio_tokens_tmp/`, `ds_pq*` while
keeping `parler-tts/`. On a fresh volume there's nothing to clean.

Point `marathi_config.json` at the uploaded dataset path before training.

### 8.4 The config (`marathi_config.json`, v7 values)

```json
{
  "model_name_or_path":         "ai4bharat/indic-parler-tts",
  "feature_extractor_name":     "ai4bharat/indic-parler-tts",
  "description_tokenizer_name": "google/flan-t5-large",
  "prompt_tokenizer_name":      "ai4bharat/indic-parler-tts",

  "num_train_epochs":            4,
  "learning_rate":               0.0001,
  "lr_scheduler_type":           "cosine",
  "warmup_steps":                50,
  "weight_decay":                0.01,

  "per_device_train_batch_size":  4,
  "gradient_accumulation_steps":  8,
  "gradient_checkpointing":       true,
  "dtype":                        "bfloat16",
  "bf16":                         true,

  "freeze_text_encoder":          true,
  "min_duration_in_seconds":      2.0,
  "max_duration_in_seconds":     20.0,
  "group_by_length":              true,

  "save_steps":                   100000,
  "seed":                         456
}
```

Why these values:

- **Effective batch 32** = 4 × 8 gradient accumulation. Fits a 24 GB card.
- **`freeze_text_encoder: true`** — flan-t5-large is frozen. You are training the
  audio decoder, not the language understanding.
- **`save_steps: 100000`** — deliberately larger than the ~1,000 total steps, so
  **no intermediate checkpoints are written**. This fixes "No space left" crashes
  caused by two checkpoints overlapping during rotation. The trade-off is real:
  a pod that dies mid-run leaves nothing to resume from.
- **Duration window 2–20 s** matches how the corpus was packed.

### 8.5 Tokenizer choice — the most common failure

Getting these wrong produces audio that is *technically fine and completely wrong*:

| Field | Value | Why |
|---|---|---|
| `description_tokenizer_name` | `google/flan-t5-large` | The description **encoder** — captions are English |
| `prompt_tokenizer_name` | `ai4bharat/indic-parler-tts` | Handles Devanagari with byte fallback |
| `feature_extractor_name` | `ai4bharat/indic-parler-tts` | Or `ylacombe/dac_44khz` — AI4Bharat confirm both work; DAC is not fine-tuned on Indic data |

Using an English single-tokenizer setup (the original Parler example) on Devanagari
gives unusable output.

### 8.6 Running it

```bash
# On the pod
huggingface-cli login
wandb login

tmux new -s train        # so a dropped SSH connection doesn't kill the run
cd /workspace/parler-tts
accelerate launch ./training/run_parler_tts_training.py /workspace/marathi_config.json
```

Detach with **Ctrl-b** then **d**; reattach with `tmux attach -t train`.

- ~1,000 steps, **~1 hour** on an L4.
- Watch `eval/loss` every 100 steps. If it rises while `train/loss` falls, that's
  overfitting — Ctrl-C is fine.
- Output: `/workspace/marathi_tts_output`

> **Cross-entropy is a weak quality signal for TTS.** v6 finished at eval/loss
> ≈ 4.472 with a ~50% clean-sample rate. **Judge by ear.**

### 8.7 The one-sitting rule

On community-cloud GPUs, **Stop releases the GPU and wipes the container disk** —
only `/workspace` survives. Combined with no intermediate checkpoints:

> Do the whole run in one continuous session, then **TERMINATE — never Stop.**

And push to HuggingFace **the instant training ends**, while the balance is still
positive. A stopped pod with $0 balance is unreachable, and an intermediate
checkpoint doesn't save you — restarting it costs money you no longer have.

### 8.8 Optional — LoRA instead of full fine-tuning

Everything above is **full fine-tuning**: every weight in the ~900M-parameter
decoder is rewritten, which is why the output is a 3.5 GB `model.safetensors`.

> ⚠️ `freeze_text_encoder: true` is **not** LoRA. It freezes flan-t5-large only;
> the whole decoder still trains. (`audio_encoder`/DAC is frozen unconditionally by
> `freeze_encoders()`, with no flag. There is no option to freeze the decoder — it
> is the only thing being trained.)

**Why consider it.** Full fine-tuning on a narrow corpus moves weights away from
what the base model knew. Measured here: stock base was roughly **twice as
consistent** on the same Marathi sentences as the v7 fine-tune. LoRA freezes the
base entirely and trains small low-rank adapters, so base ability cannot be
overwritten.

**Support status — read first.** Parler-TTS has **no official LoRA support**. It
exists as an *unmerged* pull request, open since February 2025:

- <https://github.com/huggingface/parler-tts/pull/204> — uses the `peft` library
- <https://github.com/huggingface/parler-tts/pull/159> — the earlier custom version it replaced

You do **not** need to open a PR or wait for a merge. Merging only decides whether
the official release ships LoRA by default; you can check out that branch onto your
pod today. The real cost of "unmerged" is that no maintainer reviewed it and it was
written against a 2025 library version — so it may not apply cleanly.

Reported in the PR: trains **0.5% of parameters**, fine-tunes Parler-Mini on an
**8 GB GPU**, ~4x faster per iteration, ~50% faster inference after merging.

**Setup — one extra command over the normal clone:**

```bash
git clone https://github.com/huggingface/parler-tts.git
cd parler-tts
gh pr checkout 204          # or: git fetch origin pull/204/head:lora && git checkout lora
pip install -e .[train] peft
```

**Config additions:**

```json
"use_lora":      true,
"lora_r":        8,
"lora_alpha":    16,
"lora_dropout":  0.05,
"learning_rate": 0.0002
```

Note the **higher** learning rate — LoRA moves far fewer parameters, so it wants
1e-4 to 3e-4. The 1e-4 used for full fine-tuning is conservative here.

**Verify three things before booking a full run:**

1. **Does it still train on your stack?** Run ~50 steps, not 1,000. Minutes and
   cents. The PR predates your `transformers`/`torch`, so this is where a stale
   patch shows up.
2. **Are the 9 codebook `lm_heads` covered?** The PR applies LoRA to the decoder's
   linear projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`). Parler emits audio
   through nine parallel codebook heads; if those stay frozen the model may not
   learn the voice. Most LoRA-for-TTS work targets single-codebook models, so this
   is the least-tested part. **Generate a sentence after a short run and listen** —
   if it sounds like base rather than your speaker, the heads are not training.
3. **Will your serving path load it?** LoRA produces a small
   `adapter_model.safetensors`, not a full model directory. Simplest fix — merge
   before pushing, so everything downstream is unchanged:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload()
model.save_pretrained("/workspace/marathi_tts_output")   # a normal 3.5 GB model
```

**The trade nobody mentions.** LoRA protects base knowledge, but 0.5% of parameters
is a small budget for learning a *new voice identity* — which is a large change, not
a small one. Full fine-tuning generally wins on speaker similarity. So this is a
genuine trade-off, not a free upgrade: less forgetting, possibly a weaker voice.
Judge by ear against the current model, not by assumption.

**The cheaper alternative.** LoRA's appeal is "don't move the base weights far."
A gentler full fine-tune approximates that with no third-party code:

| | v7 | Gentler |
|---|---|---|
| `learning_rate` | 1e-4 | **2e-5 – 5e-5** |
| `num_train_epochs` | 4 | **2–3** |
| Starting checkpoint | `indic-parler-tts` (already a fine-tune) | `indic-parler-tts-pretrained` |

One config change, one $1–2 run. If a low-LR run *still* degrades Marathi relative
to base, that is evidence the problem is the **data distribution** (§10) rather than
weight movement — learned before building any LoRA plumbing.

---

## 9. Step 6 — Push, then verify

```bash
python /workspace/push_to_hf.py \
    --model-dir /workspace/marathi_tts_output \
    --repo      roshni-sorigin/mar-hin-betacraft-tts
```

Smoke-test before terminating the pod:

```python
import torch, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL = "roshni-sorigin/mar-hin-betacraft-tts"
CAPTION = ("Sunita narrates a children's story in an expressive, warm, animated "
           "storytelling tone with emotional variation, at a moderate pace. "
           "Very clear audio with no background noise.")

device = "cuda:0"
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL).to(device)
prompt_tok = AutoTokenizer.from_pretrained(MODEL)
desc_tok   = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

desc = desc_tok(CAPTION, return_tensors="pt").to(device)
prm  = prompt_tok("एक आटपाट नगर होतं. तिथे एक राजा राहत होता.", return_tensors="pt").to(device)

audio = model.generate(input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                       prompt_input_ids=prm.input_ids, prompt_attention_mask=prm.attention_mask)
sf.write("test.wav", audio.cpu().numpy().squeeze(), model.config.sampling_rate)
```

Note the **two tokenizers** — one for the prompt, one for the description, the
latter resolved dynamically from `model.config.text_encoder._name_or_path`. Don't
hardcode `flan-t5-large`; read it from the config.

---

## 10. What we learned running it in production

The model works. These are the things that only surfaced after deployment, and
they're the most useful part of this document.

### The caption is load-bearing

Changing it moves the model off-distribution. Measured on the same text and seed:

| Caption | Median length ratio | Retries |
|---|---|---|
| Trained caption | **1.00x** | 0 |
| A calmer variant | 1.25x | 1 |

The opposite is true for the **base** model, which is caption-steerable by design
and scored best with the calmer wording (1.01x, 0 retries). A fine-tune wants *its*
caption; a base model wants a *good* caption.

**Exception worth knowing:** English — never fine-tuned — *improved* when its caption
named `Mary` (a recommended English voice) instead of `Sunita` (Marathi-only). For
languages the fine-tune never touched, the base model's speaker conditioning survives.

### Train on the clip lengths you will actually request

This is the biggest finding. Training clips were a median of **20 words**, with 90%
at 13+ words. Production generates **one sentence at a time — 4 to 13 words**, i.e.
below the 10th percentile of anything the model saw.

Counter-intuitively, packing production text *up* to 20 words made things **worse**
(retry rate 4% → 33%): sentence boundaries don't divide evenly, so you land near 15
words — longer than what works, still short of training, and each unit now carries
more text a single alignment slip can damage.

**So fix it at training time**: include short clips matching your inference pattern.
`min_duration_in_seconds: 2.0` already permits them — the *dataset builder* has to
produce them.

### Sampling settings matter more than expected

Measured on the deployed model:

| Setting | Effect |
|---|---|
| `temperature` 1.0 | Prosody varies between sentences — sounds like different narrators |
| `temperature` 0.65 | Best |
| `temperature` 0.5 | Loud buzzing |
| greedy (`do_sample=False`) | Never emits a stop token — ~90% garbage |
| **`top_p` 0.9** | **Dropped words and held vowels** |
| **`top_p` 1.0 (off)** | Materially better — confirmed by ear on identical seeds |

Over-constrained sampling degenerates. On a diffuse distribution — which is what an
undertrained region looks like — trimming to the top 90% of probability mass can
discard the *correct* continuation.

### Consider a gentler recipe

v7 used `lr 1e-4` for 4 epochs on ~7.6 hours of data, applied on top of AI4Bharat's
*already fine-tuned* checkpoint. Comparable community runs use `8e-5` for 2 epochs.
The stock base model measured roughly **twice as consistent** on the same sentences.

If you rerun, consider: lower LR (2e-5–5e-5), fewer epochs, and starting from
`ai4bharat/indic-parler-tts-pretrained` rather than stacking two style fine-tunes.

### Autoregressive decoding has a failure mode you can't tune away

Dropped words and held vowels are **cross-attention misalignment** — the decoder
loses its place. It is stochastic: the same sentence at one seed is clean and at
another drops words. Duration cannot detect it (failures measured at 0.52x, 1.07x,
1.24x and 1.37x of expected length — both directions).

Detecting it needs **forced alignment or ASR** comparing audio to the source text.
Flow-matching models (e.g. IndicF5) don't have this failure mode at all.

---

## 11. Checklist

```
[ ] HF token created (write scope) and `huggingface-cli login` done
[ ] parler-tts installed from git, not PyPI
[ ] Story text generated — Devanagari, numbers as words, no code-mix
[ ] Corpus packed into 2-20s units, INCLUDING short ones
[ ] Audio synthesized; metadata.csv is `relative/path.wav|transcript`
[ ] Dataset built: columns audio / text / description; per-speaker train+test split
[ ] CAPTION recorded verbatim somewhere permanent
[ ] Config: flan-t5-large description tokenizer, Indic prompt tokenizer
[ ] save_steps > total steps (no mid-run checkpoint rotation)
[ ] Trained in tmux, one continuous session
[ ] Pushed to HF IMMEDIATELY after training
[ ] Smoke-tested with the byte-identical caption
[ ] Pod TERMINATED (not Stopped)
```

---

## Reference

| | |
|---|---|
| Base model | `ai4bharat/indic-parler-tts` |
| Result | `roshni-sorigin/mar-hin-betacraft-tts` |
| Training code | <https://github.com/huggingface/parler-tts> |
| Data prep | `scripts/dataprep_gemini/`, `scripts/phase1_dataprep/` |
| Pod setup | `RUNPOD_RUNBOOK.md` |
| Serving | `scripts/narrate/betacraft_core.py`, `rp_handler.py` |
| Dataset plan | `GEMINI_DATASET_PLAN.md` |
