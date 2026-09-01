# CLAUDE.md — Indic Parler-TTS Fine-Tuning (Marathi Female Storyteller)

> **⭐ CURRENT WORK = v7 SYNTHETIC DATASET (read this first).** We have PIVOTED away from
> scraped/YouTube audio (copyright risk) to a **copyright-safe synthetic dataset**: generate kids
> stories with Gemini → synthesize audio with **Gemini TTS (Sulafat voice)** → fine-tune Parler.
> Full details + resume instructions are in the **"v7 SYNTHETIC DATASET" section below** and
> `GEMINI_DATASET_PLAN.md`. The v5 multispeaker section that follows is the previous (still-valid)
> pipeline, reused for the v7 build/train step.

> **STATUS / VERSIONS.** PRIOR shipped model = **v6** = the project's BEST run so far and the **v7
> comparison baseline** (single-focus Sunita Marathi + Divya Hindi). Metrics below in the "v6 BASELINE"
> block — use them, not the old "WER 96.2", which was English-Whisper WER (unreliable for Devanagari).
> **v5** = 4-speaker multispeaker fine-tune
> (Sunita target voice + Usha/Maya/Sarita). **For any re-train or crash, read the
> "v5 MULTISPEAKER TRAINING — OPERATIONAL REFERENCE" section just below**, then `RUNPOD_RUNBOOK.md`.
> History: **v1** = ~20-min PoC (done — model `roshni-sorigin/marathi-parler-tts`, narrations in
> `model_outputs_v1/`); **v2** = single-speaker ~5-hr (`marathi_dataset_v2/`); **v5** supersedes the
> earlier multispeaker `_v3` attempts. v1 source is archived as `marathi_dataset_v1/metadata_v1.csv`.
> **Repo layout changed** — scripts now live under `scripts/{phase1_dataprep,phase2_runpod,narrate}/`
> (see `scripts/README.md`); **run all scripts from the repo root**. v2 build artifacts are versioned:
> `marathi_parler_ds_v2/` → `marathi_parler_ds_v2.zip` → pod `/workspace/marathi_parler_ds_v2` →
> download `marathi_tts_model_v2.zip`. v2 training uses a real held-out `test` split (eval/loss is
> meaningful now). Path/command references below describe the original v1 workflow — translate them
> to the v2 names/locations above.

---

## v7 SYNTHETIC DATASET (Gemini-TTS distillation) — CURRENT WORK (2026-07-02)

**Why:** scraped kids-story audio (YouTube, storymarathi.com = the v2 "Usha" set) carries copyright
risk for a commercial app. Deep research confirmed it is unsafe, and that Gemini/OpenAI/ElevenLabs/
Sarvam all forbid training a *competing* model on their output → treat v7 as **research/PoC** until the
Gemini ToS is reviewed or narration is commissioned. Indic-Parler (Apache-2.0) stays as the model.

**The distillation chain:** Gemini generates original stories → **Gemini TTS Pro (`gemini-2.5-pro-preview-tts`),
voice `Sulafat`**, using the COMPACT storyteller prompt → clean `(wav, text)` pairs → fine-tune Parler
with a SIMPLE name-anchored caption. Expressiveness transfers through the AUDIO, not the caption.
Speaker-name anchors: **Sunita = Marathi, Divya = Hindi** (byte-identical caption train↔inference).

### Scripts (`scripts/dataprep_gemini/`)
- `generate_stories.py` — Gemini story text (original characters, Devanagari, numbers-as-words).
- `prep_marathi_corpus.py` — merges story text + filtered v2/panchtantra transcripts into 2–20s
  **units** (sentence-packing target 24w / hard 34w / min 5w), language-aware digit→word
  (`--lang mr|hi`), drops code-mix (≥2 English words). Output: `tts_corpus/<lang>_units.tsv`.
- `synthesize_tts.py` — units TSV → `tts_data/<lang>/audio/chunk_NNNNN.wav` + `metadata.csv`.
  **Resumable** (skips wavs on disk), retries 429/5xx. Uses ONLY Gemini TTS — **NO Sarvam, no ASR**
  (transcript = the exact input text → perfect alignment). Rebuilds `metadata.csv` from all clips at
  run end (so mid-run it lags — the wavs are the source of truth).
- `progress_reporter.py` — standalone detached Slack reporter (see overnight run below).

### Corpora (ready) & audio storage
- `tts_corpus/marathi_units.tsv` — **1771 units, ~4.16 h** (Gemini 373 + v2 326 + panchtantra 1072).
- `tts_corpus/hindi_units.tsv` — **1510 units, ~3.43 h** (Gemini 356 + hindi_panchtantra 1154).
- Audio → `tts_data/marathi/audio/chunk_NNNNN.wav` + `tts_data/marathi/metadata.csv`
  (`audio/chunk.wav|text`); Hindi → `tts_data/hindi/…` (same layout).

### Overnight synthesis run (launched 2026-07-02, ~10 h total on Pro; ~$16–18 all-in)
Three **detached (PPID 1)** processes survive closing Cursor/Claude/terminal:
- Marathi synth: `synthesize_tts.py --units-tsv tts_corpus/marathi_units.tsv --out-dir tts_data/marathi --voice Sulafat` (log `logs/synth_marathi.log`).
- `progress_reporter.py` (log `logs/reporter.log`) — posts to Slack every 30 min via `.env` key
  **`WEBHOOK-URL`**, **auto-launches Hindi** when Marathi hits 1771, posts 🎉 when both complete.
- `caffeinate -is -t 86400` — keeps the Mac awake 24 h (needs power + lid open / clamshell-on-power).
- Rate ~4.5–5.2 clips/min. Marathi ~4.3 h, then Hindi ~5.6 h.
- **Resume if a synth dies:** just re-run the same `synthesize_tts.py` command (skips existing wavs).
- **Restart the reporter:** `nohup venv/bin/python scripts/dataprep_gemini/progress_reporter.py > logs/reporter.log 2>&1 &`
- **Check progress:** `ls tts_data/marathi/audio/*.wav | wc -l` (target 1771) / `tts_data/hindi` (1510).

### NEXT (after both corpora synthesized) — build v7 dataset + train
1. Build a Parler dataset from `tts_data/marathi/metadata.csv` + `tts_data/hindi/metadata.csv` using
   `build_multispeaker_dataset.py` (2 speakers: Sunita=Marathi, Divya=Hindi; captions below).
2. Point `marathi_config.json` at `/workspace/<v7_ds>`; keep `save_steps: 100000`. Train per v5 flow
   below + `RUNPOD_RUNBOOK.md`. Captions (byte-identical train↔inference):
   `"Sunita narrates a children's story in an expressive, warm, animated storytelling tone with emotional variation, at a moderate pace. Very clear audio with no background noise."` (Divya = same, name swapped).

**RunPod cost & the ONE-SITTING RULE (user has ~$5 balance — enough).** A full 4-epoch run ≈ **$1–2**
(train ~1.25–1.5 h; total pod wall-clock setup→upload→train→test→download ≈ **2.5–3 h**; A5000
community ~$0.26/hr, L4 ~$0.43/hr). **$5 comfortably covers one run + a retry.** BUT: `save_steps:
100000` = NO intermediate checkpoints, so ANY mid-train stop (incl. balance hitting $0) loses all
progress → restart from scratch. And on community cloud, **Stop releases the GPU (may not get it back)
AND wipes the container disk (deps + cached base model gone → re-run `runpod_setup.sh`)** — only
`/workspace` survives. RULE: **do the whole run in one continuous session, then TERMINATE — never
Stop.** If forced to redeploy, use the SAME datacenter to reuse the uploaded dataset in `/workspace`.
Prefer On-Demand over Spot/Community for guaranteed mid-run availability.

**Balance-safety (user asked): an intermediate checkpoint does NOT save you if balance hits $0** — a
stopped pod is inaccessible and restarting it needs money. The real safeguard is to **push the final
model to HF (`roshni-sorigin/marathi-parler-tts-v7`) the instant training ends, while balance is still
positive** (~$4 will remain after a ~$0.60–1.30 run) — this gets the model OFF RunPod entirely. Then
scp a local copy + TERMINATE. **Keep 4 epochs (do NOT drop to 3)** so the recipe matches v6 exactly →
a fair v6-vs-v7 comparison for the report. Do NOT add intermediate checkpoints (re-introduces the disk
2× spike that caused past crashes; the final model always auto-saves via `save_pretrained` at the end).
- Full plan/cost/decisions: `GEMINI_DATASET_PLAN.md`. Prompt improvements for the app: `PROMPT_ENHANCEMENT.md`.

### v6 BASELINE (the number to beat with v7) — full card: `MODEL_CARD_v6.md`
Best run to date; **the head-to-head baseline for evaluating v7.** HF repo `roshni-sorigin/marathi-parler-tts-v6`.
- **Voices:** Sunita (Marathi) + Divya (Hindi), selected by caption name (same captions v7 uses).
- **Data:** scraped/real audio — Sunita = v1 (232, train ×2) + v2 (3,376); Divya = 1,667 Hindi panchtantra.
  One combined split: **train 5,395 / test 107, ~6.24 h**.
- **Recipe:** 4 epochs · eff. batch 32 (4 × grad-accum 8) · lr 1e-4 cosine · warmup 50 · bf16 · L4, ~55 min.
- **Metrics (W&B `clean-yogurt-27`):** eval/loss ≈ **4.472** · CLAP ≈ **0.367** · codebook_0 eval-loss ≈
  **3.254** · clean-sample rate ≈ **50%**. (Cross-entropy is a weak quality signal here — judge by ear.)
- **v7's thesis to prove:** synthetic (copyright-safe) data ≈ or > v6's scraped-data quality.

### EVALUATION & FINAL RESEARCH REPORT (LATER — user will prepare; keep hooks intact)
At the end, build an eval set (held-out story text, identical across models) and compare
**v6 (scraped) vs v7 (synthetic) vs Gemini-Sulafat (the teacher / distillation ceiling)** on the SAME text.
Metrics + the tool that computes each:
- **WER / CER via Sarvam Saaras STT** (`saaras:v3`, mr-IN/hi-IN codemix) — the RELIABLE Devanagari
  intelligibility metric; `scripts/dataprep_gemini/score_voices.py` (auto-chunks ≤28 s). NOT English Whisper.
- **CLAP** (text↔audio alignment) — logged in W&B during training.
- **UTMOS** (MOS predictor) + **SQUIM** STOI/PESQ/SI-SDR (relative) — `scripts/narrate/compare_squim.py`.
- **Speaker-cosine** (voice consistency across clips) + **human MOS/CMOS** (small listening panel).
- **Gemini benchmark:** same-text Parler-vs-Gemini table — `scripts/narrate/narrate_gemini.py` for the baseline.
- **Report:** regenerate `reports/tts-finetuning-research_<date>.html` via the `tts-research` skill
  (`tts-report-generator` subagent); it should fold in the v6-vs-v7 W&B comparison. Prior reports: `reports/`.
- **Plain-language explainers to INCLUDE in the report** (user asked for these): `reports/REPORT_EXPLAINERS.md`
  covers, in simple words — how Parler training works, "audio bytes embedded" & Parquet, how steps are
  calculated, what "4 epochs" means, what "codebook" is in W&B, and how to read the runtime graphs, with
  the v7 worked example. The report generator MUST weave these in as an explanatory section.

---

## v5 MULTISPEAKER TRAINING — OPERATIONAL REFERENCE (read for ANY re-train or crash)

> This is the current, working pipeline (2026-06-29). Use it as the template to retrain with a
> different dataset or to recover from a crash. Step-by-step pod commands live in `RUNPOD_RUNBOOK.md`.

**Goal of v5:** one **Sunita** narration voice (the loved v1 timbre — Sunita is also a built-in
Indic-Parler Marathi speaker, so we ADAPT a stable voice, not build from scratch), with storytelling
prosody/pronunciation lifted by a larger multi-speaker corpus.

### Dataset (build is local/CPU; `scripts/phase1_dataprep/build_multispeaker_dataset.py`)
- **One combined `train` + `test`** (parquet, audio bytes embedded, `load_dataset`-ready). Speaker
  identity is the **caption name**, NOT folders. Each speaker is split per-speaker (2%) THEN merged,
  so `test` covers every voice and `eval/loss` is meaningful.
- **Speakers (`SPEAKERS` list):** Sunita←`marathi_dataset_v1` (target voice), Usha←`marathi_dataset_v2`,
  Maya←`marathi_panchtantra_dataset`, Sarita←`hindi_panchtantra_dataset`. All captions = same template,
  only the name + pace differ. **Do NOT run `fix_maya_caption.py`** (it flips Maya moderate→fairly fast).
- **Per-speaker knobs (already wired in):** `"sample":{"n":N,"by":"random"|"english"}` subsamples at
  build time (Maya capped to 3000 random — source dataset untouched); `"train_repeat":K` oversamples a
  speaker's TRAIN rows only (Sunita ×2 → anchors the low-resource target; test untouched, no leakage).
- **Build cmd:** `venv/bin/python scripts/phase1_dataprep/build_multispeaker_dataset.py --output-dir marathi_hindi_parler_ds_v5 --eval-frac 0.02`
- v5 composition: Maya 3000 / Usha 3000 / Sarita 1667 / Sunita 232(train ×2) → train 7967 / test 159.
- **Local disk is tight (~93-96% full); embed-bytes build needs ~11 GB transient.** Delete the prior
  superseded dataset before rebuilding.

### Caption MUST match at inference (or voice/style drifts)
`scripts/phase2_runpod/test_inference.py` + `scripts/narrate/narrate_story.py` `DESCRIPTION` must be
**byte-identical** to the training caption (build script `CAPTION` with name=Sunita, pace=moderate):
`"Sunita narrates a children's story in an expressive, warm, animated storytelling tone with emotional variation, at a moderate pace. Very clear audio with no background noise."`
Prosody is caption-driven — it is NOT transferred speaker→speaker; Sunita's intonation = her own clips
+ the caption's style words; the shared decoder only lifts general pronunciation.

### Training config (`marathi_config.json`) — key v5 values
- `train/eval_dataset_name: /workspace/marathi_hindi_parler_ds_v5` (single; dropped the old `+`-concat).
- `num_train_epochs: 4` (eval/loss plateaus by ~1-2 epochs), `warmup_steps: 100`, `bf16: true`.
- **`save_steps: 100000` (NOT `save_strategy`)** + `save_total_limit: 1`. See checkpoint gotcha below.
- `output_dir: /workspace/marathi_tts_output` (what `download.sh` zips).

### ⚠️ CHECKPOINTING GOTCHA (root cause of past "No space left" crashes)
`run_parler_tts_training.py` is a **manual accelerate loop, NOT HF Trainer** → it **ignores
`save_strategy`** and honors only integer **`save_steps`**. Trigger: `cur_step % save_steps == 0 or
cur_step == total_train_steps`; `rotate_checkpoints` writes the new full-state checkpoint (~10 GB:
model + AdamW optimizer) BEFORE deleting the old → transient 2× (~20 GB) → fills a small volume.
Earlier runs died at step 600 (= 2×save_steps 300). **Fix: `save_steps: 100000` (≫ total ~1000
steps) → NO intermediate checkpoints; the final model is ALWAYS saved at the end via
`unwrapped_model.save_pretrained(output_dir)`.** Trade-off: no resume if the pod dies — just restart.

### GPU + disk requirements
- **GPU: 24 GB+ Ampere/Ada** (A5000 / RTX 4090 / A6000 / A40). Config is bf16 + sized for 24 GB.
  **Avoid 16 GB cards (A4000) and T4/V100** (no native bf16 → OOM / would need fp16 + smaller batch).
- **RunPod disk model:** stopping a pod **wipes the container disk** (pip deps + `~/.cache` base model
  are LOST → re-run `runpod_setup.sh` on restart); only the **network volume `/workspace`** persists.
- **`/workspace` is MooseFS** → `df -h /workspace` shows the whole cluster (~PB), useless. Use
  **`du -sh /workspace`** for your real usage vs the provisioned volume size. Use **`df -h /`** for the
  container disk (real overlay fs). Run peak: ~20 GB on `/workspace`, ~10 GB on container — 50/30 GB fine.

### Retrain with a DIFFERENT dataset (the reusable recipe)
1. Prep each source as `metadata.csv` (`audio/chunk.wav|text`, no empty transcripts) + an `audio/` dir.
2. Edit `SPEAKERS` (names, paths, optional `sample`/`train_repeat`/`pace`).
3. Rebuild: `build_multispeaker_dataset.py --output-dir <NEW_DS> --eval-frac 0.02`.
4. Point `marathi_config.json` `*_dataset_name` at `/workspace/<NEW_DS>`; keep `save_steps` huge.
5. Update inference `DESCRIPTION` to the new training caption. Follow `RUNPOD_RUNBOOK.md`.

### Crash / resume recovery
- **Mid-train crash:** no checkpoint exists (by design) → just relaunch the `accelerate launch` cmd.
- **OOM:** drop `per_device_train_batch_size` 4→2→1 and raise `gradient_accumulation_steps` to keep
  effective batch ~32; drop `per_device_eval_batch_size` 2→1; consider `"optim":"adamw_bnb_8bit"`.
- **Disk-full:** confirm `save_steps` is huge; `du -sh /workspace/*` and delete leftover
  `audio_tokens_tmp` / old datasets / `marathi_tts_output` from prior attempts (keep `parler-tts`).
- **Pod GPU "no longer available" (community cloud):** migrate, or deploy fresh on any 24 GB+ Ampere/Ada
  card; the network volume is datacenter-locked, so a new DC needs a fresh volume (old data is rebuildable).

---

## Project Objective
Fine-tune **ai4bharat/indic-parler-tts** to produce a single **female** Marathi
storyteller voice, narrating **one story genre**, trained on **~20 minutes** of
clean, single-narrator audio (already recorded, no background music).

This is a proof-of-concept. Single speaker, single style, small dataset.
Expect strong style mimicry; accept that rare/unseen words may be weaker, and
that ~20 min is below the comfortable 30-min floor → watch overfitting closely.

### Where the work runs (TWO phases — important)
- **Phase A — LOCAL (your laptop CPU, free):** data prep. Segmentation,
  transcription (faster-whisper on CPU), and building the dataset. No GPU needed.
  These steps are slow-ish on CPU but fine for 20 min of audio.
- **Phase B — RunPod (RTX A5000, ~2 hrs, ~₹45):** ONLY the fine-tuning run and
  the inference test. Upload the prepared dataset, train, download, terminate.

Do NOT attempt the training step (Phase B) on CPU — a 0.9B model trains
50–200× slower on CPU and may take days or never finish. CPU is for prep only.

### Input data (already in hand)
- ~20 min of clean female Marathi narration, single voice, NO background music.
- Stored locally in `./audio_data/` (raw recordings).
- Because it is clean by design, **demucs is NOT used** — skip all separation.

---

## Critical Architecture Facts (read before writing any training code)
- Parler-TTS is **autoregressive**. It predicts **DAC audio-codec tokens** with a
  decoder LM, conditioned on a **frozen Flan-T5 text encoder**.
- There is **NO mel-spectrogram loss and NO GAN discriminator loss** in this
  pipeline. Those belong to VITS/HiFi-GAN architectures. Do **NOT** add or watch
  for `loss_mel` or `loss_disc` — they do not exist here.
- The training objective is **cross-entropy loss** over predicted audio tokens.
- Three components: (1) Flan-T5 text encoder (frozen), (2) decoder LM (trained),
  (3) DAC codec (converts tokens ↔ waveform).
- The model uses a **description/caption** (English voice prompt) + a
  **transcript** (Marathi text). Voice identity is controlled by the caption.
- **Speaker anchoring:** Marathi has built-in named speakers. Recommended female
  voice is **Sunita** (others: Radha, Isha). Anchor the caption to **"Sunita"**
  so the model adapts an existing stable voice rather than building one from
  scratch — far safer with only 30–60 min of data. Alternatively introduce a new
  consistent name, but Sunita is the lower-risk choice.
- **Emotion control:** Marathi officially supports emotion prompts. The relevant
  one for this project is **"Narration"** — put it in the caption. (Others
  available: Happy, Sad, Fear, Surprise, Neutral, Conversation, etc.)
- **Magic phrase:** caption MUST include **"very clear audio"** for high fidelity.
- Punctuation (commas) in the transcript controls pauses/prosody — keep it.

---

## Environment

### Phase A — Local laptop (data prep, CPU only)
- Python 3.10+. No GPU required.
- Install:
```bash
pip install -r requirements.txt
# ffmpeg system dependency:
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt-get install -y ffmpeg
#   Windows: download ffmpeg, add to PATH
```
- Library roles (local):
  - `pydub` — silence-based segmentation into 2–20s chunks
  - `faster-whisper` — Marathi transcription on CPU (`device="cpu"`,
    `compute_type="int8"` for speed)
  - `datasets` — build the HF dataset Parler expects
  - `soundfile` / `ffmpeg` — audio format/rate conversion

### Phase B — RunPod (training only)
- **Compute:** RunPod RTX A5000, 24GB VRAM, accessed via SSH.
- **Template:** RunPod PyTorch 2.x (CUDA pre-installed).
- **Persistent storage:** mount volume at `/workspace`. ONLY `/workspace`
  survives a pod stop/restart. Save dataset, checkpoints, and logs here.
- **Precision:** `bf16=true` (A5000 is Ampere → native BF16, prevents NaN).
- **Sampling rate:** Do NOT hardcode. Read from `model.config.sampling_rate`
  after loading the model; resample all audio to that value before training.
- Install on the pod:
```bash
cd /workspace
bash runpod_setup.sh
```

---

## Logins Required
- **Hugging Face:** YES. Needed to download `ai4bharat/indic-parler-tts` and its
  description/text encoder. Run `huggingface-cli login` and paste a read token.
  You must also click "Agree and access" on the model page once — the repo is
  gated behind a contact-info acceptance. Load the **description tokenizer
  dynamically** from `model.config.text_encoder._name_or_path` rather than
  hardcoding `flan-t5-large`.
- **Weights & Biases:** YES (free personal tier — never hits paywall). Run
  `wandb login` and paste the key. Used only for live metric/audio logging.
  If skipping W&B entirely, set `"report_to": ["tensorboard"]` in the config
  instead — no account needed, logs stay in `/workspace`.

---

## Dataset Format (the target Parler-TTS expects)
A Hugging Face dataset saved to disk at `/workspace/marathi_parler_ds` with
THREE columns per row:

| Column | Content | Example |
|---|---|---|
| `audio` | mono wav at model's sample rate, 2–20 sec | (audio array) |
| `text` | Marathi transcript (Devanagari) | एक आटपाट नगर होतं. |
| `description` | English caption (KEEP IDENTICAL across all rows) | Sunita narrates in an expressive, warm storytelling tone at a moderate pace. Very clear audio with no background noise. |

Rules:
- Single female speaker → ONE fixed `description` string for every row.
- Anchor the caption to **"Sunita"** (recommended Marathi female speaker).
- Frame it as **narration/storytelling** to use the supported Narration emotion.
- `description` must contain the phrase **"very clear audio"** (triggers
  high-fidelity generation).
- Each clip strictly between **2.0 and 20.0 seconds** (target 4–12s sweet spot).
- No background music, hum, echo, or clicks.

---

## STEP-BY-STEP PIPELINE (this PoC)

### PHASE A — LOCAL (laptop CPU, free)

#### Step 1 — Input audio (already done)
- ~20 min clean female narration sits in `./audio_data/` as WAV/MP3.
- No demucs, no separation — the audio is clean by design. Skip all cleaning
  beyond format normalisation.

#### Step 2 — Segmentation → `segment_audio.py`
Slice the long recordings into 2–20s clips at natural silences.
1. Load each file from `./audio_data/`; set `channels=1` (mono) and
   `frame_rate=44100` (Parler's typical rate; the pod will re-confirm/resample).
2. `pydub.silence.split_on_silence` with:
   - `min_silence_len=500` (ms)
   - `silence_thresh=-40` (dBFS)
   - `keep_silence=250` (ms)
3. Merge fragments so each chunk is 2,000–20,000 ms; discard anything <2s.
4. Export to `./marathi_dataset/audio/chunk_00001.wav`, incrementing.
5. Print total chunk count and total duration.

#### Step 3 — Transcription (faster-whisper on CPU) → `transcribe.py`
Turn each chunk into its exact Marathi transcript.
1. Load faster-whisper: `WhisperModel("large-v3", device="cpu",
   compute_type="int8")`. (int8 keeps CPU transcription tractable for 20 min.
   If too slow, drop to `"medium"`.)
2. Loop over `./marathi_dataset/audio/`; transcribe each with `language="mr"`,
   `task="transcribe"` (NOT translate — keep Devanagari output).
3. Write a draft `./marathi_dataset/metadata.csv` as pipe-delimited, headerless:
   `audio/chunk_00001.wav|एक आटपाट नगर होतं.`
4. **MANDATORY human review** (this is the #1 quality lever). While listening to
   each clip, correct the CSV:
   - Numbers → spoken form ("५" → "पाच", never "5").
   - Proper nouns / character names Whisper mangles.
   - Joined words (जोडाक्षरे), missing or extra words.
   - Commas where the narrator pauses (controls TTS prosody).
   - Transcript MUST match audio exactly — mismatched pairs are the top cause of
     a bad fine-tune.
   TIP: if you recorded from a known script, align against that script instead of
   trusting Whisper blind — faster and more accurate.

#### Step 4 — Build HF dataset → `prepare_dataset.py`
1. Read `metadata.csv`. Attach the SAME fixed caption to every row (see Dataset
   Format below).
2. `cast_column("audio", Audio(sampling_rate=44100))`.
3. `ds.save_to_disk("./marathi_parler_ds")`.

#### Step 5 — Upload to RunPod
Zip and copy the prepared dataset to the pod's persistent volume:
```bash
zip -r marathi_parler_ds.zip marathi_parler_ds/
scp -P <POD_PORT> marathi_parler_ds.zip root@<POD_IP>:/workspace/
# then on the pod: cd /workspace && unzip marathi_parler_ds.zip
```

### PHASE B — RUNPOD (RTX A5000, training only)
Proceed to the Training Config section. Confirm the dataset path is
`/workspace/marathi_parler_ds`, run the fine-tune, do the inference test,
download the model, TERMINATE the pod.

---

## Training Config (`/workspace/marathi_config.json`)
Key values for a ~20 min, single-female-speaker run on A5000. Note epochs are set
to 6 (not 8) because 20 min overfits faster — let `eval/loss` decide the real
stopping point. `description_tokenizer_name` below is Parler's default text
encoder; if model load complains, read it from
`model.config.text_encoder._name_or_path` instead of hardcoding.

Launch:
```bash
cd /workspace/parler-tts
accelerate launch ./training/run_parler_tts_training.py /workspace/marathi_config.json
```

---

## Metrics to Watch (CORRECTED for Parler-TTS)
Watch these in the W&B dashboard. There is NO mel loss or discriminator loss here.

- **`train/loss` (cross-entropy)** — should fall steadily then flatten. This is
  the primary signal that the decoder is learning the voice's token patterns.
- **`eval/loss`** — should track train loss down. If eval loss starts RISING
  while train loss keeps falling → **overfitting** (very likely with 30 min of
  one style). Stop / reduce epochs when this happens.
- **Logged audio samples** (`add_audio_samples_to_wandb: true`) — the real
  quality check. Listen to generated clips each eval step. Trust your ears over
  the loss number: pronunciation correct, female voice consistent, no artefacts.
- **`learning_rate`** — confirm the cosine schedule is decaying as expected.

Convergence for ~20 min is usually 4–6 epochs. **Overfitting is the dominant
risk** — stop when `eval/loss` turns up even if epochs remain. ~1–2 hrs on A5000.

---

## Inference Test
After training, run:
```bash
python test_inference.py
```

---

## Save & Teardown (avoid losing work / wasting money)
```bash
cd /workspace
zip -r marathi_tts_model.zip marathi_tts_output/
# Download from LOCAL machine:
# scp -P <PORT> root@<POD_IP>:/workspace/marathi_tts_model.zip ./
```
Then **TERMINATE** the pod in the RunPod console. Billing stops immediately.
Everything outside `/workspace` is already gone on any restart — never rely on it.

---

## Rules for Claude Code / Cursor
1. Read this entire file before writing any script.
2. Two-phase workflow: scripts in Phase A run LOCALLY on CPU (`segment_audio.py`,
   `transcribe.py`, `prepare_dataset.py`); training + `test_inference.py` run on
   the RunPod pod. Do NOT put training on CPU.
3. Input audio is already in `./audio_data/` and is clean — NO demucs, NO yt-dlp,
   no separation step. Do not add them.
4. faster-whisper runs `device="cpu", compute_type="int8"` locally.
5. Write standalone scripts with high code density, robust error logging, no
   placeholder pseudocode.
6. Never introduce mel/discriminator loss logic — wrong architecture.
7. Local prep writes under `./marathi_dataset/` and `./marathi_parler_ds/`;
   pod outputs write under `/workspace/`.
8. Keep the **"Sunita" Narration caption** identical across the whole dataset.
9. On the pod, read sampling rate from `model.config.sampling_rate` and resample;
   load the description tokenizer from `model.config.text_encoder._name_or_path`.
