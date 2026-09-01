# Marathi Parler-TTS Fine-Tuning

Proof-of-concept fine-tune of [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) for a single female Marathi storyteller voice (Sunita), trained on ~20 minutes of clean narration.

**Two phases:** data prep runs locally on CPU (free); training runs on RunPod A5000 GPU only.

> **v1 / v2.** This README describes **v1** (the ~20-min PoC; archived as `marathi_dataset_v1/`,
> `model_outputs_v1/`). Active work is **v2** (~5 hr, `marathi_dataset_v2/metadata.csv`). Scripts moved
> under `scripts/{phase1_dataprep,phase2_runpod,narrate}/` — **run them from the repo root** (see
> `scripts/README.md`). v2 artifacts are versioned `*_v2` (dataset/zip/model). See `PLAN.md` for the v2 plan.

See [CLAUDE.md](CLAUDE.md) for the full specification.

## Prerequisites

- Python 3.10+
- ffmpeg (`brew install ffmpeg` on macOS)
- Hugging Face account with access to `ai4bharat/indic-parler-tts` (gated model — click "Agree and access" on the model page)
- Weights & Biases account (free tier)

## Phase A — Local data prep (CPU)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place raw audio

Put your ~20 min of clean female Marathi narration (WAV or MP3) in:

```
./audio_data/
```

No background music. Single narrator only.

### 3. Run the pipeline

```bash
python segment_audio.py
python transcribe.py
```

**Stop and review:** Open `marathi_dataset/metadata.csv`. Listen to each clip and correct transcripts — this is the #1 quality lever. Fix numbers, proper nouns, punctuation (commas control pauses), and any Whisper errors. If you have the original script, align against it.

```bash
python prepare_dataset.py
```

This writes `./marathi_parler_ds/` — the Hugging Face dataset ready for upload.

### Expected runtime (Phase A)

| Step | Time (approx.) |
|---|---|
| Segmentation | 1–2 min |
| Transcription (large-v3, CPU int8) | 30–60 min for ~20 min audio |
| Dataset build | < 1 min |

## Upload dataset to RunPod

```bash
zip -r marathi_parler_ds.zip marathi_parler_ds/
scp -P <POD_PORT> marathi_parler_ds.zip root@<POD_IP>:/workspace/
scp -P <POD_PORT> marathi_config.json root@<POD_IP>:/workspace/
scp -P <POD_PORT> test_inference.py root@<POD_IP>:/workspace/
scp -P <POD_PORT> runpod_setup.sh root@<POD_IP>:/workspace/
```

On the pod:

```bash
cd /workspace
unzip marathi_parler_ds.zip
```

## Phase B — RunPod training (GPU only)

### 1. Launch pod

- Template: RunPod PyTorch 2.x
- GPU: RTX A5000 (24 GB VRAM)
- Mount persistent volume at `/workspace`

### 2. Setup

```bash
cd /workspace
bash runpod_setup.sh
huggingface-cli login
wandb login
```

### 3. Train

```bash
cd /workspace/parler-tts
accelerate launch ./training/run_parler_tts_training.py /workspace/marathi_config.json
```

### 4. Monitor (W&B)

| Metric | What to watch |
|---|---|
| `train/loss` | Should fall steadily (cross-entropy on audio tokens) |
| `eval/loss` | Rising while train loss falls = overfitting — stop early |
| Audio samples | Listen each eval step — trust your ears |

There is **no** `loss_mel` or `loss_disc` — Parler-TTS does not use mel/GAN losses.

Expected training time: **1–2 hours** on A5000.

### 5. Inference test

```bash
cd /workspace
python test_inference.py
# optional custom prompt:
python test_inference.py --prompt "तुमचा मराठी मजकूर येथे."
```

Output: `/workspace/test.wav`

### 6. Download and terminate

```bash
cd /workspace
zip -r marathi_tts_model.zip marathi_tts_output/
```

From your local machine:

```bash
scp -P <POD_PORT> root@<POD_IP>:/workspace/marathi_tts_model.zip ./
```

**Terminate the pod** in the RunPod console when done. Only `/workspace` survives restarts.

## Project layout

```
TTS_finetuning/
├── audio_data/              # raw recordings (you provide)
├── marathi_dataset/         # generated chunks + metadata.csv
├── marathi_parler_ds/       # generated HF dataset
├── segment_audio.py         # Step 2: silence-based chunking
├── transcribe.py            # Step 3: faster-whisper transcription
├── prepare_dataset.py       # Step 4: HF dataset builder
├── marathi_config.json      # RunPod training config
├── test_inference.py        # post-training smoke test
└── runpod_setup.sh          # pod dependency installer
```

## Caption (fixed across all rows)

```
Sunita narrates in an expressive, warm storytelling tone at a moderate pace. Very clear audio with no background noise.
```

Anchors to the built-in Sunita female voice and Narration emotion. Must include "very clear audio".
