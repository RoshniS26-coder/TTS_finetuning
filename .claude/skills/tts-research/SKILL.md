---
name: tts-research
description: >
  Research and report on Parler-TTS / Indic-Parler fine-tuning for this repo. Answers TTS
  fine-tuning questions (background music, multi-speaker, data amount, epochs, multilingual,
  evaluation/benchmarking vs Gemini-TTS) with verified, cited sources; pulls live W&B
  training metrics for the marathi-parler run; and compiles a self-contained HTML report
  with diagrams, tables, and references under reports/. Trigger when the user asks to
  research a TTS fine-tuning question, benchmark against Gemini, analyse a training run, or
  (re)generate the fine-tuning research report. Also triggers on "/tts-research".
---

# tts-research — TTS fine-tuning research + report agent

A focused research agent for **fine-tuning `ai4bharat/indic-parler-tts` for storytelling**.
Goal context: approach Gemini-TTS quality at a fraction of the cost, across Marathi and
other Indian languages. It produces a cited HTML report and can fold in live training
metrics from Weights & Biases.

## When to use
- Answering TTS fine-tuning questions (data cleanliness, multi-speaker captions, hours of
  data, epochs, per-language datasets, evaluation metrics, Gemini benchmarking).
- Analysing a RunPod training run logged to W&B (`run_name: marathi-parler-v2`).
- (Re)generating `reports/tts-finetuning-research_<date>.html`.

## Repo grounding (read these first, don't re-derive)
- `CLAUDE.md` — architecture facts: Parler-TTS is autoregressive over **DAC codec tokens**,
  conditioned on a **frozen Flan-T5** encoder; objective is **cross-entropy** (NO mel/GAN
  loss). Voice/emotion ride on the **description caption**; "very clear audio" triggers
  high fidelity; Marathi named speakers are **Sunita / Radha / Isha**.
- `marathi_config.json` — the live training config (epochs, lr 8e-5 cosine, bf16,
  freeze_text_encoder, batch 2 × grad-accum 4).
- `scripts/narrate/compare_squim.py` — SQUIM STOI/PESQ/SI-SDR scorer (relative comparator).
- `scripts/narrate/narrate_gemini.py` — Gemini-TTS baseline (Kore voice, 24 kHz) for benchmarks.
- `.env` — `wandb_apikey`, `gemini-key`, `sarvam-api-key`, `huggingface-cli` (never print values).

## Subagents (in this repo, `.claude/agents/`)
- **`wandb-metrics-fetcher`** — read-only pull of the W&B run history (train/eval loss, lr,
  WER, CLAP), summary, state, config, URL. Returns a Markdown table + JSON for charts.
  Entity `roshnis-betacraft`, project `parler-speech`. NOTE: the config `run_name` does not
  currently reach W&B, so it matches by config fingerprint and may report an auto-named run.
- **`tts-report-generator`** — compiles the self-contained HTML report (inline-SVG diagrams,
  tables, references; offline-friendly; prints cleanly) under `reports/`.

## Workflow
1. **Scope.** Confirm which questions to answer and whether to include a live W&B pull.
2. **Research (parallel).** Verify each claim with web search and collect real citations.
   Prefer: Parler-TTS paper (Lyth & King, arXiv 2402.01912) + repo/training README, DAC
   (arXiv 2306.06546), DataSpeech, indic-parler-tts model card, TorchAudio-SQUIM, UTMOS,
   Gemini-TTS docs, multilingual/PEFT TTS literature. Every section = **VERDICT → reasoning
   → references**. Be numeric and defensible; flag engineering recommendations vs documented
   defaults.
3. **Metrics.** Spawn `wandb-metrics-fetcher` for `marathi-parler-v2`. Handle "not started"
   gracefully (it will list existing runs instead).
4. **Compile.** Hand the brief + metrics block to `tts-report-generator` →
   `reports/tts-finetuning-research_<date>.html`.
5. **Report back.** Summarise key findings + action items (don't dump the whole report).

## Default question set (the canonical brief)
1. Background music in training data — does Parler need clean audio, what leaks, why
   (DAC encodes the whole waveform; the "very clear audio" tag is derived from measured
   SNR/PESQ, so caption and cleanliness are causally coupled).
2. Fine-tuning theory + the hyperparameter table (what's trained vs frozen; LoRA/PEFT option).
3. Multi-speaker: name-anchored captions separate voices; unlabeled mixing → averaged/muddy
   voice; Sunita/Radha/Isha **already exist** in the base → name-collision/overwrite risk.
4. Data amount & epochs: 5–6 h/speaker is generous; start 3–5 epochs; stop when eval/loss
   rises while train/loss falls.
5. Per-language datasets: YES — phonetics/prosody/script; prefer one balanced multilingual
   fine-tune over sequential (catastrophic forgetting).
6. Single-narrator data & how Gemini differs (caption-conditioned multi-speaker; Gemini =
   massive proprietary multilingual pretraining; be honest about scope).
7. Evaluation & Gemini benchmark: SQUIM (STOI/PESQ/SI-SDR) + UTMOS/MOS + WER/CER + speaker
   cosine; same-text Gemini-vs-Parler table; how to read W&B loss curves.

## Guardrails
- Read-only on W&B and `.env`; never print secrets.
- Never invent citations — carry verified URLs through to the report.
- Inputs are facts as of generation time; if a fact names a file/flag, confirm it still
  exists before recommending it.
