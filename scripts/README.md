# scripts/

Organised by pipeline stage. **Run all scripts from the repo root** (`TTS_finetuning/`)
— they use paths relative to the current directory (e.g. `audio_data/`, `marathi_dataset_v2/`),
so `cd` into a subfolder will break those defaults.

```
python scripts/phase1_dataprep/segment_audio.py        # ✅ from repo root
```

## phase0_fetch/ — source-audio fetch (Phase 0, before data prep)
Reusable YouTube → clean training-audio pipeline (used for the pre-v7 scraped datasets;
v7 pivoted to synthetic Gemini-TTS audio, so this is kept for reference / re-use).
- `fetch_story_audio.py` — yt-dlp download → demucs music separation → mono WAV + `manifest.csv`
- `demucs_yt.py`, `yt_to_audio.py` — lower-level download / separation helpers
- `test_indic_parler.py` — Indic-Parler smoke test · `mystory.txt` — sample narration text
- `url_lists/` — per-genre YouTube URL lists (panchtantra, lok-katha, bedtime stories, …)

## phase1_dataprep/ — local data prep (Phase 1, before RunPod)
Runs on the laptop (CPU). Produces the dataset RunPod trains on.
- `segment_audio.py`   — slice raw `audio_data/` into 2–20s clips → `marathi_dataset_v2/audio/`
- `transcribe.py`      — faster-whisper transcription → `metadata.csv`
- `transcribe_saaras.py` — Sarvam Saaras (codemix) transcription alternative
- `review_numbers.py`  — audio-in-the-loop number/spoken-form correction of `metadata.csv`
- `prepare_dataset.py` — build the HF dataset → `marathi_parler_ds/`

## phase2_runpod/ — training + transfer (Phase 2, on/with RunPod)
- `upload.sh`        — push `marathi_parler_ds_v2.zip` + `marathi_config.json` (both at repo root)
                       plus `runpod_setup.sh` + `test_inference.py` to the pod
- `runpod_setup.sh`  — install deps + launch training on the pod (`/workspace`)
- `test_inference.py`— inference smoke-test on the pod after training
- `connect.sh`       — ssh helper
- `download.sh`      — zip + pull the trained model back to repo root

## narrate/ — narration / TTS inference across models
- `narrate_story.py`   — Indic Parler-TTS (pretrained or fine-tuned, via `--model-dir`)
- `run_finetune.py`    — narrate with the published fine-tuned model
- `narrate_indicf5.py` — IndicF5
- `narrate_mms.py`     — MMS
- `narrate_gemini.py`  — Gemini TTS
- `compare_squim.py`   — SQuIM audio-quality scoring of narration outputs
- `run_all_rabbit.sh`  — narrate one story across all 4 models (sequential)

Narration input texts live in `story_texts/` (e.g. `story_texts/lion_story.txt`); run logs in `logs/`.

Stays at repo root: data dirs (`audio_data/`, `marathi_dataset_v2/`, `marathi_parler_ds/`, `stories/`),
`story_texts/`, `logs/`, `marathi_config.json`, `marathi_parler_ds_v{1,2}.zip`, `runpodctl`, docs, `venv/`.
```
