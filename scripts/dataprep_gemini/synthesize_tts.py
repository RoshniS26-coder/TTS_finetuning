#!/usr/bin/env python3
"""Synthesize a units TSV into a Parler-ready dataset with Gemini TTS (Sulafat voice).

Reads tts_corpus/<lang>_units.tsv (idx, source, text) -> <out-dir>/audio/chunk_NNNNN.wav + metadata.csv.
Each unit is a complete-sentence 2-20s chunk -> one clip -> a perfectly-aligned (audio, text) pair.

Uses the FINALISED COMPACT storyteller prompt (chosen by ear for consistent training audio) — NOT the
app's detailed Prompt-4 directive. Resumable (skips clips already on disk) and retries on 429/5xx.

  venv/bin/python scripts/dataprep_gemini/synthesize_tts.py \
      --units-tsv tts_corpus/marathi_units.tsv --out-dir tts_data/marathi --voice Sulafat
"""
from __future__ import annotations
import argparse, os, sys, time, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# COMPACT prompt — the one chosen by listening (consistent, not theatrical). {text} appended.
STYLE = ("You are a warm, cheerful children's storyteller narrating a story for kids aged 2 to 12. "
         "Speak in a sweet, animated, expressive voice with a lively but clear pace, gentle warmth, "
         "and playful energy. Pronounce every word clearly. Narrate this story:")
GEMINI_SR = 24000  # Gemini TTS returns 24kHz 16-bit mono PCM


def load_key(env: Path) -> str:
    for line in (env.read_text(encoding="utf-8").splitlines() if env.exists() else []):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line and "gemini" in line.split("=", 1)[0].lower():
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GEMINI_API_KEY", "")


def write_wav(path: Path, pcm: bytes, rate: int = GEMINI_SR) -> float:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return len(pcm) / 2 / rate


QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "quota")


def is_quota_error(msg: str) -> bool:
    return any(m.lower() in msg.lower() for m in QUOTA_MARKERS)


def synth(client, model, voice, text, max_retries=5):
    from google.genai import types
    prompt = f"{STYLE}\n\n{text}"
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
                ),
            )
            cand = resp.candidates[0]
            return cand.content.parts[0].inline_data.data
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            wait = min(2 ** attempt, 30)
            if any(c in msg for c in ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "UNAVAILABLE")):
                time.sleep(wait); continue
            if attempt < max_retries:
                time.sleep(wait); continue
            raise
    # Retries exhausted — surface the REAL last error (quota, etc.), not a generic message,
    # and tag quota exhaustion explicitly so it is not mistaken for a hang.
    detail = str(last_exc) if last_exc else "unknown error"
    kind = "QUOTA EXHAUSTED" if last_exc and is_quota_error(detail) else "failed"
    raise RuntimeError(f"{kind} after {max_retries} retries: {detail}") from last_exc


def read_units(tsv: Path):
    rows = []
    for i, line in enumerate(tsv.read_text(encoding="utf-8").splitlines()):
        if i == 0 and line.startswith("idx\t"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].strip():
            rows.append((int(parts[0]), parts[2].strip()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-tsv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--voice", default="Sulafat")
    ap.add_argument("--model", default="gemini-2.5-pro-preview-tts")
    ap.add_argument("--limit", type=int, default=0, help="only synth first N (0 = all); for smoke tests")
    ap.add_argument("--sleep", type=float, default=0.6, help="seconds between calls (rate-limit friendly)")
    ap.add_argument("--env", type=Path, default=ROOT / ".env")
    args = ap.parse_args()

    from google import genai
    key = load_key(args.env)
    if not key:
        sys.exit(f"no Gemini key in {args.env}")
    client = genai.Client(api_key=key)

    rows = read_units(args.units_tsv)
    if args.limit:
        rows = rows[:args.limit]
    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    quota_streak = 0  # consecutive quota-exhausted chunks → daily cap almost certainly hit
    t0 = time.time()
    for idx, text in rows:
        wav = audio_dir / f"chunk_{idx:05d}.wav"
        if wav.exists() and wav.stat().st_size > 1000:
            skipped += 1; continue
        try:
            pcm = synth(client, args.model, args.voice, text)
            dur = write_wav(wav, pcm)
            done += 1
            quota_streak = 0
            if done % 25 == 0 or done <= 3:
                el = time.time() - t0
                print(f"  [{done}] chunk_{idx:05d} {dur:.1f}s  (skipped {skipped}, {el/60:.1f}min elapsed)", flush=True)
        except Exception as exc:
            failed += 1
            msg = str(exc)
            print(f"  ERR chunk_{idx:05d}: {msg[:120]}", flush=True)
            if is_quota_error(msg):
                quota_streak += 1
                if quota_streak >= 3:
                    el = time.time() - t0
                    print(
                        f"\n  ABORT: {quota_streak} consecutive quota-exhausted chunks — daily API "
                        f"quota is almost certainly hit. Stopping so we don't burn hours failing.\n"
                        f"  Progress: {done} synthesized this run, {skipped} already on disk, "
                        f"{failed} failed, {el/60:.1f}min elapsed.\n"
                        f"  Resume after the quota resets (Gemini free tier: midnight Pacific) by "
                        f"re-running this exact command — it skips wavs already on disk.",
                        flush=True,
                    )
                    break
            else:
                quota_streak = 0
        time.sleep(args.sleep)

    # (re)build metadata.csv from all clips present on disk
    meta = args.out_dir / "metadata.csv"
    text_by_idx = {i: t for i, t in read_units(args.units_tsv)}
    lines = []
    for wav in sorted(audio_dir.glob("chunk_*.wav")):
        idx = int(wav.stem.split("_")[1])
        if idx in text_by_idx:
            lines.append(f"audio/{wav.name}|{text_by_idx[idx]}")
    meta.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"\nDone. synthesized {done}, skipped(existing) {skipped}, failed {failed}")
    print(f"  clips on disk: {len(lines)} | metadata -> {meta}")
    if failed:
        print("  NOTE: re-run the same command to retry failed clips (resumable).")


if __name__ == "__main__":
    main()
