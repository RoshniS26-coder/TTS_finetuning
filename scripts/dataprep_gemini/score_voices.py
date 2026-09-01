#!/usr/bin/env python3
"""Score TTS voice candidates by INTELLIGIBILITY.

Transcribe each generated clip with Sarvam Saaras STT and compute WER + CER against the
known input script. Lower = clearer pronunciation = the key objective metric for picking
a *teacher* voice (its mistakes get baked into the fine-tuned model). Also prints what the
ASR actually heard, so you can eyeball specific mispronunciations.

  venv/bin/python scripts/dataprep_gemini/score_voices.py \
      --clips-dir gemini_voice_final --ref-file story_texts/budbud_gagri_test.txt --language mr-IN
"""
from __future__ import annotations
import argparse, os, re, sys, time
from pathlib import Path
import requests
import jiwer

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.sarvam.ai/speech-to-text"
KEY_NAMES = ("SARVAM_API_KEY", "sarvam-api-key", "sarvam_api_key", "SARVAM_KEY")


def load_key(env: Path) -> str:
    for n in KEY_NAMES:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, val = line.partition("=")
            if k.strip() in KEY_NAMES:
                return val.strip().strip('"').strip("'")
    return ""


def normalize(t: str) -> str:
    t = re.sub(r"[।.,!?;:\"'()\-–—…]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _stt_one(wav: Path, key: str, language: str, mode: str = "codemix") -> str:
    headers = {"api-subscription-key": key}
    data = {"model": "saaras:v3", "language_code": language, "mode": mode}
    with wav.open("rb") as fh:
        files = {"file": (wav.name, fh, "audio/wav")}
        r = requests.post(API_URL, headers=headers, data=data, files=files, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    return (r.json().get("transcript") or "").strip()


def stt(wav: Path, key: str, language: str, mode: str = "codemix", max_s: float = 28.0) -> str:
    """Transcribe a wav of any length, auto-chunking to stay under Sarvam's 30 s cap."""
    import wave, tempfile
    with wave.open(str(wav)) as f:
        sr, ch, sw, n = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()
        dur = n / sr
        if dur <= max_s:
            return _stt_one(wav, key, language, mode)
        frames = f.readframes(n)
    per = int(max_s * sr) * ch * sw  # bytes per chunk
    parts = []
    for i in range(0, len(frames), per):
        seg = frames[i:i + per]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp = Path(tf.name)
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(ch); w.setsampwidth(sw); w.setframerate(sr); w.writeframes(seg)
        try:
            parts.append(_stt_one(tmp, key, language, mode))
        finally:
            tmp.unlink(missing_ok=True)
        time.sleep(0.4)
    return " ".join(p for p in parts if p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", type=Path, default=ROOT / "gemini_voice_final")
    ap.add_argument("--ref-file", type=Path, default=ROOT / "story_texts" / "budbud_gagri_test.txt")
    ap.add_argument("--language", default="mr-IN", help="Sarvam language_code (mr-IN / hi-IN)")
    ap.add_argument("--transcribe", type=Path, default=None,
                    help="transcribe ONE audio file (any length, auto-chunked) and exit; optionally --save-to")
    ap.add_argument("--save-to", type=Path, default=None, help="write the --transcribe result here")
    ap.add_argument("--env", type=Path, default=ROOT / ".env")
    args = ap.parse_args()

    key = load_key(args.env)
    if not key:
        sys.exit(f"ERROR: no Sarvam key in {args.env}")

    if args.transcribe:
        text = stt(args.transcribe, key, args.language)
        print(text)
        if args.save_to:
            args.save_to.write_text(text + "\n", encoding="utf-8")
            print(f"\n[saved -> {args.save_to}]", file=sys.stderr)
        return
    ref = normalize(args.ref_file.read_text(encoding="utf-8"))
    clips = sorted(args.clips_dir.glob("*.wav"))
    if not clips:
        sys.exit(f"ERROR: no wavs in {args.clips_dir}")

    print(f"Reference: {len(ref.split())} words | clips: {len(clips)} | lang: {args.language}\n")
    rows = []
    for wav in clips:
        try:
            hyp_raw = stt(wav, key, args.language)
        except Exception as exc:
            print(f"  {wav.name:<26} STT ERROR: {str(exc)[:80]}")
            continue
        hyp = normalize(hyp_raw)
        wer = jiwer.wer(ref, hyp)
        cer = jiwer.cer(ref, hyp)
        voice = wav.stem.split("_")[0]
        rows.append((voice, wer, cer, hyp_raw))
        print(f"  {voice:<12} WER {wer*100:5.1f}%  CER {cer*100:5.1f}%  | heard: {hyp_raw[:75]}…")
        time.sleep(0.5)

    rows.sort(key=lambda r: (r[1], r[2]))
    print("\n=== RANKED by intelligibility (lower = clearer pronunciation) ===")
    print(f"{'Voice':<12}{'WER%':>8}{'CER%':>8}")
    for v, w, c, _ in rows:
        print(f"{v:<12}{w*100:8.1f}{c*100:8.1f}")
    if rows:
        print(f"\n→ Clearest voice: {rows[0][0]} (WER {rows[0][1]*100:.1f}%, CER {rows[0][2]*100:.1f}%)")
    print("\nNOTE: WER absolute values for Marathi are noisy (ASR is imperfect); read them"
          " RELATIVELY across voices on identical text. Pair with your ear scorecard.")


if __name__ == "__main__":
    main()
