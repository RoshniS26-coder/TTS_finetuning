"""Compare TTS outputs: SQUIM-Objective + UTMOS (reference-free) + optional Sarvam WER.

For every wav under --dir (layout: <dir>/<model>/<clip>.wav) reports reference-free
scores, then a PER-MODEL MEANS summary:

  STOI    0-1    intelligibility            ] SQUIM-Objective
  PESQ    1-4.5  perceptual quality         ] (torchaudio, signal-quality)
  SI-SDR  dB     cleanliness / low distortion]
  UTMOS   1-5    predicted human MOS (naturalness)   — torch.hub tarepan/SpeechMOS
  WER     %      word error rate (LOWER=better)      — optional, --wer

SQUIM/UTMOS are reference-free (higher=better). WER is opt-in: it transcribes each
clip with Sarvam Saaras STT (codemix) and compares against the source text that was
synthesized, so it needs (a) SARVAM_API_KEY (env or .env) and (b) a --ref-dir of
<keyword>.txt reference texts. Clips are matched to a reference by shared keyword
(e.g. lion_finetuned.wav -> lion_story.txt); unmatched clips get WER = "-".

SQUIM needs 16 kHz mono -> we resample. Audio loaded with soundfile (not
torchaudio.load) to dodge the torchcodec/FFmpeg-8 issue.

  python scripts/narrate/compare_squim.py --dir model_outputs_v2
  python scripts/narrate/compare_squim.py --dir model_outputs_v2 --wer --ref-dir story_texts
"""
from __future__ import annotations
import argparse, glob, logging, math, os, re, time
from collections import defaultdict
from pathlib import Path

import numpy as np, soundfile as sf, torch, torchaudio
from torchaudio.functional import resample
from torchaudio.pipelines import SQUIM_OBJECTIVE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("squim")
SR = 16000
UTMOS_HUB = "tarepan/SpeechMOS:v1.2.0"               # standalone UTMOS22 port (no fairseq)

# --- Sarvam STT (mirrors scripts/phase1_dataprep/transcribe_saaras.py) ---
SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL, SARVAM_MODE = "saaras:v3", "codemix"
MAX_STT_SECONDS = 28          # Sarvam real-time STT rejects clips > 30 s -> chunk under that
SARVAM_KEY_NAMES = ("SARVAM_API_KEY", "sarvam-api-key", "sarvam_api_key", "SARVAM_KEY")
# tokens too generic to match a clip to a reference on their own:
GENERIC_TOKENS = {"story", "stories", "test", "narrate", "output", "sample", "clip",
                  "wav", "mr", "hi", "marathi", "hindi", "final", "fixed", "finetuned",
                  "pretrained", "indic", "parler", "gemini", "mms", "indicf5", "f5",
                  "after", "resegment", "v1", "v2"}


def load_16k_mono(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, ch)
    wav = torch.from_numpy(data).T                              # (ch, frames)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)                         # -> mono
    if sr != SR:
        wav = resample(wav, sr, SR)
    return wav                                                  # (1, frames) @ 16 kHz


def load_utmos():
    """Load the UTMOS22 predictor via torch.hub. Returns the model, or None on failure."""
    try:
        log.info("Loading UTMOS22 predictor (torch.hub: %s) ...", UTMOS_HUB)
        return torch.hub.load(UTMOS_HUB, "utmos22_strong", trust_repo=True).eval()
    except Exception as e:
        log.warning("UTMOS unavailable (%s) -> reporting without UTMOS.", type(e).__name__)
        return None


# ---------------- WER helpers (Sarvam STT + reference matching) ----------------
def load_sarvam_key(env_file: Path) -> str:
    for name in SARVAM_KEY_NAMES:
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, _, value = line.partition("=")
                if name.strip() in SARVAM_KEY_NAMES:
                    return value.strip().strip('"').strip("'")
    return ""


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower()))


def build_refs(ref_dir: Path) -> dict[str, tuple[set[str], str]]:
    """Map each reference .txt stem -> (its keyword tokens, its text)."""
    refs = {}
    for txt in sorted(ref_dir.glob("*.txt")):
        refs[txt.stem] = (tokens(txt.stem), txt.read_text(encoding="utf-8").strip())
    return refs


def match_reference(clip_stem: str, refs: dict) -> str | None:
    """Pick the reference whose keyword tokens best overlap the clip name.
    Requires at least one shared NON-generic token (so bare 'story' never matches)."""
    ct = tokens(clip_stem)
    best, best_score = None, 0
    for stem, (rt, _text) in refs.items():
        overlap = ct & rt
        distinctive = overlap - GENERIC_TOKENS
        if not distinctive:
            continue
        score = len(distinctive) * 10 + len(overlap)
        if score > best_score:
            best, best_score = stem, score
    return best


def normalize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation (incl Devanagari danda), keep Devanagari/Latin word chars."""
    return re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE).split()


def word_error_rate(ref: str, hyp: str) -> float:
    r, h = normalize_words(ref), normalize_words(hyp)
    if not r:
        return math.nan
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / len(r)


def sarvam_transcribe(wav_path: str, key: str, language: str, retries: int = 4) -> str:
    import requests
    headers = {"api-subscription-key": key}
    data = {"model": SARVAM_MODEL, "language_code": language, "mode": SARVAM_MODE}
    for attempt in range(1, retries + 1):
        try:
            with open(wav_path, "rb") as fh:
                resp = requests.post(SARVAM_URL, headers=headers, data=data,
                                     files={"file": (os.path.basename(wav_path), fh, "audio/wav")}, timeout=120)
            if resp.status_code == 200:
                return (resp.json().get("transcript") or "").strip()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 20)); continue
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160]}")
        except Exception as exc:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("unreachable")


def sarvam_transcribe_clip(wav_path: str, key: str, language: str) -> str:
    """Transcribe a clip of any length: single request if <=28 s, else split into
    <=28 s windows (Sarvam real-time STT caps at 30 s) and concatenate the parts.
    Fixed-window splits may clip a word at each boundary — fine for relative WER
    since every model's audio is chunked identically."""
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    if len(data) / sr <= MAX_STT_SECONDS:
        return sarvam_transcribe(wav_path, key, language)
    import tempfile
    win = int(MAX_STT_SECONDS * sr)
    parts = []
    with tempfile.TemporaryDirectory() as td:
        for k, start in enumerate(range(0, len(data), win)):
            seg = data[start:start + win]
            seg_path = os.path.join(td, f"seg_{k:03d}.wav")
            sf.write(seg_path, seg, sr)
            parts.append(sarvam_transcribe(seg_path, key, language))
    return " ".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="model_outputs_v1",
                    help="folder of <model>/<clip>.wav outputs to score (e.g. model_outputs_v2)")
    ap.add_argument("--no-utmos", action="store_true", help="skip UTMOS (SQUIM only)")
    ap.add_argument("--wer", action="store_true", help="add a WER column via Sarvam STT (needs SARVAM_API_KEY + --ref-dir)")
    ap.add_argument("--ref-dir", type=Path, default=Path("story_texts"), help="dir of <keyword>.txt reference texts for WER")
    ap.add_argument("--stt-language", default="mr-IN", help="Sarvam language_code for WER (mr-IN / hi-IN / unknown)")
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    args = ap.parse_args()

    paths = sorted(glob.glob(f"{args.dir}/*/*.wav"))
    if not paths:
        raise SystemExit(f"no wavs under {args.dir}/*/")

    log.info("Loading SQUIM-Objective model ...")
    squim = SQUIM_OBJECTIVE.get_model().eval()
    utmos = None if args.no_utmos else load_utmos()
    have_utmos = utmos is not None

    # ---- optional WER setup ----
    do_wer, refs, sarvam_key = False, {}, ""
    if args.wer:
        sarvam_key = load_sarvam_key(args.env_file)
        if not sarvam_key:
            log.error("--wer requested but SARVAM_API_KEY not found in env or %s", args.env_file); raise SystemExit(1)
        if not args.ref_dir.is_dir():
            log.error("--wer requested but --ref-dir %s not found", args.ref_dir); raise SystemExit(1)
        refs = build_refs(args.ref_dir)
        if not refs:
            log.error("no *.txt reference texts in %s", args.ref_dir); raise SystemExit(1)
        do_wer = True
        log.info("WER on: Sarvam STT (lang=%s), %d reference texts from %s", args.stt_language, len(refs), args.ref_dir)

    # rows: (model, clip, stoi, pesq, sisdr, utmos, wer)   — utmos/wer are nan when unavailable
    rows = []
    for p in paths:
        model_name = os.path.basename(os.path.dirname(p))
        clip = os.path.basename(p)
        wav = load_16k_mono(p)
        with torch.no_grad():
            stoi, pesq, sisdr = squim(wav)
            mos = float(utmos(wav, SR)) if have_utmos else math.nan
        wer = math.nan
        if do_wer:
            ref_stem = match_reference(Path(clip).stem, refs)
            if ref_stem is None:
                log.warning("  no reference matched for %s -> WER skipped", clip)
            else:
                try:
                    hyp = sarvam_transcribe_clip(p, sarvam_key, args.stt_language)
                    wer = word_error_rate(refs[ref_stem][1], hyp)
                except Exception as exc:
                    log.warning("  STT failed for %s: %s -> WER skipped", clip, exc)
        rows.append((model_name, clip, float(stoi), float(pesq), float(sisdr), mos, wer))
        log.info("  %-22s %-24s STOI=%.3f PESQ=%.3f SI-SDR=%6.2f UTMOS=%s WER=%s",
                 model_name, clip[:24], float(stoi), float(pesq), float(sisdr),
                 f"{mos:.3f}" if mos == mos else " - ", f"{wer*100:5.1f}%" if wer == wer else "  -  ")

    umos_h = f"{'UTMOS↑':>9}" if have_utmos else f"{'UTMOS':>9}"
    wer_h = f"{'WER%↓':>8}" if do_wer else ""

    # ---- PER-MODEL MEANS (headline) ----
    agg = defaultdict(list)
    for m, _c, stoi, pesq, sisdr, mos, wer in rows:
        agg[m].append((stoi, pesq, sisdr, mos, wer))
    means = []
    for m, vals in agg.items():
        a = np.array(vals, dtype=float)
        means.append((m, len(vals), float(np.nanmean(a[:, 0])), float(np.nanmean(a[:, 1])),
                      float(np.nanmean(a[:, 2])), float(np.nanmean(a[:, 3])),
                      float(np.nanmean(a[:, 4])) if do_wer else math.nan))
    means.sort(key=lambda r: (r[5] if have_utmos and r[5] == r[5] else r[3]), reverse=True)

    def wer_cell(w, width=8):
        return f"{w*100:>{width-1}.1f}%" if (do_wer and w == w) else (f"{'-':>{width}}" if do_wer else "")

    print("\n" + "=" * 86)
    print("PER-MODEL MEANS" + ("  (ranked by UTMOS)" if have_utmos else "  (ranked by PESQ)"))
    print("-" * 86)
    print(f"{'model':<22}{'n':>4}{'STOI↑':>9}{'PESQ↑':>9}{'SI-SDR↑':>11}{umos_h}{wer_h}")
    print("-" * 86)
    for m, n, stoi, pesq, sisdr, mos, wer in means:
        mos_s = f"{mos:>9.3f}" if mos == mos else f"{'-':>9}"
        print(f"{m:<22}{n:>4}{stoi:>9.3f}{pesq:>9.3f}{sisdr:>11.2f}{mos_s}{wer_cell(wer)}")
    print("=" * 86)

    # ---- PER-CLIP DETAIL (sorted by PESQ desc) ----
    rows.sort(key=lambda r: r[3], reverse=True)
    print("\nPER-CLIP DETAIL")
    print("-" * 86)
    print(f"{'model':<18}{'clip':<22}{'STOI↑':>8}{'PESQ↑':>8}{'SI-SDR↑':>10}{umos_h}{wer_h}")
    print("-" * 86)
    for m, clip, stoi, pesq, sisdr, mos, wer in rows:
        mos_s = f"{mos:>9.3f}" if mos == mos else f"{'-':>9}"
        print(f"{m:<18}{clip[:21]:<22}{stoi:>8.3f}{pesq:>8.3f}{sisdr:>10.2f}{mos_s}{wer_cell(wer)}")
    print("=" * 86)
    print("STOI 0-1 | PESQ 1.0-4.5 | SI-SDR dB | UTMOS 1-5 -> higher=better." +
          ("  WER% -> LOWER=better." if do_wer else "") +
          ("" if have_utmos else "  [UTMOS skipped]"))


if __name__ == "__main__":
    main()
