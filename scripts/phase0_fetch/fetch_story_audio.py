#!/usr/bin/env python3
"""Reusable YouTube -> clean training-audio pipeline for Marathi story TTS.

One command per genre. Given a list of YouTube / YT-Music URLs (single videos or
playlists) it will:
  1. download audio-only (yt-dlp, wav)            -> .yt_work/<out>/raw/
  2. separate vocals from background music         -> .yt_work/<out>/separated/   (demucs htdemucs_ft, optional)
  3. convert to mono @ target SR + light cleanup   -> <out>/                       (final training audio)
and write <out>/manifest.csv with per-file durations + a total.

Examples
--------
  # whole Panchatantra playlist, separate music, light denoise:
  python fetch_story_audio.py panchtantra-audio --urls-file url_lists/panchtantra.txt

  # sources that already have NO background music -> skip demucs (faster):
  python fetch_story_audio.py pauranik-katha-audio --urls-file url_lists/pauranik-katha.txt --no-separate

  # quick end-to-end validation on just the first track of a playlist:
  python fetch_story_audio.py panchtantra-audio --urls-file url_lists/panchtantra.txt --limit 1

Run from the TTS_finetuning repo root with this repo's venv python
(e.g. `venv/bin/python scripts/phase0_fetch/fetch_story_audio.py ...`).
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("fetch")

# ffmpeg audio-filter chains. Kept conservative on purpose: aggressive denoise
# adds artefacts that hurt TTS training more than a little hiss does.
DENOISE_CHAINS = {
    "none":  "",                                                  # mono/resample only
    "light": "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11",       # rumble cut + level match
    "strong": "highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",  # + FFT denoise
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="final folder name for training audio, e.g. panchtantra-audio")
    src = p.add_argument_group("sources (give at least one)")
    src.add_argument("--urls", nargs="+", default=[], help="one or more YouTube/YT-Music URLs")
    src.add_argument("--urls-file", type=Path, help="text file, one URL per line (# comments ok)")

    p.add_argument("--separate", dest="separate", action="store_true", default=True,
                   help="run demucs to remove background music (default)")
    p.add_argument("--no-separate", dest="separate", action="store_false",
                   help="skip demucs (use for sources that already have no music)")
    p.add_argument("--denoise", choices=list(DENOISE_CHAINS), default="light",
                   help="post-cleanup strength (default: light)")
    p.add_argument("--sr", type=int, default=44100, help="target sample rate (default 44100)")
    p.add_argument("--demucs-model", default="htdemucs_ft", help="demucs model (default htdemucs_ft)")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                   help="demucs device (default cpu; mps is flaky on macOS)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap items downloaded per playlist URL (0 = no cap; use 1 to validate)")
    p.add_argument("--workdir", type=Path, default=Path(".yt_work"),
                   help="scratch dir for raw + separated audio (default .yt_work)")
    p.add_argument("--no-playlist", dest="playlist", action="store_false", default=True,
                   help="treat URLs with &list= as a single video, not the whole playlist")
    return p.parse_args()


def collect_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.urls)
    if args.urls_file:
        if not args.urls_file.exists():
            log.error("urls-file not found: %s", args.urls_file)
            sys.exit(1)
        for line in args.urls_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        log.error("No URLs given. Use --urls or --urls-file.")
        sys.exit(1)
    # de-dup, preserve order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def safe_stem(name: str) -> str:
    """Path-safe, readable stem (keeps Devanagari, drops shell-hostile chars)."""
    name = re.sub(r"[\\/:*?\"<>|]+", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:120] or "track"


def download(urls: list[str], raw_dir: Path, args: argparse.Namespace) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ".download-archive.txt"   # lets re-runs skip already-fetched items
    for url in urls:
        log.info("⬇️  downloading: %s", url)
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-x", "--audio-format", "wav", "--audio-quality", "0",
            "--extractor-args", "youtube:player_client=android,web",
            "--ignore-errors", "--no-overwrites",
            "--download-archive", str(archive),
            "-o", str(raw_dir / "%(playlist_index)s-%(title)s.%(ext)s"),
            "--yes-playlist" if args.playlist else "--no-playlist",
            url,
        ]
        if args.limit:
            cmd[-2:-2] = ["--playlist-end", str(args.limit)]
        subprocess.run(cmd, check=False)   # check=False: skip dead items, keep going


def separate(raw_files: list[Path], sep_dir: Path, args: argparse.Namespace) -> None:
    sep_dir.mkdir(parents=True, exist_ok=True)
    log.info("🎙️  demucs (%s) on %d file(s) [device=%s]", args.demucs_model, len(raw_files), args.device)
    for af in raw_files:
        out_track = sep_dir / args.demucs_model / af.stem / "vocals.wav"
        if out_track.exists():
            log.info("   ⏭️  vocals exist: %s", af.name); continue
        cmd = [sys.executable, "-m", "demucs", "-n", args.demucs_model, "--two-stems=vocals",
               "-d", args.device, "-o", str(sep_dir), str(af)]
        try:
            subprocess.run(cmd, check=True)
            log.info("   ✅ separated: %s", af.name)
        except subprocess.CalledProcessError as e:
            log.warning("   ⚠️  demucs failed on %s: %s", af.name, e)


def postprocess(src_files: list[Path], final_dir: Path, args: argparse.Namespace) -> list[tuple[str, float]]:
    final_dir.mkdir(parents=True, exist_ok=True)
    af_chain = DENOISE_CHAINS[args.denoise]
    rows: list[tuple[str, float]] = []
    for sf in src_files:
        # name final file after the original track (demucs nests as <track>/vocals.wav)
        track = sf.parent.name if sf.name == "vocals.wav" else sf.stem
        out_path = final_dir / f"{safe_stem(track)}.wav"
        if not out_path.exists():
            cmd = ["ffmpeg", "-y", "-i", str(sf), "-ac", "1", "-ar", str(args.sr)]
            if af_chain:
                cmd += ["-af", af_chain]
            cmd.append(str(out_path))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                log.warning("   ⚠️  ffmpeg failed on %s: %s", sf.name, r.stderr.strip().splitlines()[-1:])
                continue
        dur = audio_duration(out_path)
        rows.append((out_path.name, dur))
        log.info("   → %s  (%.1fs)", out_path.name, dur)
    return rows


def audio_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    urls = collect_urls(args)

    work = args.workdir / args.out
    raw_dir, sep_dir = work / "raw", work / "separated"
    final_dir = Path(args.out)

    log.info("=== %s | %d URL(s) | separate=%s | denoise=%s | sr=%d ===",
             args.out, len(urls), args.separate, args.denoise, args.sr)

    download(urls, raw_dir, args)
    raw_files = sorted(f for f in raw_dir.iterdir()
                       if f.suffix.lower() in (".wav", ".mp3", ".m4a", ".opus"))
    if not raw_files:
        log.error("Nothing downloaded into %s — check the URLs / yt-dlp.", raw_dir); sys.exit(1)
    log.info("downloaded %d raw file(s)", len(raw_files))

    if args.separate:
        separate(raw_files, sep_dir, args)
        src_files = sorted(sep_dir.rglob("vocals.wav"))
        if not src_files:
            log.error("demucs produced no vocals.wav — check separation."); sys.exit(1)
    else:
        src_files = raw_files

    rows = postprocess(src_files, final_dir, args)

    manifest = final_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["file", "duration_sec"]); w.writerows(rows)
    total = sum(d for _, d in rows)
    log.info("✅ %d clip(s) in %s | total %.1f min (%.2f hr) | manifest: %s",
             len(rows), final_dir, total / 60, total / 3600, manifest)
    log.info("⚠️  NEXT: LISTEN and delete any watery/metallic/music-leak files before segmenting.")


if __name__ == "__main__":
    main()
