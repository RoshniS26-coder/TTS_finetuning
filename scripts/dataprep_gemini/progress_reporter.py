#!/usr/bin/env python3
"""Standalone TTS-synthesis progress reporter -> Slack incoming webhook.

Fully independent of Claude Code / Cursor: run it detached and it will, every INTERVAL:
  1. count Marathi + Hindi wavs, compute clips/min (last 10 min) and ETA,
  2. post a progress message to the Slack webhook (key 'slack' in .env),
  3. auto-launch Hindi synthesis (detached) once Marathi hits its target,
  4. post a final "done" message and exit once BOTH are complete.

  nohup venv/bin/python scripts/dataprep_gemini/progress_reporter.py > logs/reporter.log 2>&1 &
"""
from __future__ import annotations
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERVAL = 1800            # 30 min
MR_TARGET, HI_TARGET = 1770, 1510  # Marathi 1770: chunk_00159 intentionally dropped (bad pronunciation)
MR_DIR = ROOT / "tts_data" / "marathi" / "audio"
HI_DIR = ROOT / "tts_data" / "hindi" / "audio"
HI_UNITS = ROOT / "tts_corpus" / "hindi_units.tsv"
HI_OUT = ROOT / "tts_data" / "hindi"


def load_key(name: str) -> str:
    env = ROOT / ".env"
    for line in (env.read_text(encoding="utf-8").splitlines() if env.exists() else []):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line and name in line.split("=", 1)[0].lower():
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def post(webhook: str, text: str) -> None:
    if not webhook:
        print(text); return
    try:
        req = urllib.request.Request(
            webhook, data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15)
    except Exception as exc:
        print(f"[slack post failed] {exc}\n{text}")


def count(d: Path) -> int:
    return len(list(d.glob("chunk_*.wav"))) if d.exists() else 0


def rate_per_min(d: Path, window=600) -> float:
    if not d.exists():
        return 0.0
    now = time.time()
    n = sum(1 for f in d.glob("chunk_*.wav") if now - f.stat().st_mtime <= window)
    return n / (window / 60)


def eta_str(remaining: int, rpm: float) -> str:
    if rpm <= 0:
        return "stalled?"
    mins = remaining / rpm
    return f"~{mins/60:.1f}h ({mins:.0f} min)"


def hindi_running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "synthesize_tts.py.*hindi"],
                             capture_output=True, text=True).stdout.strip()
        return bool(out)
    except Exception:
        return False


def launch_hindi() -> None:
    log = open(ROOT / "logs" / "synth_hindi.log", "a")
    subprocess.Popen(
        [str(ROOT / "venv" / "bin" / "python"),
         str(ROOT / "scripts" / "dataprep_gemini" / "synthesize_tts.py"),
         "--units-tsv", str(HI_UNITS), "--out-dir", str(HI_OUT), "--voice", "Sulafat"],
        cwd=str(ROOT), stdout=log, stderr=log, start_new_session=True)


def main() -> None:
    webhook = load_key("webhook") or load_key("slack")
    (ROOT / "logs").mkdir(exist_ok=True)
    post(webhook, ":wave: TTS progress reporter started. Marathi first, then Hindi (Gemini Pro / Sulafat).")
    while True:
        mr, hi = count(MR_DIR), count(HI_DIR)
        mr_done = mr >= MR_TARGET
        hi_done = hi >= HI_TARGET

        if mr_done and not hi_done and not hindi_running():
            launch_hindi()
            post(webhook, f":white_check_mark: *Completed Marathi* ({mr}/{MR_TARGET} clips). Now starting Hindi synthesis…")

        if mr_done and hi_done:
            post(webhook, f":white_check_mark: *Completed Hindi* ({hi}/{HI_TARGET} clips).  :tada: Both languages done — "
                          f"Marathi {mr} + Hindi {hi}. Dataset ready to build; metadata.csv rebuilt in tts_data/marathi + tts_data/hindi.")
            return

        if not mr_done:
            rpm = rate_per_min(MR_DIR)
            msg = (f":microphone: *Marathi* {mr}/{MR_TARGET}  ({100*mr/MR_TARGET:.0f}%)  "
                   f"| {rpm:.1f} clips/min | ETA {eta_str(MR_TARGET-mr, rpm)}")
        else:
            rpm = rate_per_min(HI_DIR)
            msg = (f":microphone: Marathi ✅ done | *Hindi* {hi}/{HI_TARGET}  ({100*hi/HI_TARGET:.0f}%)  "
                   f"| {rpm:.1f} clips/min | ETA {eta_str(HI_TARGET-hi, rpm)}")
        post(webhook, msg)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
