"""Measure TTS reliability per language: generate a probe set, then score BY EAR.

WHY THIS EXISTS. On 2026-09-03 the app produced audible failures — a word skipped
("ढप्प!" not spoken), a word stalled ("रोहन" → "Rooooohan") — that re-running the
SAME text at seed 42 did NOT reproduce: the line came back clean. So the defect is
not in the text, the segmentation, or the model's ability to say these words. It is
STOCHASTIC: `do_sample: True` at temperature 0.65 draws fresh each generation, and
occasionally the decoder's attention slips off the text and it either stalls on a
phoneme or skips past one.

That means the honest measurement is a FAILURE RATE, not a pass/fail — and the way
to get one is to generate each probe at several seeds and listen.

WHAT WAS TRIED AND REJECTED as an automatic scorer:
  * DURATION. Measured: the sentences that failed in the app came back at
    0.81-1.33x their expected length, i.e. inside the normal band. A skip masked by
    a stall leaves total duration unchanged, so duration cannot see it.
  * ASR/WER via a speech-to-text service. Rejected by the project owner. (The first
    attempt also used Sarvam's speech-to-text-TRANSLATE endpoint by mistake, which
    returns ENGLISH for Devanagari audio and produced a bogus 100% defect rate that
    listening immediately contradicted — a good reminder that an automatic scorer
    is itself a thing that can be silently wrong.)

So the ear is the ground truth. This script's job is to make listening FAST and to
turn what you hear into a table you can paste into a report.

USAGE
    # 1. generate the probe set (GPU; ~$0.001 per clip)
    venv/bin/python scripts/narrate/score_quality.py --all --runs 2

    # 2. listen and score — plays each clip, one keypress per verdict
    venv/bin/python scripts/narrate/score_quality.py --listen quality_runs/<stamp>

    # 3. the run directory then holds report.json + REPORT.md
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
import urllib.request
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "5tk0794i74t8lh"
JAANTEHO_ENV = Path("/Users/roshnisundrani/jaanteho-fresh/.env")

# Held-out probe sentences. Deliberately INCLUDES the constructs that failed in the
# app (onomatopoeia, an ellipsis countdown, comma-appositive names) alongside plain
# narration, so the report shows whether those are genuinely riskier or just where
# the dice happened to land. `what` is what the listener should check for.
PROBES: dict[str, list[tuple[str, str]]] = {
    "mr": [
        ("ढप्प, तो डबा जमिनीवर पडला.", "is ढप्प spoken at all?"),
        ("रोहन चिन्मयवर खूप रागावला आणि त्याने त्याचा लाल कपडा पटकन ओढून घेतला.", "रोहन stretched into Roooohan?"),
        ("तेवढ्यात त्यांचा छोटा चुलत भाऊ, चिन्मय, तिथे आला.", "भाऊ / चिन्मय stretched?"),
        ("त्याने डोळे मिटले आणि जोरात मोजायला सुरुवात केली, तीन, दोन, एक, झूम!", "all four counted: 3, 2, 1, झूम?"),
        ("गरमागरम पोळ्यांचा खमंग वास सगळीकडे दरवळत होता.", "control — plain narration"),
        ("चंद्राच्या पांढऱ्या शुभ्र जमिनीवर आरव आपल्या मोठ्या रॉकेटमधून उतरला.", "control — plain narration"),
        ("आरवच्या हातात एक मोठी आणि गोड चॉकलेटची पट्टी होती.", "control — plain narration"),
        ("सानिकाने सांगितल्याप्रमाणे रोहनने तिच्यासोबत मिळून स्वच्छता करावी का?", "control — question, rising intonation"),
    ],
    "hi": [
        ("इसी हड़बड़ाहट में आरव का हाथ कछुए के फटे हुए नक्शे पर लग गया.", "the original long-sentence case"),
        ("ठंडी हवा चल रही थी और आसमान में सफेद चाँद चमक रहा था।", "control"),
        ("तभी आरव ने देखा कि चाँद पर एक छोटा और प्यारा सा खरगोश बैठा है।", "control"),
        ("आरव ने अपने दोस्त का हाथ पकड़ा और रंगीन गुब्बारे को छोड़ दिया।", "control"),
        ("खरगोश के पास एक टोकरी थी जिसमें बहुत सारे चमकते पत्थर थे।", "control"),
        ("नन्हा कछुआ यह देखकर थोड़ा दुखी हो गया।", "control — short"),
    ],
    # NOTE: English is served by stock ai4bharat/indic-parler-tts, NOT the
    # betacraft fine-tune (which trained only mr/hi). Report it separately.
    "en": [
        ("The moon was big and round in the dark blue sky that night.", "control"),
        ("Aarav put on his blue cap and picked up a small bag.", "control"),
        ("He counted slowly, three, two, one, and closed his eyes.", "countdown — all three numbers?"),
        ("Everywhere on the moon there was soft white and grey dust.", "control"),
        ("Should Aarav share his chocolate with his friend?", "control — question"),
    ],
}


def read_key(names: tuple[str, ...]) -> str:
    for line in JAANTEHO_ENV.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() in names:
            return v.strip().strip('"').strip("'")
    return ""


def synth(text: str, lang: str, key: str, seed: int, timeout: int = 600):
    base = f"https://api.runpod.ai/v2/{ENDPOINT}"
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    req = urllib.request.Request(
        base + "/run", headers=hdr,
        data=json.dumps({"input": {"text": text, "language": lang, "seed": seed}}).encode())
    jid = json.load(urllib.request.urlopen(req, timeout=60))["id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(
            urllib.request.Request(f"{base}/status/{jid}", headers=hdr), timeout=60
        ) as h:
            st = json.load(h)
        if st["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(4)
    if st["status"] != "COMPLETED" or "error" in (st.get("output") or {}):
        return None, None, st
    o = st["output"]
    wf = wave.open(io.BytesIO(base64.b64decode(o["audio_b64"])))
    return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16), o["sr"], st


def save_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    w = wave.open(str(path), "w")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(pcm.tobytes()); w.close()


# --- generate ---------------------------------------------------------------
def generate(langs: list[str], runs: int, out_dir: Path) -> Path:
    key = read_key(("BETACRAFT_API_KEY",))
    if not key:
        sys.exit("no BETACRAFT_API_KEY found in jaanteho-fresh/.env")
    out = out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"when": out.name, "endpoint": ENDPOINT, "runs_per_sentence": runs,
                    "languages": {}}
    for lang in langs:
        print(f"\n{'=' * 70}\n{lang.upper()} — {len(PROBES[lang])} sentence(s) x {runs} run(s)\n{'=' * 70}")
        rows = []
        for si, (text, what) in enumerate(PROBES[lang]):
            for run in range(runs):
                seed = 42 + run * 1000
                pcm, sr, st = synth(text, lang, key, seed)
                if pcm is None:
                    print(f"  [{si}.{run}] GENERATION FAILED")
                    rows.append({"i": si, "run": run, "text": text, "check": what,
                                 "seed": seed, "error": "generation_failed"})
                    continue
                wav = out / f"{lang}_{si:02d}_r{run}.wav"
                save_wav(wav, pcm, sr)
                rows.append({"i": si, "run": run, "text": text, "check": what, "seed": seed,
                             "wav": wav.name, "audio_sec": round(len(pcm) / sr, 2),
                             "exec_ms": st.get("executionTime"), "delay_ms": st.get("delayTime"),
                             "verdict": None, "note": None})
                print(f"  [{si}.{run}] {len(pcm) / sr:5.2f}s  {text[:52]}")
        report["languages"][lang] = {"rows": rows}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwavs -> {out}\nNow listen:  venv/bin/python scripts/narrate/score_quality.py --listen {out}")
    return out


# --- listen -----------------------------------------------------------------
def listen(run: Path) -> None:
    """Play each clip and record a verdict. Resumable — already-scored rows are
    skipped, so a long session can be done in sittings."""
    path = run / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    print("\nFor each clip:  [g]ood  [b]ad  [r]epeat  [s]kip  [q]uit and save\n"
          "A 'bad' verdict will ask what you heard — that note lands in the report.\n")
    for lang, block in report["languages"].items():
        for row in block["rows"]:
            if row.get("verdict") or not row.get("wav"):
                continue
            wav = run / row["wav"]
            print(f"\n[{lang} {row['i']}.{row['run']}] {row['audio_sec']}s — CHECK: {row['check']}")
            print(f"  {row['text']}")
            while True:
                subprocess.run(["afplay", str(wav)], check=False)
                ans = input("  [g]ood / [b]ad / [r]epeat / [s]kip / [q]uit: ").strip().lower()
                if ans == "r":
                    continue
                if ans == "q":
                    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                    print("saved — rerun --listen to carry on")
                    return
                if ans == "s":
                    break
                if ans in ("g", "b"):
                    row["verdict"] = "good" if ans == "g" else "bad"
                    if ans == "b":
                        row["note"] = input("  what did you hear? ").strip()
                    break
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summarise(run, report)


def summarise(run: Path, report: dict) -> None:
    lines = [f"# TTS listening test — {report['when']}", "",
             f"Endpoint `{report['endpoint']}` · {report['runs_per_sentence']} generation(s) per "
             "sentence, each at a different seed.", "",
             "Scored by ear. English is the stock `ai4bharat/indic-parler-tts`, **not** the "
             "betacraft fine-tune (which trained only mr/hi) — read its row separately.", "",
             "| lang | clips scored | defects | defect rate |", "|---|---|---|---|"]
    for lang, block in report["languages"].items():
        scored = [r for r in block["rows"] if r.get("verdict")]
        bad = [r for r in scored if r["verdict"] == "bad"]
        rate = f"{100 * len(bad) / len(scored):.0f}%" if scored else "—"
        block["summary"] = {"scored": len(scored), "defects": len(bad),
                            "defect_rate": round(len(bad) / len(scored), 3) if scored else None}
        lines.append(f"| {lang} | {len(scored)} | {len(bad)} | {rate} |")
    defects = [(lang, r) for lang, b in report["languages"].items()
               for r in b["rows"] if r.get("verdict") == "bad"]
    if defects:
        lines += ["", "## Defects heard", "", "| clip | sentence | what was heard |", "|---|---|---|"]
        for lang, r in defects:
            lines.append(f"| `{r['wav']}` | {r['text'][:60]} | {r.get('note') or ''} |")
    # Same sentence good at one seed and bad at another = the stochastic failure,
    # caught in the act. This is the finding, not the raw rate.
    flaky = []
    for lang, block in report["languages"].items():
        by_sentence: dict[int, list[str]] = {}
        for r in block["rows"]:
            if r.get("verdict"):
                by_sentence.setdefault(r["i"], []).append(r["verdict"])
        flaky += [f"{lang} sentence {i}" for i, v in by_sentence.items() if len(set(v)) > 1]
    if flaky:
        lines += ["", "## Same text, different result by seed", "",
                  "These scored differently across runs of the *identical* sentence — direct "
                  "evidence the defect is a sampling draw, not a bad input:", ""]
        lines += [f"- {f}" for f in flaky]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "\n".join(lines[3:]))
    print(f"\n-> {run / 'REPORT.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["mr", "hi", "en"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs", type=int, default=2,
                    help="generations per sentence, each at a different seed — 1 run "
                         "will MISS the stochastic failures this is built to find")
    ap.add_argument("--listen", type=Path, metavar="RUN_DIR")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "quality_runs")
    a = ap.parse_args()
    if a.listen:
        listen(a.listen)
    else:
        generate(["mr", "hi", "en"] if a.all else [a.lang or "mr"], a.runs, a.out_dir)


if __name__ == "__main__":
    main()
