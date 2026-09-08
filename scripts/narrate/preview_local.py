"""Preview segmentation + SFX LOCALLY, without rebuilding or redeploying the image.

WHY THIS WORKS: the server splits whatever text it is handed. If we run the split
locally and send each resulting unit as its OWN job, the server's splitter is a
no-op on units already under its cap — so the audio is byte-for-byte what a
redeployed image with the new cap would produce. Segmentation is pure text, so
cap changes cost nothing to try; only the audio needs the GPU.

  # text only, free, instant — compare caps side by side
  python scripts/narrate/preview_local.py --file story.txt --caps 16,18,22
  # render audio through the CURRENT endpoint, with SFX substitution
  python scripts/narrate/preview_local.py --file story.txt --cap 18 --audio out.wav --sfx
"""
from __future__ import annotations
import argparse, base64, io, json, logging, re, sys, time, urllib.request, wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("preview")

# --- load the REAL split functions out of betacraft_core.py -----------------
# Imported by source extraction, not `import betacraft_core`, because that module
# pulls in torch/parler_tts (multi-GB, GPU-oriented) for what is pure string work.
def load_splitters():
    src = (ROOT / "scripts/narrate/betacraft_core.py").read_text(encoding="utf-8")
    ns = {"log": log}
    pats = [r"CLAUSE_CONJUNCTIONS = .*?\n", r"CONJ_MIN_SIDE_WORDS = \d+\n",
            r"MIN_UNIT_WORDS = \d+\n", r"PACK_HARD_CAP_WORDS = \d+\n",
            r"def _merge_short.*?\n    return merged\n",
            r"def _split_at_conjunction.*?\n    return out\n",
            r"def _split_overlong.*?\n    return out\n"]
    for p in pats:
        m = re.search(p, src, re.S)
        if m: exec(m.group(0), ns)
    return ns

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[।.!?])\s+|\n+", text.strip()) if s.strip()]

# --- SFX: synthesised from noise, no external assets, no licensing ----------
# AMBIENT/ENVIRONMENTAL sounds only — deliberately NOT lexical onomatopoeia.
#
# Tried on 2026-09-02 and REJECTED: substituting a synthesised rocket for
# 'झूऽऽऽम'. Judged worse than simply letting the narrator say the word. The
# reason is that झूऽऽऽम is a WORD a Marathi child says and a narrator performs —
# playful, not literal — so swapping in an engine recording removes the charm and
# leaves a dangling referent in the sentence ("ज्यातून असा आवाज येत होता").
# Three rocket variants (deep rumble / fuller roar / building launch, brown noise
# 40-900Hz with combustion-style amplitude modulation) all failed the same way:
# they sound like a rocket, but the story did not want a rocket, it wanted "झूम".
#
# THE RULE: SFX for sounds a NARRATOR CANNOT MAKE (wind, rain, water, birds —
# atmosphere that lives between or under sentences). Leave anything the voice
# says out loud to the voice. If a spoken onomatopoeia reads flatly, that is a
# TRAINING-DATA problem (the corpora contain none), not one SFX can solve.
SFX_MAP = {
    "सरसर": "wind",
}
def render_sfx(kind, sr, rng=None):
    rng = rng or np.random.default_rng(7)
    dur = {"hiss": .85, "wind": 1.2, "whoosh": .7, "thud": .35}.get(kind, .8)
    n = int(sr * dur); x = rng.standard_normal(n)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / sr)
    band = {"hiss": (3000, 9000), "wind": (200, 1800), "whoosh": (400, 6000), "thud": (40, 300)}[kind]
    X[(f < band[0]) | (f > band[1])] = 0
    x = np.fft.irfft(X, n); x /= (np.abs(x).max() or 1)
    if kind == "whoosh":                    # rising sweep -> a "zoom" past the ear
        x *= np.linspace(0.2, 1.0, n) ** 2
    a, d = int(sr * .12), int(sr * min(.35, dur * .5))
    env = np.ones(n)
    env[:a] = .5 * (1 - np.cos(np.linspace(0, np.pi, a)))
    env[-d:] = .5 * (1 + np.cos(np.linspace(0, np.pi, d)))
    return (x * env * 0.16).astype(np.float32)

# Quote marks the story model wraps sound words in ('झूऽऽऽम'). Removed WITH the
# word: stripping only the word leaves `ज्यातून ' ' असा`, and the model then reads
# the empty quotes as a pause/artefact.
_QUOTES = "\"'‘’“”‛‟"

def extract_sfx(text):
    """Strip onomatopoeia (and any quotes around it) from the narration.

    Returns (clean_text, kinds). Longest keys first so 'झूऽऽऽम' is matched before
    the shorter 'झूम' that is a prefix-ish variant of it.
    """
    kinds = []
    for word in sorted(SFX_MAP, key=len, reverse=True):
        pat = rf"[{re.escape(_QUOTES)}]?\s*{re.escape(word)}\s*[{re.escape(_QUOTES)}]?"
        if re.search(pat, text):
            kinds.append(SFX_MAP[word])
            text = re.sub(pat, " ", text)
    # collapse the whitespace the removal leaves, including before punctuation
    text = re.sub(r"\s{2,}", " ", text)
    return re.sub(r"\s+([।.,!?])", r"\1", text).strip(), kinds

# --- units ------------------------------------------------------------------
def build_units(text, cap, ns, use_sfx):
    units = []
    for sent in split_sentences(text):
        kinds = []
        if use_sfx:
            sent, kinds = extract_sfx(sent)
            if not sent: 
                units.append(("", kinds)); continue
        for piece in ns["_split_overlong"](sent, cap):
            units.append((piece, kinds)); kinds = []
    if "_merge_short" in ns:
        texts = ns["_merge_short"]([u for u, _ in units if u])
        if len(texts) == len([u for u, _ in units if u]):
            units = [(t, k) for t, (_, k) in zip(texts, units)]
    return units

# --- audio via the CURRENT endpoint -----------------------------------------
def tts(text, lang, key, ep, seed=648):
    base, hdr = f"https://api.runpod.ai/v2/{ep}", {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    r = urllib.request.Request(base + "/run", headers=hdr, data=json.dumps(
        {"input": {"text": text, "language": lang, "seed": seed}}).encode())
    jid = json.load(urllib.request.urlopen(r, timeout=60))["id"]
    t0 = time.time()
    while time.time() - t0 < 600:
        with urllib.request.urlopen(urllib.request.Request(f"{base}/status/{jid}", headers=hdr), timeout=60) as h:
            st = json.load(h)
        if st["status"] in ("COMPLETED", "FAILED"): break
        time.sleep(4)
    if st["status"] != "COMPLETED" or "error" in (st.get("output") or {}):
        print(f"   !! job {st['status']}: {json.dumps(st)[:200]}"); return None, 44100
    o = st["output"]
    wf = wave.open(io.BytesIO(base64.b64decode(o["audio_b64"])))
    return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768, o["sr"]

def read_key():
    for line in open("/Users/roshnisundrani/jaanteho-fresh/.env", encoding="utf-8"):
        m = re.match(r'\s*BETACRAFT_API_KEY\s*=\s*"?([^"\s#]+)', line)
        if m and not line.lstrip().startswith("#"): return m.group(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file"); ap.add_argument("--text")
    ap.add_argument("--cap", type=int); ap.add_argument("--caps")
    ap.add_argument("--lang", default="mr"); ap.add_argument("--sfx", action="store_true")
    ap.add_argument("--audio"); ap.add_argument("--gap-ms", type=int, default=350)
    ap.add_argument("--endpoint", default="5tk0794i74t8lh")
    a = ap.parse_args()
    text = a.text or Path(a.file).read_text(encoding="utf-8")
    ns = load_splitters()

    caps = [int(c) for c in a.caps.split(",")] if a.caps else [a.cap or ns["PACK_HARD_CAP_WORDS"]]
    for cap in caps:
        units = build_units(text, cap, ns, a.sfx)
        speech = [u for u, _ in units if u]
        print(f"\n=== cap={cap}: {len(speech)} speech unit(s), "
              f"{sum(len(u.split()) for u in speech)} words ===")
        for i, (u, k) in enumerate(units):
            tag = f"  [SFX:{','.join(k)}]" if k else ""
            print(f"  {i:2d} ({len(u.split()):2d}w){tag} {u[:78]}")

    if not a.audio: return
    cap = caps[-1]; key = read_key()
    out, sr = [], 44100
    for i, (u, kinds) in enumerate(build_units(text, cap, ns, a.sfx)):
        for k in kinds:
            out.append(render_sfx(k, sr)); out.append(np.zeros(int(sr * .12), np.float32))
        if not u: continue
        print(f"  synth {i}: {len(u.split())}w …")
        w, sr = tts(u, a.lang, key, a.endpoint)
        if w is not None:
            out.append(w); out.append(np.zeros(int(sr * a.gap_ms / 1000), np.float32))
    y = np.concatenate(out)
    wf = wave.open(a.audio, "w"); wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()); wf.close()
    print(f"\n-> {a.audio}  ({len(y)/sr:.1f}s)")

if __name__ == "__main__":
    main()
