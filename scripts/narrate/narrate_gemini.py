"""Narrate a Marathi text file with Google Gemini TTS and save a WAV.

Gemini's TTS models return raw 16-bit PCM mono at 24 kHz; we wrap that in a WAV
container. Voice + style are controlled by a natural-language instruction prefix
plus a prebuilt voice name. The API key is read from the `gemini-key=` line in
.env (or --api-key / GEMINI_API_KEY env var).

  python narrate_gemini.py --text-file ../phase0_fetch/mystory.txt \
      --output model_outputs/gemini/mystory_gemini.wav --voice Kore
"""
from __future__ import annotations
import argparse, logging, os, re, sys, wave
from pathlib import Path

from google import genai
from google.genai import types

SR = 24000          # Gemini TTS output: 24 kHz, 16-bit, mono PCM
SAMPLE_WIDTH = 2
CHANNELS = 1
DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Kore"   # warm, steady female voice; see --voice for others
STYLE = ("Narrate this Marathi children's story in a warm, expressive, gentle "
         "storyteller's voice at a calm, moderate pace, with clear pronunciation:")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gemini")


def load_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    env = Path(".env")
    if env.is_file():
        m = re.search(r"^\s*gemini-key\s*=\s*(\S+)", env.read_text(encoding="utf-8"), re.M)
        if m:
            return m.group(1).strip()
    sys.exit("ERROR: no Gemini key (pass --api-key, set GEMINI_API_KEY, or add gemini-key= to .env)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-file", type=Path)
    ap.add_argument("--text", type=str)
    ap.add_argument("--output", type=Path, default=Path("model_outputs/gemini/mystory_gemini.wav"))
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="Prebuilt voice: Kore, Aoede, Leda, Callirrhoe, Puck, Charon, Fenrir...")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--style", default=STYLE, help="Natural-language style instruction prepended to the text")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text.strip()
    else:
        sys.exit("ERROR: provide --text-file or --text")
    if not text:
        sys.exit("ERROR: empty text")

    client = genai.Client(api_key=load_key(args.api_key))
    prompt = f"{args.style}\n\n{text}"
    log.info("model=%s voice=%s | %d chars", args.model, args.voice, len(text))

    resp = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice)
                )
            ),
        ),
    )

    try:
        pcm = resp.candidates[0].content.parts[0].inline_data.data
    except (AttributeError, IndexError, TypeError):
        sys.exit(f"ERROR: no audio in response: {resp}")
    if not pcm:
        sys.exit("ERROR: empty audio payload returned")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SR)
        w.writeframes(pcm)

    secs = len(pcm) / (SR * SAMPLE_WIDTH * CHANNELS)
    log.info("Wrote %s (%.1fs @ %dHz)", args.output, secs, SR)


if __name__ == "__main__":
    main()
