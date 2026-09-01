#!/usr/bin/env python3
"""Generate kids' story text with Gemini, across registers, in Marathi + Hindi.

Same intent/approach as the Claude hand-written samples, codified into a prompt so it can
scale. This first pass mirrors the 6 sample topics (incl. the Ganesha "how birds got colours"
folk-tale) so you can compare Gemini vs Claude quality 1:1.

  venv/bin/python scripts/dataprep_gemini/generate_stories.py --model gemini-2.5-pro --out-dir generate_stories_gemini

Gemini key read from .env (a line whose key contains "gemini"); never printed.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# id -> (language, register, topic/premise, moral, interactive?, hero_name)
SPECS = [
    ("mr_01_panchatantra_sasa_gajar", "Marathi", "Panchatantra fable with talking animals",
     "A small rabbit secretly eats a farmer's sweet carrots and first denies it, then feels guilty and admits the truth, and is gently forgiven.",
     "Always tell the truth and admit your mistake.", False, ""),
    ("mr_02_atmakatha_chhatri", "Marathi", "Atmakatha — a first-person autobiography told by an object",
     "A red umbrella tells its own life story: how a little girl chose it, how it kept her dry in the rain, and how one day it sheltered a drenched old grandfather.",
     "Being useful to others is the real joy.", False, ""),
    ("mr_03_fun_zoomzoom_gadi", "Marathi", "Playful, funny vehicle story",
     "A little white ambulance called Zoomzoom-gaadi feels dull and asks the King of Colours to make her colourful; then people mistake her for a bus, she gets stuck in traffic, and she happily chooses to be a plain ambulance again.",
     "Be content and happy with who you are.", False, ""),
    ("hi_04_folktale_birds_colors", "Hindi", "Origin folk-tale (how something came to be), gentle and full of wonder",
     "How the birds got their colours: Lord Ganesha calls the birds one by one and gives each a colour they choose; the crow arrives last, all colours are finished, so the leftover colours mix into a shiny black just for him.",
     "Every colour — and everyone — is special in their own way.", False, ""),
    ("hi_05_interactive_aarav_pilla", "Hindi", "Interactive kid-hero story that ends on a yes/no choice",
     "At dusk the hero hears a soft whimper and finds a small shivering lost puppy behind a bush, just as the hero's favourite TV show is about to start.",
     "Helping others matters more than screen time.", True, "आरव"),
    ("hi_06_brave_veeru_nadi", "Hindi", "Brave-little-hero adventure",
     "When the village river dries up, a tiny but big-hearted boy climbs a hard, rocky mountain path alone to clear the blocked spring and bring the water back.",
     "Courage and perseverance win — those who keep trying never truly lose.", False, ""),
]


def load_key(env: Path) -> str:
    for line in (env.read_text(encoding="utf-8").splitlines() if env.exists() else []):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if "gemini" in k.lower():
            return v.strip().strip('"').strip("'")
    return os.environ.get("GEMINI_API_KEY", "")


def build_prompt(language, register, topic, moral, interactive, hero) -> str:
    p = f"""You are a warm, animated Indian children's storyteller. Write ONE short story for kids aged 3 to 8, entirely in {language} (Devanagari script only — absolutely no English words).

Story type: {register}
Premise: {topic}
Moral (let it emerge naturally through what characters do — NEVER announce it): {moral}

Rules:
- Use simple, easy, everyday {language} words a young child already knows. Short, flowing sentences.
- Use ORIGINAL characters and names that feel native to {language}'s culture. Do NOT use any trademarked or branded characters (no Chhota Bheem, Spider-Man, Motu Patlu, etc.).
- Weave in playful sound-words / onomatopoeia that are natural in {language} (animal sounds, thuds, whooshes) and one short repeated refrain the child can echo.
- Write any numbers as words, not digits.
- Do NOT open with a formulaic phrase ("once upon a time" / "एक होता"). Open with a vivid image or mid-action.
- Length: about 150 to 220 words.
- Warm, sweet, animated storytelling tone.
"""
    if interactive:
        p += (f'- The hero is a child named "{hero}". Name {hero} inside the narration as the protagonist.\n'
              f'- End with ONE yes/no decision question in {language}, phrased in the third person about {hero} '
              f'(e.g. asking whether {hero} chooses to do the kind thing). The question must be answerable only with yes or no.\n')
    else:
        p += "- End with a warm, satisfying close (no question).\n"
    p += f"\nOutput ONLY the story in {language} (a title on the first line is fine). No English, no meta, no explanation."
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro", help="Gemini text model id")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "generate_stories_gemini")
    ap.add_argument("--env", type=Path, default=ROOT / ".env")
    ap.add_argument("--only", default=None, help="comma-separated ids to generate (default: all)")
    args = ap.parse_args()

    try:
        from google import genai
    except ImportError:
        sys.exit("install: venv/bin/python -m pip install google-genai")
    key = load_key(args.env)
    if not key:
        sys.exit(f"no Gemini key in {args.env}")
    client = genai.Client(api_key=key)

    specs = SPECS
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        specs = [s for s in SPECS if s[0] in want]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {args.model} | {len(specs)} stories -> {args.out_dir}\n")
    for sid, lang, register, topic, moral, interactive, hero in specs:
        prompt = build_prompt(lang, register, topic, moral, interactive, hero)
        try:
            resp = client.models.generate_content(model=args.model, contents=prompt)
            text = (resp.text or "").strip()
        except Exception as exc:
            print(f"  ERR {sid}: {str(exc)[:100]}")
            continue
        out = args.out_dir / f"{sid}.txt"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"  OK  {sid:<34} {len(text.split()):>3} words -> {out.name}")
        time.sleep(0.5)
    print("\nDone. Compare against ../generate_stories_claude/")


if __name__ == "__main__":
    main()
