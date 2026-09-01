import os
import subprocess
from pathlib import Path

# --- PATHS ---
RAW_DIR = Path("./downloaded_raw_stories")
SEPARATED_DIR = Path("./vocal_isolated_outputs")
FINAL_DIR = Path("./clean_vocals_mono")   # ready-to-segment audio
RAW_DIR.mkdir(parents=True, exist_ok=True)
SEPARATED_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Set this from your TTS model: model.config.sampling_rate (Parler ~44100)
TARGET_SR = 44100


def download_from_youtube(url: str):
    """Download audio-only from a YouTube / YT Music URL or playlist."""
    print(f"⬇️  Downloading audio from: {url}")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--extractor-args", "youtube:player_client=android,web",
        "-o", str(RAW_DIR / "%(title)s.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)
    print("✅ Download complete.\n")


def isolate_vocals():
    """Use the fine-tuned htdemucs_ft model to separate vocals from music."""
    raw_files = [f for f in RAW_DIR.iterdir()
                 if f.suffix.lower() in (".mp3", ".wav", ".m4a")]
    if not raw_files:
        print("❌ No audio files found in input directory.")
        return

    print(f"🚀 Separating vocals for {len(raw_files)} file(s) with htdemucs_ft...\n")
    for audio in raw_files:
        # -n htdemucs_ft = fine-tuned model (cleaner vocals)
        # --two-stems=vocals = output only vocals.wav + no_vocals.wav
        cmd = [
            "demucs",
            "-n", "htdemucs_ft",
            "--two-stems=vocals",
            "-o", str(SEPARATED_DIR),
            str(audio),
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Separated: {audio.name}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed on {audio.name}: {e}")


def convert_to_mono_target_sr():
    """Collect every vocals.wav, convert to mono at TARGET_SR for training."""
    # demucs writes to: SEPARATED_DIR/htdemucs_ft/<trackname>/vocals.wav
    vocal_files = list(SEPARATED_DIR.rglob("vocals.wav"))
    if not vocal_files:
        print("❌ No vocals.wav produced. Check the separation step.")
        return

    print(f"\n🎚️  Converting {len(vocal_files)} vocal track(s) to mono @ {TARGET_SR} Hz...")
    for vf in vocal_files:
        out_path = FINAL_DIR / f"{vf.parent.name}_vocal.wav"
        if out_path.exists():
            print(f"   ⏭️  Skipping {out_path.name} (already exists)")
            continue
        cmd = [
            "ffmpeg", "-y", "-i", str(vf),
            "-ac", "1",                # mono
            "-ar", str(TARGET_SR),     # target sample rate
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"   → {out_path.name}  (from: {vf.parent.name})")

    print("\n✅ Clean mono vocals ready in:", FINAL_DIR)
    print("⚠️  NEXT: LISTEN to each file. Delete any that sound watery, "
          "metallic, or have leftover music/animal noise before segmenting.")


if __name__ == "__main__":
    YT_URLS = [
        # "https://youtu.be/_J5N7ZyYgzU?si=dxVK5CXQ7UlmeqQ5",
        # "https://youtu.be/OZsvD71PGko?si=2PcWS9KbN9LDsq2w",
        # "https://youtu.be/O1K7oBljyVY?si=PJCzxwpxb7etNwm0"
        "https://youtu.be/c8jzNme5m8o?si=uIbAcdKmi2GKG4pU"
    ]
    for url in YT_URLS:
        download_from_youtube(url)  # comment out if files already downloaded
    isolate_vocals()
    convert_to_mono_target_sr()