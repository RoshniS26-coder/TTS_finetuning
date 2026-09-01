import subprocess
from pathlib import Path

FINAL_DIR = Path("./clean_vocals_mono")
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def download_audio(url: str, out_name: str = "audio_003_mono.wav"):
    out_path = FINAL_DIR / out_name
    print(f"Downloading audio from: {url}")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--extractor-args", "youtube:player_client=android,web",
        "--postprocessor-args", "ffmpeg:-ac 1",
        "-o", str(out_path),
        url,
    ]
    subprocess.run(cmd, check=True)
    print(f"Done. Saved to: {out_path}")


if __name__ == "__main__":
    # YT_URL = "https://youtu.be/gVlBA9iIFGA?si=yBXBQTgpWi1yd877"
    # YT_URL = "https://youtu.be/XRn08lWkz7A?si=JdrddUejjDDJIYaJ"
    YT_URL="https://youtu.be/PDD3y-XkChA?si=bw5oBPdYYp8szf3c"
    download_audio(YT_URL)
