#!/usr/bin/env python3
"""Push the trained Parler-TTS model to the Hugging Face Hub (PRIVATE).

RUN THIS ON THE POD, immediately after training finishes, WHILE the pod is still
up and your RunPod balance is positive. This gets the model OFF RunPod so a later
Stop/Terminate (or a $0 balance) can't lose it.

Prereqs on the pod:
  huggingface-cli login          # paste an HF token with WRITE scope
Usage:
  python push_to_hf.py                                   # defaults below
  python push_to_hf.py --model-dir /workspace/marathi_tts_output \
                       --repo roshni-sorigin/mar-hin-betacraft-tts
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/marathi_tts_output",
                    help="folder saved by training (save_pretrained output)")
    ap.add_argument("--repo", default="roshni-sorigin/mar-hin-betacraft-tts",
                    help="target HF repo id")
    ap.add_argument("--public", action="store_true",
                    help="make the repo public (default: PRIVATE)")
    ap.add_argument("--samples-dir", default=None,
                    help="optional dir of sample .wav files to upload under samples/")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"ERROR: model dir not found: {model_dir}")
    weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not weights:
        raise SystemExit(f"ERROR: no model weights (*.safetensors/*.bin) in {model_dir}")

    private = not args.public
    print(f"==> Repo {args.repo}  (private={private})")
    create_repo(args.repo, private=private, repo_type="model", exist_ok=True)

    api = HfApi()
    print(f"==> Uploading {model_dir} ...")
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo,
                      repo_type="model", commit_message="Add mar-hin-betacraft-tts fine-tuned model")

    if args.samples_dir and Path(args.samples_dir).is_dir():
        print(f"==> Uploading samples from {args.samples_dir} ...")
        api.upload_folder(folder_path=args.samples_dir, path_in_repo="samples",
                          repo_id=args.repo, repo_type="model",
                          commit_message="Add inference samples")

    print(f"==> DONE. https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
