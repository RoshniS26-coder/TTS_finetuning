#!/usr/bin/env python3
"""Publish the TTS research reports to a HuggingFace Static Space.
Run:  venv/bin/python publish_hf_space.py
(Requires `huggingface-cli login` already done — you are logged in.)
"""
from huggingface_hub import create_repo, HfApi

REPO = "roshni-sorigin/tts-research-reports"
PRIVATE = False  # set True for a private Space (manager then needs explicit access)

print(f">> creating static Space: {REPO} (private={PRIVATE})")
create_repo(REPO, repo_type="space", space_sdk="static", private=PRIVATE, exist_ok=True)

print(">> uploading site (index + betacraft + research + audio/ + charts/) ...")
HfApi().upload_folder(
    folder_path="tts-research-reports",
    repo_id=REPO, repo_type="space",
    ignore_patterns=[".git*", ".nojekyll"],
    commit_message="TTS research reports: mar-hin-betacraft-tts + full v1-v7 (audio, charts, metrics)",
)
print(">> DONE")
print(f">> Space page: https://huggingface.co/spaces/{REPO}")
print(">> Live URL:   https://roshni-sorigin-tts-research-reports.hf.space")
