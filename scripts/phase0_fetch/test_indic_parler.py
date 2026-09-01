import re
import torch
import numpy as np
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model (first run will download ~2GB)...")
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
print("Model loaded.\n")

# Lion and Mouse story in Marathi (kept for reference)
# STORY = (
#     "एकदा एक मोठा सिंह जंगलात झोपला होता. "
#     "तेवढ्यात एक छोटासा उंदीर त्याच्या अंगावरून पळत गेला. "
#     "सिंहाला जाग आली आणि त्याने उंदराला पकडले. "
#     "उंदीर म्हणाला, महाराज, मला सोडा. मी एक दिवस तुमच्या उपयोगी पडेन. "
#     "सिंहाला हसू आले, पण त्याने उंदराला सोडून दिले. "
#     "काही दिवसांनी सिंह शिकाऱ्याच्या जाळ्यात अडकला. "
#     "उंदराने ते जाळे कुरतडले आणि सिंहाला मुक्त केले. "
#     "सिंहाने उंदराचे आभार मानले. लहान मित्रही मोठ्या संकटात कामी येतो."
# )

# Fallback used if mystory.txt is missing or empty
DEFAULT_PROMPT = "एक आटपाट नगर होतं. तिथे एक राजा राहत होता."

DESCRIPTION = (
    "Sunita narrates with Narration emotion in an expressive, warm storytelling tone at a moderate pace. Very clear audio with no background noise."
)

# Story read from external file
try:
    with open("mystory.txt", encoding="utf-8") as f:
        STORY = f.read().strip()
except FileNotFoundError:
    STORY = ""
if not STORY:
    STORY = DEFAULT_PROMPT

# Parler-TTS handles only short text per generation, so split the story into
# sentence-sized chunks, synthesize each, then concatenate the audio.
def split_into_chunks(text, max_chars=200):
    # Break on sentence enders (। ! ? .) and newlines, keeping the punctuation.
    pieces = re.split(r"(?<=[।!?\.\n])", text)
    chunks, cur = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if cur and len(cur) + 1 + len(piece) > max_chars:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


chunks = split_into_chunks(STORY)
print(f"Story split into {len(chunks)} chunks.\n")

filename = "test1.wav"
sr = model.config.sampling_rate
silence = np.zeros(int(0.3 * sr), dtype=np.float32)  # gap between sentences

desc_inputs = description_tokenizer(DESCRIPTION, return_tensors="pt").to(device)
segments = []
for i, chunk in enumerate(chunks, 1):
    print(f"  [{i}/{len(chunks)}] {chunk[:50]}...")
    prompt_inputs = tokenizer(chunk, return_tensors="pt").to(device)
    generation = model.generate(
        input_ids=desc_inputs.input_ids,
        attention_mask=desc_inputs.attention_mask,
        prompt_input_ids=prompt_inputs.input_ids,
        prompt_attention_mask=prompt_inputs.attention_mask,
        max_new_tokens=60000,
    )
    audio = generation.cpu().numpy().squeeze().astype(np.float32)
    segments.append(audio)
    segments.append(silence)

full_audio = np.concatenate(segments)
sf.write(filename, full_audio, sr)
print(f"\n  -> saved: {filename} ({len(full_audio) / sr:.1f} sec)")

print("\nDone.")
