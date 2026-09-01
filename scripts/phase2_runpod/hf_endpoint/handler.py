"""Custom HF Inference Endpoint handler for the Parler-TTS betacraft model.

Parler-TTS is NOT a stock Transformers pipeline: it uses the custom
`ParlerTTSForConditionalGeneration` class and takes TWO inputs — a `description`
caption (voice/style, via the frozen Flan-T5) and the `prompt` text to speak.
The default HF TTS handler can't do that, so we provide this one.

Deploy: put this file + requirements.txt in the ROOT of the model repo
(roshni-sorigin/mar-hin-betacraft-tts), then create a dedicated Inference
Endpoint pointing at that repo on a GPU (T4/L4/A10 is plenty).

Request  JSON: {"inputs": "<Devanagari text>", "parameters": {"description": "<caption>"}}
Response JSON: {"audio_base64": "<wav bytes b64>", "sampling_rate": <int>}
"""
from typing import Any, Dict
import io, base64

import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Caption-match rule: use the EXACT training caption; change ONLY the speaker name.
# Marathi text -> Sunita ; Hindi text -> Divya. Language follows the PROMPT text.
DEFAULT_DESCRIPTION = (
    "Sunita narrates a children's story in an expressive, warm, animated "
    "storytelling tone with emotional variation, at a moderate pace. "
    "Very clear audio with no background noise."
)


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(path).to(self.device)
        self.model.eval()
        # prompt tokenizer = the model repo; description tokenizer = its text encoder
        self.prompt_tok = AutoTokenizer.from_pretrained(path)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sampling_rate = self.model.config.sampling_rate

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = (data.get("inputs") or "").strip()
        if not text:
            return {"error": "no 'inputs' text provided"}
        params = data.get("parameters") or {}
        description = params.get("description", DEFAULT_DESCRIPTION)

        desc_ids = self.desc_tok(description, return_tensors="pt").input_ids.to(self.device)
        prompt_ids = self.prompt_tok(text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            audio = self.model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids)

        wav = audio.cpu().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, wav, self.sampling_rate, format="WAV")
        return {
            "audio_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "sampling_rate": self.sampling_rate,
        }
