# IndicF5 evaluation — rejected (2026-09-25)

**Verdict: does not work for our Marathi/Hindi story narration. Do not re-run without
reading the "what was already ruled out" table first.**

Evaluated `ai4bharat/IndicF5` (flow-matching, voice-cloning) as an architectural
alternative to our autoregressive Parler-TTS (`betacraft-tts`). The motivating idea was
sound: flow matching has no sequential EOS decision, so it structurally cannot produce
the early-EOS **word drops** we chase in `HALLUCINATION_STRATEGY.md`. The idea survives.
This implementation does not deliver it.

Output was garbage (Marathi and Hindi) under every configuration tested.

## The published model is broken as shipped

`model.py` at revision `ba85abe` — **the current release** (cached sha == live sha) — has
three independent defects. Anyone following the documented
`AutoModel.from_pretrained(..., trust_remote_code=True)` path either crashes or silently
gets an **untrained DiT**:

| Defect | Consequence |
|---|---|
| Vocoder built inside `__init__` | Crashes under transformers >=5 meta-device init: `RuntimeError: Tensor on device cpu is not on the expected device meta!` |
| `load_model()` called with no `ckpt_path` | `TypeError` — required positional missing (f5-tts 1.1.22) |
| `load_state_dict` from `model.safetensors` **commented out** | The 1.4 GB trained checkpoint is never loaded |

So any *earlier* negative assessment of IndicF5 made through the HF wrapper was measuring
random weights and should be disregarded. This evaluation bypassed the wrapper.

## How to load it correctly (if ever revisited)

Drive `f5_tts` directly — skip the HF wrapper entirely:

- `ckpt`  = `hf_hub_download("ai4bharat/IndicF5", "model.safetensors")`
- `vocab` = `hf_hub_download("ai4bharat/IndicF5", "checkpoints/vocab.txt")`
- arch    = `dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)`
- `load_model(DiT, arch, ckpt, mel_spec_type="vocos", vocab_file=vocab, device=...)`
- then `load_vocoder("vocos")` + `infer_process(...)`

**Required patch:** the published checkpoint was saved through `torch.compile`, so every
key carries an `_orig_mod.` prefix. `f5_tts.load_checkpoint` strips `ema_model.` but not
`_orig_mod.`, so a stock load fails with a full missing/unexpected key split. Strip both
prefixes (and drop `initted`/`step`). Confirmation you got it right: **0 missing keys,
447 DiT tensors loaded**, 83 "unexpected" (those are the `vocoder.*` keys — see below).

## What was already ruled out

Each of these was tested and eliminated. Do not re-test them.

| Suspected cause | Test performed | Result |
|---|---|---|
| Weights not loading | Key-by-key state_dict comparison | **0 missing**, 447/447 DiT tensors loaded |
| Wrong / fine-tuned vocoder | Bit-compared checkpoint's 83 `vocoder.*` tensors vs stock `charactr/vocos-mel-24khz` | **Identical**, max abs deviation **0.0** — stock vocos is correct |
| Bad reference clip (ours is synthetic betacraft output, i.e. out-of-distribution for a cloning model) | Re-ran with ai4bharat's **own official** `prompts/PAN_F_HAPPY_00001.wav` + its README-documented ref-text | **Still garbage** |

References tried: `mr02_recheck/20260923-144309/Sunita_s648.wav` (Marathi, ear-confirmed
clean), `quality_runs/20260903-150327/hi_02_r0.wav` (Hindi, normal pace 0.39 s/word), and
the official Punjabi prompt. Ref-text was the normalised (comma-free) transcript in each
case. Sample outputs kept in `indicf5_test/`.

## The one remaining confound (not chased, deliberately)

The test venv was **Python 3.14 / torch 2.14 / transformers 5.17** — far ahead of what
f5-tts targets. It produced one hard incompatibility (the meta-device crash above) and a
`torch.jit` deprecation warning on MPS. MPS op differences wrecking the output cannot be
fully excluded.

Not pursued, because even a clean result would leave IndicF5 unusable for us:

1. It is **voice cloning** — it needs natural recorded reference audio, which we do not
   have. All our clips are synthetic TTS output.
2. It offers **no caption/description conditioning**, so no control over prosody, style or
   expressiveness — which was the actual goal, not a voice change.
3. `betacraft-tts` at `top_p=1.0` is measurably working (Hindi worst-unit length ratio
   1.46x -> 1.15x on 5/5 seeds; retries eliminated).

Speed, for the record: rtf 5.3-9.0 on local MPS (betacraft local MPS is ~10.6, RunPod
~1.24). Local numbers do not predict RunPod.

## Conclusion

The word-drop fix we need remains the forced-alignment detector
(`HALLUCINATION_STRATEGY.md` §5), not a model swap.

Test artifacts (`venv-f5`, the 2.6 GB HF cache, vocos) were deleted to reclaim ~4.5 GB.
