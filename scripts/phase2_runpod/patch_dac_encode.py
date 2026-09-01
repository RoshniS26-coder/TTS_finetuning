#!/usr/bin/env python3
"""Patch parler-tts so DAC audio-token encoding doesn't pass the unsupported
'bandwidth' kwarg to DacModel.encode().

parler-tts main HEAD checks `if bandwidth is not None:` FIRST when choosing the
codec arg; for the DAC codec `bandwidth` is non-None, so it passes batch["bandwidth"]
and DacModel.encode() (transformers 4.46.1) raises:
    TypeError: DacModel.encode() got an unexpected keyword argument 'bandwidth'
Disabling that branch makes the code fall through to the `n_quantizers` branch,
which DAC's encode() accepts.

Indentation-agnostic (the block is deeply nested), idempotent, and safe: if the
target line isn't found it changes nothing and exits 1.

Run on the pod:  python /workspace/patch_dac_encode.py
"""
import re, sys

PATH = "/workspace/parler-tts/training/run_parler_tts_training.py"
MARK = "# patched: DAC.encode rejects bandwidth"

src = open(PATH).read()
if MARK in src:
    print("already patched — nothing to do.")
    sys.exit(0)

pat = re.compile(r'^([ \t]*)if bandwidth is not None:[ \t]*$', re.M)
n = len(pat.findall(src))
if n == 0:
    print("ERROR: 'if bandwidth is not None:' not found. File NOT modified.")
    sys.exit(1)

src = pat.sub(lambda m: f"{m.group(1)}if False:  {MARK}", src)
open(PATH, "w").write(src)
print(f"patched {n} occurrence(s) in {PATH} -> DAC uses n_quantizers, not bandwidth.")
