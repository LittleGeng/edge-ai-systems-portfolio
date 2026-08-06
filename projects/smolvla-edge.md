# SmolVLA edge feasibility study

## Claim

Three LoRA fine-tuning seeds produced **13.3%** success on the main condition and **18.1%** on a secondary condition. The secondary tasks were excluded from model selection but included in final fine-tuning, so this is **not** a zero-shot transfer claim. On Jetson AGX Orin, the frozen PyTorch path measured **1190.17 ms** first-action P50 and **1191.74 ms** steady-chunk P50, only **0.837 Hz**, with **500/500** misses at 30 Hz.

## Method

- Audited and deterministically split the pinned LeRobot LIBERO spatial subset.
- Screened hyperparameters, then trained and evaluated three seeds.
- Frozen one predeclared representative policy for asynchronous runtime and Orin tests.
- Measured first action separately from steady action chunks and retained a negative deployment decision.

## Evidence

- Dataset recovery and split hashes: [`../evidence/smolvla-dataset-audit.json`](../evidence/smolvla-dataset-audit.json)
- Three-seed evaluation: [`../evidence/smolvla-quality.json`](../evidence/smolvla-quality.json)
- Async runtime: [`../evidence/smolvla-async-runtime.json`](../evidence/smolvla-async-runtime.json)
- Orin timing: [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json)
- Control replay: [`../evidence/control-replay.json`](../evidence/control-replay.json)

## Limitations

Orin inputs were synthetic, with no physical robot or simulator task execution. Offline scheduling replay cannot establish task success. The original VLA-01 per-frame audit file was not synchronized; recovered official metadata and split hashes are published with that limitation. Current execution is rejected for a 30 Hz direct-control claim.
