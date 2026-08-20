# SmolVLA edge feasibility study

## Claim

Three LoRA fine-tuning seeds produced **13.3%** success on the main condition and **18.1%** on a secondary condition. The secondary tasks were excluded from model selection but included in final fine-tuning, so this is **not** a zero-shot transfer claim. On Jetson AGX Orin, the frozen PyTorch path measured **1190.17 ms** first-action P50 and **1191.74 ms** steady-chunk P50, only **0.837 Hz**, with **500/500** misses at 30 Hz.

## Method

- Reconciled three original August 4 audits: 432 selected episodes, 52,970 frames and 377/377 identical parquet shard hashes, with every integrity gate passed.
- Screened hyperparameters, then trained and evaluated three seeds.
- Frozen one predeclared representative policy for asynchronous runtime and Orin tests.
- Measured first action separately from steady action chunks and retained a negative deployment decision.
- Ran a paired 50-episode official LIBERO Spatial screen: eager **20.0%**, `torch.compile(reduce-overhead)` **22.0%**, paired delta **+2.0%**; the point-estimate rule passed, but CI [-10.0%, +14.0%] does not establish statistical non-inferiority.
- Ran a paired 50-episode visual resize quality gate: eager **20.0%** versus 256x256 **0.0%**, paired delta **-20.0%**; the predeclared +/-5 percentage-point gate rejected direct downsampling.
- Fine-tuned a 384x384 recovery candidate for 6000 steps, then stopped the planned three-seed/20k expansion when the paired 50-episode gate measured **10.0%** versus **20.0%** eager.
- Tested targeted TorchAO INT8 weight-only and dynamic quantization on physical Orin; both passed the action-difference gate but were slower than the **1202.12 ms** baseline and were rejected.

## Evidence

- Original dataset audit reconciliation and split hashes: [`../evidence/smolvla-dataset-audit.json`](../evidence/smolvla-dataset-audit.json)
- Three-seed evaluation: [`../evidence/smolvla-quality.json`](../evidence/smolvla-quality.json)
- Async runtime: [`../evidence/smolvla-async-runtime.json`](../evidence/smolvla-async-runtime.json)
- Orin timing: [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json)
- Orin stage profile: [`../evidence/orin-smolvla-strengthening.json`](../evidence/orin-smolvla-strengthening.json)
- Physical Orin chunk scheduler, threshold 0.5: [`../evidence/orin-smolvla-chunk-scheduler-05.json`](../evidence/orin-smolvla-chunk-scheduler-05.json)
- Physical Orin chunk scheduler, threshold 0.8: [`../evidence/orin-smolvla-chunk-scheduler-08.json`](../evidence/orin-smolvla-chunk-scheduler-08.json)
- Control replay: [`../evidence/control-replay.json`](../evidence/control-replay.json)
- Paired compile quality gate: [`../evidence/smolvla-quality-gate.json`](../evidence/smolvla-quality-gate.json)
- Paired visual resize quality gate: [`../evidence/smolvla-resize-quality-gate.json`](../evidence/smolvla-resize-quality-gate.json)
- Paired 384px fine-tune recovery gate: [`../evidence/smolvla-resize384-finetune-quality-gate.json`](../evidence/smolvla-resize384-finetune-quality-gate.json)
- Orin targeted INT8 study: [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json)
- Orin latency-envelope closed-loop gate: [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json)

## Supplemental runtime strengthening

Using the same frozen policy, fixed input seed and official Jetson container, the eager baseline measured **1287.54 ms** first-action P50. `torch.compile(reduce-overhead)` measured **1005.55 ms** (21.9% lower) but increased peak RAM from **7813 MB** to **8912 MB**. The Orin path still missed the 30 Hz deadline, so the compile result is a runtime candidate, not a direct-control claim.

The diagnostic 10-step stage profile attributes **744.92 ms (60.60%)** to denoise VLM forward, **332.45 ms (27.04%)** to prefix embedding and **96.90 ms (7.88%)** to prefix KV forward. Denoise VLM is nested inside denoise-step total; these rows must not be added together. The representative action matched baseline exactly.

## Supplemental visual resize quality gate

Using the same frozen seed-43 policy, LeRobot revision and 50 paired LIBERO Spatial episode IDs, changing only `resize_imgs_with_padding` from 512x512 to 256x256 produced **0/50 = 0.0%** success versus **10/50 = 20.0%** for eager 512x512. The paired delta was **-20.0%** with bootstrap 95% CI **[-32.0%, -10.0%]**; the absolute +/-5 percentage-point gate failed. Direct downsampling is rejected for the frozen checkpoint; lower-resolution deployment requires retraining or distillation.

## Supplemental 384px fine-tune recovery gate

A rank-32, learning-rate `5e-4`, seed-43 LoRA candidate was trained for 6000 steps at 384x384 and evaluated on the same 50 official LIBERO Spatial episode IDs. It reached **5/50 = 10.0%** versus **10/50 = 20.0%** for eager 512x512, a **-10.0%** delta with bootstrap 95% CI **[-24.0%, +2.0%]**. The candidate missed the predeclared 15% minimum, so the three-seed and 20k-step expansion was stopped.

## Supplemental Orin targeted INT8 gate

TorchAO v0.13.0 targeted 80 K/O/MLP linear layers in the SmolVLA action expert while excluding LoRA-bearing Q/V projections. Both candidates passed the action-difference limits, but weight-only measured **1415.67 ms** and dynamic INT8 **5450.43 ms** first-action P50, versus **1202.12 ms** eager. The measured latency changes were **-17.8%** and **-353.4%**; both candidates were rejected. This is a physical-Orin runtime result with synthetic observations, not a robot task-success result.

## Supplemental physical Orin chunk scheduler

Before the simulator gate, a 300-tick synthetic-input scheduler replay ran on physical Orin. Threshold 0.5 measured **1302.48/1354.78 ms** chunk P50/P95, **68.7%** fresh-action ticks, 94 fallback and 100 stale ticks. Threshold 0.8 raised freshness to **86.7%** and reduced fallback to 40, but stale ticks increased to 201. This exposes the underflow-versus-action-age tradeoff; it does not establish task success.

## Supplemental latency-envelope closed-loop gate

The paired LIBERO simulator test injected the physical Orin P50/P95 latency envelope into the async scheduler. At threshold 0.5, sync/async success was **24%/8%** with **3618** fallback ticks. At threshold 0.8, sync/async success was **40%/4%** with **4028** fallback ticks. Both violated the predeclared +/-5 percentage-point gate, so the current async policy is rejected under this injected latency envelope.

## Limitations

The paired LIBERO compile point estimate passes its predeclared screening rule, but the wide confidence interval does not establish statistical non-inferiority, a compile win, physical-robot success or 30 Hz control. The closed-loop gate is LIBERO simulation with an Orin latency envelope, not a physical-robot experiment; it uses P50/P95 endpoints rather than raw per-chunk timestamps. Orin inputs were synthetic, and scheduling replay cannot establish task success. The original row-level dataset audits are now locally archived and reconciled; their remaining boundary is that they verify parquet rows and shard identity, not decoded video pixels. Current execution remains rejected for a 30 Hz direct-control claim.
