# EdgeDiffusion systems deployment

## Claim

This project connects task quality, profiling, compiler selection and target-board deployment. The official Diffusion Policy checkpoint reached **95%** high-quality completion over 60 Push-T episodes. A separate same-seed 20-episode rerun reached **90%**, exactly matching its 20-episode baseline and passing the predeclared 2 percentage-point rerun tolerance. On one fixed RTX 4090, TensorRT reduced the 100-step denoising-loop P50 from **566.71 ms** to **235.80 ms** (**2.40x**). A board-native FP16 U-Net engine on Jetson AGX Orin reached **3.37/4.41 ms P50/P95** with **0/500** 50 ms deadline misses.

## Method

- Pinned upstream repository, checkpoint and evaluation seeds.
- Separated operator, denoising-loop and simulator episode measurements.
- Required numerical and Push-T quality gates before performance claims.
- Built the final engine on Orin and retained target-board timing and telemetry.

## Supplemental step-count quality sweep

The frozen checkpoint was evaluated on paired 20-episode Push-T seed windows at five denoising step counts. The controller P50 fell from **1076.5 ms** at 100 steps to **65.2 ms** at 6 steps, but only the 100-step condition passed the predeclared mean-score and completion-rate gates. The lower-step variants are therefore measured speed/quality tradeoffs, not deployment recommendations.

| Denoising steps | Controller P50 | Mean score | High-quality completion | Quality gate |
|---:|---:|---:|---:|---|
| 100 | 1076.5 ms | 0.922 | 90% | PASS |
| 20 | 217.6 ms | 0.119 | 0% | REJECT |
| 10 | 88.4 ms | 0.093 | 0% | REJECT |
| 8 | 80.4 ms | 0.067 | 0% | REJECT |
| 6 | 65.2 ms | 0.104 | 0% | REJECT |

## Complete denoising loop on physical Orin

The TensorRT U-Net was measured inside the frozen scheduler with synthetic fixed-shape tensors. The 100-step quality reference measured **557.81/568.67 ms P50/P95** and missed all 20 local 50 ms deadlines. Six steps reached **32.43/36.28 ms** with 0/20 misses, but its paired Push-T condition produced **0%** high-quality completion. No measured step count passed both timing and task-quality gates.

## Evidence

- Quality: [`../evidence/edge-quality.json`](../evidence/edge-quality.json)
- Critical rerun: [`../evidence/independent-rerun.json`](../evidence/independent-rerun.json)
- Fixed-node comparison: [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json)
- Orin correctness: [`../evidence/edge-orin-correctness.json`](../evidence/edge-orin-correctness.json)
- Orin timing: [`../evidence/edge-orin.json`](../evidence/edge-orin.json)
- Orin complete denoising loop: [`../evidence/edge-orin-denoising-steps.json`](../evidence/edge-orin-denoising-steps.json)
- Step-count quality sweep: [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json)

## Limitations

The 3.37/4.41 ms Orin number covers one U-Net invocation; the separate scheduler-plus-U-Net table covers the denoising loop but still excludes observation and robot I/O. The step-count quality sweep used simulator observations and is not a physical-robot result. The rented board exposes no confirmed total-module rail, so energy/action is intentionally not reported. The 20 Hz control result is an offline timing replay, not a physical-robot success claim.
