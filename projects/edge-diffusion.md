# EdgeDiffusion deployment study

## What I built

This project follows a Diffusion Policy workload from simulator evaluation to GPU optimization and Jetson deployment. The official checkpoint achieved **95%** high-quality completion over 60 Push-T episodes. A separate 20-episode rerun used the same seeds as its reference run and reproduced the same **90%** result.

On one fixed RTX 4090, TensorRT reduced P50 for the 100-step denoising loop from **566.71 ms** to **235.80 ms**, a **2.40x** speedup. I then built an FP16 U-Net engine directly on Jetson AGX Orin. A single U-Net call measured **3.37/4.41 ms P50/P95**, with **0/500** misses against a 50 ms target.

## How I evaluated it

- Kept the upstream revision, checkpoint, and evaluation seeds fixed across comparisons.
- Timed individual operators, the full denoising loop, and simulator episodes separately.
- Checked numerical differences and Push-T task quality before treating a faster backend as usable.
- Built the final engine on the Orin itself and retained board timing and telemetry.

## Denoising-step tradeoff

I evaluated 100, 20, 10, 8, and 6 denoising steps on matched 20-episode Push-T seed sets. Controller P50 fell from **1076.5 ms** at 100 steps to **65.2 ms** at 6 steps, but only the 100-step setting stayed within the preset mean-score and completion-rate differences.

| Denoising steps | Controller P50 | Mean score | High-quality completion | Result |
|---:|---:|---:|---:|---|
| 100 | 1076.5 ms | 0.922 | 90% | Quality retained |
| 20 | 217.6 ms | 0.119 | 0% | Too much quality loss |
| 10 | 88.4 ms | 0.093 | 0% | Too much quality loss |
| 8 | 80.4 ms | 0.067 | 0% | Too much quality loss |
| 6 | 65.2 ms | 0.104 | 0% | Too much quality loss |

## Complete loop on Orin

I also ran the TensorRT U-Net inside the DDPM scheduler on physical Orin using synthetic, fixed-shape tensors. The 100-step loop measured **557.81/568.67 ms P50/P95** and missed all 20 local 50 ms targets. Six steps reached **32.43/36.28 ms** with 0/20 misses, but the matching Push-T evaluation had 0% high-quality completion. None of the tested step counts met both the timing and task-quality requirements.

## Source results

- [Push-T quality](../evidence/edge-quality.json)
- [Independent rerun](../evidence/independent-rerun.json)
- [RTX 4090 backend comparison](../evidence/edge-fixed-node.json)
- [Orin correctness](../evidence/edge-orin-correctness.json)
- [Single U-Net timing on Orin](../evidence/edge-orin.json)
- [Complete denoising loop on Orin](../evidence/edge-orin-denoising-steps.json)
- [Denoising-step quality comparison](../evidence/edge-step-quality.json)

## What the results do not show

The **3.37/4.41 ms** result is for one U-Net invocation, not the full controller. The scheduler-plus-U-Net measurement still excludes observation processing, sensors, actuators, networking, and robot safety logic. Push-T quality was measured in simulation, not on a physical robot. I also do not report energy per action because the available telemetry did not provide a confirmed, non-overlapping total-module power rail.
