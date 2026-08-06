# EdgeDiffusion systems deployment

## Claim

This project connects task quality, profiling, compiler selection and target-board deployment. The official Diffusion Policy checkpoint reached **95%** high-quality completion over 60 Push-T episodes. A separate same-seed 20-episode rerun reached **90%** and stayed within the predeclared 2 percentage-point tolerance. On one fixed RTX 4090, TensorRT reduced the 100-step denoising-loop P50 from **566.71 ms** to **235.80 ms** (**2.40x**). A board-native FP16 U-Net engine on Jetson AGX Orin reached **3.37/4.41 ms P50/P95** with **0/500** 50 ms deadline misses.

## Method

- Pinned upstream repository, checkpoint and evaluation seeds.
- Separated operator, denoising-loop and simulator episode measurements.
- Required numerical and Push-T quality gates before performance claims.
- Built the final engine on Orin and retained target-board timing and telemetry.

## Evidence

- Quality: [`../evidence/edge-quality.json`](../evidence/edge-quality.json)
- Critical rerun: [`../evidence/independent-rerun.json`](../evidence/independent-rerun.json)
- Fixed-node comparison: [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json)
- Orin correctness: [`../evidence/edge-orin-correctness.json`](../evidence/edge-orin-correctness.json)
- Orin timing: [`../evidence/edge-orin.json`](../evidence/edge-orin.json)

## Limitations

The Orin number covers U-Net inference, not the full 100-step controller or robot I/O. The rented board exposes no confirmed total-module rail, so energy/action is intentionally not reported. The 20 Hz control result is an offline timing replay, not a physical-robot success claim.
