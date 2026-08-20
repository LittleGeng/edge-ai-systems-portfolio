# AI systems resume entries

- Built an evidence-gated Diffusion Policy deployment pipeline across six RTX 4090 nodes and Jetson AGX Orin 64GB; reduced fixed-node 100-step denoising P50 from 566.7 ms to 235.8 ms (2.40x) with TensorRT while preserving numerical and Push-T quality gates, including a same-seed 90% success rerun.
- Ran a paired Push-T denoising-step sweep (100/20/10/8/6): controller P50 fell from 1076.5 ms to 65.2 ms, but the predeclared quality gate rejected every reduction below 100 steps; documented the negative deployment decision with raw episode evidence.
- Built and validated a board-native FP16 TensorRT U-Net engine on Orin (20/20 correctness cases), reaching 3.37/4.41 ms for one U-Net invocation; measured the complete 6-step scheduler loop at 32.4/36.3 ms with 0/20 timing misses, then rejected it because the paired Push-T quality gate fell to 0%.

Interview boundary: the 3.37/4.41 ms result is U-Net-only; the complete loop still excludes observation and robot I/O, and its quality evidence is simulation rather than a physical-robot run.
