# Resume bullets for AI systems roles

- Built a reproducible Diffusion Policy evaluation and deployment workflow across six RTX 4090 nodes and Jetson AGX Orin 64GB; used TensorRT to reduce 100-step denoising P50 from 566.7 ms to 235.8 ms (2.40x) on a fixed node, while checking numerical differences and Push-T task quality.
- Compared 100, 20, 10, 8, and 6 denoising steps on matched Push-T episodes. Controller P50 fell from 1076.5 ms to 65.2 ms, but every setting below 100 steps produced 0% high-quality completion, so I kept the slower quality-preserving reference.
- Built and validated a board-native FP16 TensorRT U-Net engine on Orin with 20/20 correctness cases. One U-Net call reached 3.37/4.41 ms P50/P95; the complete 6-step scheduler loop reached 32.4/36.3 ms with 0/20 timing misses, but its paired Push-T quality result was 0%.

The 3.37/4.41 ms figure covers only one U-Net call. The complete loop still excludes observation processing and robot I/O, and task quality was measured in simulation rather than on a physical robot.
