# AI systems resume entries

- Built an evidence-gated Diffusion Policy deployment pipeline across six RTX 4090 nodes and Jetson AGX Orin 64GB; reduced fixed-node 100-step denoising P50 from 566.7 ms to 235.8 ms (2.40x) with TensorRT while preserving numerical and Push-T quality gates, including a same-seed 90% success rerun.
- Built and validated a board-native FP16 TensorRT U-Net engine on Orin (20/20 correctness cases), reaching 3.37/4.41 ms P50/P95 and 0/500 misses against a 50 ms inference deadline; rejected unsupported energy claims when the total power rail could not be verified.

Interview boundary: the Orin latency is U-Net-only and the scheduler result is an offline replay, not robot task success.
