# Where each published result comes from

This table links every headline number to the file that produced it and states the conditions under which it was measured.

| Result ID | Value | Measurement conditions | Source |
|---|---:|---|---|
| `EDGE-SUCCESS` | 0.95 ratio | 60 simulator episodes using the official checkpoint | [`../evidence/edge-quality.json`](../evidence/edge-quality.json) |
| `EDGE-SUCCESS-RERUN` | 0.9 ratio | Independent rerun of 20 official-checkpoint episodes with the same seeds as the reference run | [`../evidence/independent-rerun.json`](../evidence/independent-rerun.json) |
| `EDGE-4090-EAGER` | 566.709 ms | One fixed RTX 4090 node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-4090-TRT` | 235.798 ms | The same fixed RTX 4090 node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-4090-SPEEDUP` | 2.40337 x | Ratio of P50 measurements on the same node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-STEPS-100-P50` | 1076.48 ms | 20 matched Push-T episodes; quality stayed within the preset limits | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-STEPS-20-P50` | 217.602 ms | 20 matched Push-T episodes; quality fell outside the preset limits | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-STEPS-10-P50` | 88.3921 ms | 20 matched Push-T episodes; quality fell outside the preset limits | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-STEPS-8-P50` | 80.4068 ms | 20 matched Push-T episodes; quality fell outside the preset limits | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-STEPS-6-P50` | 65.2405 ms | 20 matched Push-T episodes; quality fell outside the preset limits | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-ORIN-P50` | 3.37094 ms | One U-Net call only; excludes simulator and robot I/O | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `EDGE-ORIN-P95` | 4.41489 ms | One U-Net call only; excludes simulator and robot I/O | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `EDGE-ORIN-MISS` | 0 count/500 | U-Net inference measured on the board | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `EDGE-ORIN-LOOP-100-P50` | 557.809 ms | Physical Orin, synthetic tensors, complete denoising loop; slower than the 50 ms target | [`../evidence/edge-orin-denoising-steps.json`](../evidence/edge-orin-denoising-steps.json) |
| `EDGE-ORIN-LOOP-6-P50` | 32.4274 ms | Physical Orin; met the timing target, but the corresponding Push-T run lost task quality | [`../evidence/edge-orin-denoising-steps.json`](../evidence/edge-orin-denoising-steps.json) |
| `EDGE-ORIN-LOOP-6-MISS` | 0 count/20 | Physical Orin timing only; the corresponding Push-T result was 0% high-quality completion | [`../evidence/edge-orin-denoising-steps.json`](../evidence/edge-orin-denoising-steps.json) |
| `VLA-MAIN` | 0.133333 ratio | 3 seeds and 45 runs; not a zero-shot result | [`../evidence/smolvla-quality.json`](../evidence/smolvla-quality.json) |
| `VLA-GEN` | 0.180952 ratio | Tasks excluded from selection but included in final fine-tuning | [`../evidence/smolvla-quality.json`](../evidence/smolvla-quality.json) |
| `VLA-ORIN-P50` | 1191.74 ms | Synthetic input on the PyTorch path | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-ORIN-RATE` | 0.836851 Hz | Synthetic input on the PyTorch path | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-ORIN-MISS` | 500 count/500 | Every measured inference missed the 30 Hz target | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-ORIN-STRENGTHENING` | 1005.55 ms | Fixed synthetic input; task quality was measured separately in LIBERO | [`../evidence/orin-smolvla-strengthening.json`](../evidence/orin-smolvla-strengthening.json) |
| `VLA-DATA-SHARD-AUDIT` | 377 count | Three original records agree on 432 episodes, 52,970 frames, and all shard hashes; later synchronization did not rerun experiments | [`../evidence/smolvla-dataset-audit.json`](../evidence/smolvla-dataset-audit.json) |
| `VLA-STAGE-DENOISE-VLM` | 60.5977 percent | Diagnostic CUDA-event profile; included within the denoising-step total | [`../evidence/orin-smolvla-strengthening.json`](../evidence/orin-smolvla-strengthening.json) |
| `VLA-STAGE-PREFIX-EMBED` | 27.0439 percent | Diagnostic CUDA-event profile | [`../evidence/orin-smolvla-strengthening.json`](../evidence/orin-smolvla-strengthening.json) |
| `VLA-CHUNK-05-FRESH` | 0.686667 ratio | 300-tick replay on the board with synthetic input; not task success | [`../evidence/orin-smolvla-chunk-scheduler-05.json`](../evidence/orin-smolvla-chunk-scheduler-05.json) |
| `VLA-CHUNK-08-FRESH` | 0.866667 ratio | 300-tick replay on the board with synthetic input; not task success | [`../evidence/orin-smolvla-chunk-scheduler-08.json`](../evidence/orin-smolvla-chunk-scheduler-08.json) |
| `VLA-CHUNK-08-STALE` | 201 count/300 | A higher refill threshold reduced fallback but increased action age | [`../evidence/orin-smolvla-chunk-scheduler-08.json`](../evidence/orin-smolvla-chunk-scheduler-08.json) |
| `VLA-LIBERO-EAGER-SUCCESS` | 0.2 ratio | 50 official LIBERO Spatial episodes using the selected seed-43 policy | [`../evidence/smolvla-quality-gate.json`](../evidence/smolvla-quality-gate.json) |
| `VLA-LIBERO-COMPILE-SUCCESS` | 0.22 ratio | The same 50 episodes with `reduce-overhead` | [`../evidence/smolvla-quality-gate.json`](../evidence/smolvla-quality-gate.json) |
| `VLA-LIBERO-COMPILE-DELTA` | 0.02 ratio | Paired episode difference; preset screening range +/-5 percentage points | [`../evidence/smolvla-quality-gate.json`](../evidence/smolvla-quality-gate.json) |
| `VLA-LIBERO-RESIZE-SUCCESS` | 0 ratio | 50 official LIBERO Spatial episodes at 256x256; quality fell outside the preset range | [`../evidence/smolvla-resize-quality-gate.json`](../evidence/smolvla-resize-quality-gate.json) |
| `VLA-LIBERO-RESIZE-DELTA` | -0.2 ratio | Paired episode difference; outside the +/-5-point range | [`../evidence/smolvla-resize-quality-gate.json`](../evidence/smolvla-resize-quality-gate.json) |
| `VLA-LIBERO-RESIZE384-FT-SUCCESS` | 0.1 ratio | 6000-step rank-32 LoRA candidate on 50 matched episodes; below the 15% continuation target | [`../evidence/smolvla-resize384-finetune-quality-gate.json`](../evidence/smolvla-resize384-finetune-quality-gate.json) |
| `VLA-LIBERO-RESIZE384-FT-DELTA` | -0.1 ratio | Paired episode difference; 10% candidate versus 20% eager reference | [`../evidence/smolvla-resize384-finetune-quality-gate.json`](../evidence/smolvla-resize384-finetune-quality-gate.json) |
| `VLA-ORIN-INT8-WO-P50` | 1415.67 ms | Physical Orin, synthetic input, targeted action-expert linear layers; slower than eager | [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json) |
| `VLA-ORIN-INT8-DYNAMIC-P50` | 5450.43 ms | Physical Orin, synthetic input, targeted action-expert linear layers; slower than eager | [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json) |
| `VLA-ORIN-INT8-WO-ACTION-MAX` | 0.0245989 abs | Within the chosen action-difference limit, but P50 was slower than eager | [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json) |
| `VLA-ORIN-INT8-DYNAMIC-ACTION-MAX` | 0.025344 abs | Within the chosen action-difference limit, but P50 was slower than eager | [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json) |
| `VLA-TRACE-05-DELTA` | -0.16 ratio | Paired LIBERO simulation with the Orin latency range; not a physical-robot run | [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json) |
| `VLA-TRACE-08-DELTA` | -0.36 ratio | Paired LIBERO simulation with the Orin latency range; not a physical-robot run | [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json) |
| `VLA-TRACE-05-FALLBACK` | 3618 count | 25 paired LIBERO episodes with latency-range injection | [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json) |
| `VLA-TRACE-08-FALLBACK` | 4028 count | 25 paired LIBERO episodes with latency-range injection | [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json) |
| `VLA-REPLAY-FRESH` | 0.0266667 ratio | Offline timing replay, not task execution | [`../evidence/control-replay.json`](../evidence/control-replay.json) |
