# Public claim evidence map

| Claim | Value | Scope | Evidence |
|---|---:|---|---|
| `EDGE-SUCCESS` | 0.95 ratio | 60 official-checkpoint simulator episodes | [`../evidence/edge-quality.json`](../evidence/edge-quality.json) |
| `EDGE-SUCCESS-RERUN` | 0.9 ratio | 20 same-seed official-checkpoint episodes; independent rerun | [`../evidence/independent-rerun.json`](../evidence/independent-rerun.json) |
| `EDGE-4090-EAGER` | 566.709 ms | fixed RTX 4090 node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-4090-TRT` | 235.798 ms | fixed RTX 4090 node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-4090-SPEEDUP` | 2.40337 x | P50 ratio on same node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-STEPS-100-P50` | 1076.48 ms | 20 paired Push-T episodes; quality gate passed | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-STEPS-20-P50` | 217.602 ms | 20 paired Push-T episodes; quality gate rejected | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-STEPS-10-P50` | 88.3921 ms | 20 paired Push-T episodes; quality gate rejected | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-STEPS-8-P50` | 80.4068 ms | 20 paired Push-T episodes; quality gate rejected | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-STEPS-6-P50` | 65.2405 ms | 20 paired Push-T episodes; quality gate rejected | [`../evidence/edge-step-quality.json`](../evidence/edge-step-quality.json) |
| `EDGE-ORIN-P50` | 3.37094 ms | U-Net inference only; no simulator/robot I/O | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `EDGE-ORIN-P95` | 4.41489 ms | U-Net inference only; no simulator/robot I/O | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `EDGE-ORIN-MISS` | 0 count/500 | measured board inference | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `EDGE-ORIN-LOOP-100-P50` | 557.809 ms | physical Orin, synthetic tensors, complete denoising loop; 50 ms timing gate missed | [`../evidence/edge-orin-denoising-steps.json`](../evidence/edge-orin-denoising-steps.json) |
| `EDGE-ORIN-LOOP-6-P50` | 32.4274 ms | physical Orin timing gate passed; paired Push-T quality gate rejected | [`../evidence/edge-orin-denoising-steps.json`](../evidence/edge-orin-denoising-steps.json) |
| `EDGE-ORIN-LOOP-6-MISS` | 0 count/20 | physical Orin timing only; paired Push-T quality gate rejected | [`../evidence/edge-orin-denoising-steps.json`](../evidence/edge-orin-denoising-steps.json) |
| `VLA-MAIN` | 0.133333 ratio | 3 seeds, 45 runs; no zero-shot claim | [`../evidence/smolvla-quality.json`](../evidence/smolvla-quality.json) |
| `VLA-GEN` | 0.180952 ratio | tasks excluded from selection but included in final fine-tuning | [`../evidence/smolvla-quality.json`](../evidence/smolvla-quality.json) |
| `VLA-ORIN-P50` | 1191.74 ms | synthetic input, PyTorch path | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-ORIN-RATE` | 0.836851 Hz | synthetic input, PyTorch path | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-ORIN-MISS` | 500 count/500 | every measured inference missed | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-ORIN-STRENGTHENING` | 1005.55 ms | fixed synthetic input; paired LIBERO task-quality gate reported separately | [`../evidence/orin-smolvla-strengthening.json`](../evidence/orin-smolvla-strengthening.json) |
| `VLA-DATA-SHARD-AUDIT` | 377 count | D/E/F original audits agree; 432 episodes and 52,970 frames; archive synchronization did not rerun experiments | [`../evidence/smolvla-dataset-audit.json`](../evidence/smolvla-dataset-audit.json) |
| `VLA-STAGE-DENOISE-VLM` | 60.5977 percent | diagnostic CUDA-event stage profile; nested inside denoise-step total | [`../evidence/orin-smolvla-strengthening.json`](../evidence/orin-smolvla-strengthening.json) |
| `VLA-STAGE-PREFIX-EMBED` | 27.0439 percent | diagnostic CUDA-event stage profile | [`../evidence/orin-smolvla-strengthening.json`](../evidence/orin-smolvla-strengthening.json) |
| `VLA-CHUNK-05-FRESH` | 0.686667 ratio | 300-tick synthetic-input board replay; not task success | [`../evidence/orin-smolvla-chunk-scheduler-05.json`](../evidence/orin-smolvla-chunk-scheduler-05.json) |
| `VLA-CHUNK-08-FRESH` | 0.866667 ratio | 300-tick synthetic-input board replay; not task success | [`../evidence/orin-smolvla-chunk-scheduler-08.json`](../evidence/orin-smolvla-chunk-scheduler-08.json) |
| `VLA-CHUNK-08-STALE` | 201 count/300 | higher refill threshold trades fallback for action age | [`../evidence/orin-smolvla-chunk-scheduler-08.json`](../evidence/orin-smolvla-chunk-scheduler-08.json) |
| `VLA-LIBERO-EAGER-SUCCESS` | 0.2 ratio | 50 official LIBERO Spatial episodes, frozen seed-43 policy | [`../evidence/smolvla-quality-gate.json`](../evidence/smolvla-quality-gate.json) |
| `VLA-LIBERO-COMPILE-SUCCESS` | 0.22 ratio | 50 official LIBERO Spatial episodes, reduce-overhead | [`../evidence/smolvla-quality-gate.json`](../evidence/smolvla-quality-gate.json) |
| `VLA-LIBERO-COMPILE-DELTA` | 0.02 ratio | paired episode delta; acceptance threshold +/-5 percentage points | [`../evidence/smolvla-quality-gate.json`](../evidence/smolvla-quality-gate.json) |
| `VLA-LIBERO-RESIZE-SUCCESS` | 0 ratio | 50 official LIBERO Spatial episodes; frozen seed-43 policy; quality gate rejected | [`../evidence/smolvla-resize-quality-gate.json`](../evidence/smolvla-resize-quality-gate.json) |
| `VLA-LIBERO-RESIZE-DELTA` | -0.2 ratio | paired episode delta; absolute +/-5 percentage-point gate rejected | [`../evidence/smolvla-resize-quality-gate.json`](../evidence/smolvla-resize-quality-gate.json) |
| `VLA-LIBERO-RESIZE384-FT-SUCCESS` | 0.1 ratio | 6000-step rank-32 LoRA recovery candidate; 50 paired official LIBERO Spatial episodes; quality gate rejected | [`../evidence/smolvla-resize384-finetune-quality-gate.json`](../evidence/smolvla-resize384-finetune-quality-gate.json) |
| `VLA-LIBERO-RESIZE384-FT-DELTA` | -0.1 ratio | paired episode delta; candidate required at least 15% success and was rejected at 10% | [`../evidence/smolvla-resize384-finetune-quality-gate.json`](../evidence/smolvla-resize384-finetune-quality-gate.json) |
| `VLA-ORIN-INT8-WO-P50` | 1415.67 ms | physical Orin, synthetic input, TorchAO targeted action-expert linear layers; latency gate rejected | [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json) |
| `VLA-ORIN-INT8-DYNAMIC-P50` | 5450.43 ms | physical Orin, synthetic input, TorchAO targeted action-expert linear layers; latency gate rejected | [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json) |
| `VLA-ORIN-INT8-WO-ACTION-MAX` | 0.0245989 abs | correctness gate passed; promotion rejected because P50 was slower than baseline | [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json) |
| `VLA-ORIN-INT8-DYNAMIC-ACTION-MAX` | 0.025344 abs | correctness gate passed; promotion rejected because P50 was slower than baseline | [`../evidence/orin-smolvla-quantization.json`](../evidence/orin-smolvla-quantization.json) |
| `VLA-TRACE-05-DELTA` | -0.16 ratio | paired LIBERO simulation with Orin latency envelope; not physical robot | [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json) |
| `VLA-TRACE-08-DELTA` | -0.36 ratio | paired LIBERO simulation with Orin latency envelope; not physical robot | [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json) |
| `VLA-TRACE-05-FALLBACK` | 3618 count | 25 paired LIBERO episodes; latency envelope injection | [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json) |
| `VLA-TRACE-08-FALLBACK` | 4028 count | 25 paired LIBERO episodes; latency envelope injection | [`../evidence/smolvla-orin-latency-envelope.json`](../evidence/smolvla-orin-latency-envelope.json) |
| `VLA-REPLAY-FRESH` | 0.0266667 ratio | offline timing replay, not task execution | [`../evidence/control-replay.json`](../evidence/control-replay.json) |
