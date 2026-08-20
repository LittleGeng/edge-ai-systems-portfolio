# SmolVLA edge feasibility study

## What I tested

I fine-tuned SmolVLA LoRA policies with three seeds and measured **13.3%** success on the main condition and **18.1%** on a secondary condition. The secondary tasks were excluded from model selection but were included in the final fine-tuning data, so this is not a zero-shot transfer result.

The selected PyTorch policy was then run on Jetson AGX Orin with synthetic observations. First-action P50 was **1190.17 ms**, steady-chunk P50 was **1191.74 ms**, and throughput was **0.837 Hz**. All **500/500** measurements missed a 30 Hz target, so this implementation is not suitable for direct 30 Hz control.

## Work completed

- Cross-checked three August 4 dataset records. All three contain 432 selected episodes, 52,970 frames, and the same 377/377 parquet shard hashes.
- Screened hyperparameters, then trained and evaluated three seeds.
- Selected one seed-43 policy before running the asynchronous and Orin experiments.
- Measured first-action and steady action-chunk latency separately.
- Compared eager execution and `torch.compile(reduce-overhead)` on the same 50 LIBERO episodes: **20.0%** versus **22.0%**, a paired difference of **+2.0 percentage points**. The 95% confidence interval of [-10.0%, +14.0%] is too wide to establish non-inferiority.
- Tested 256x256 input on the same episodes. Success fell from **20.0%** to **0.0%**, so direct downsampling was not kept.
- Fine-tuned a 384x384 LoRA candidate for 6000 steps. It reached **10.0%**, below the preset 15% level for expanding the run to three seeds and 20k steps, so I stopped there.
- Tested targeted TorchAO INT8 weight-only and dynamic quantization on Orin. Both stayed within the chosen action-difference limits, but both were slower than the **1202.12 ms** eager baseline.

## Runtime findings

With fixed synthetic input, `torch.compile(reduce-overhead)` reduced first-action P50 from **1287.54 ms** to **1005.55 ms**, or **21.9%**, while peak RAM rose from **7813 MB** to **8912 MB**. This is a useful speedup, but the resulting rate is still only about 1 Hz.

A 10-step CUDA-event profile attributes **744.92 ms (60.60%)** to denoise VLM forward, **332.45 ms (27.04%)** to prefix embedding, and **96.90 ms (7.88%)** to prefix KV forward. Denoise VLM is part of the denoising-step total, so those rows should not be added together. The profile points to kernel or export-level analysis of denoise VLM and prefix embedding as the next useful optimization work.

Directly reducing the input to 256x256 was faster on fixed input, but the paired LIBERO result dropped from **10/50** successes to **0/50**. A 384x384, rank-32 LoRA follow-up trained for 6000 steps recovered only **5/50** successes. These results suggest that a lower-resolution path would need stronger retraining or distillation before it could be considered for deployment.

For targeted INT8, weight-only measured **1415.67 ms** and dynamic quantization **5450.43 ms** first-action P50, compared with **1202.12 ms** for eager execution. The result applies only to this model partition, TorchAO version, Jetson runtime, and synthetic input; it does not imply that INT8 is generally slower on Orin.

## Scheduling findings

I replayed the 50-action chunk scheduler for 300 ticks on physical Orin with synthetic observations. Raising the refill threshold from 0.5 to 0.8 increased fresh-action ticks from **68.7%** to **86.7%** and reduced fallback ticks from 94 to 40, but stale ticks increased from 100 to 201. Buffering changes the balance between queue underflow and action age; it does not make the policy itself produce actions faster.

I then injected the measured Orin P50/P95 latency range into paired LIBERO runs. At threshold 0.5, synchronous and asynchronous success were **24%** and **8%**; at threshold 0.8, they were **40%** and **4%**. These simulator results show that the current asynchronous setup loses task quality under the measured latency range.

## Source results

- [Dataset records and split hashes](../evidence/smolvla-dataset-audit.json)
- [Three-seed evaluation](../evidence/smolvla-quality.json)
- [Asynchronous runtime](../evidence/smolvla-async-runtime.json)
- [Orin timing](../evidence/smolvla-orin.json)
- [Orin stage profile](../evidence/orin-smolvla-strengthening.json)
- [Scheduler replay at threshold 0.5](../evidence/orin-smolvla-chunk-scheduler-05.json)
- [Scheduler replay at threshold 0.8](../evidence/orin-smolvla-chunk-scheduler-08.json)
- [Offline control replay](../evidence/control-replay.json)
- [Eager and compile task comparison](../evidence/smolvla-quality-gate.json)
- [256x256 task comparison](../evidence/smolvla-resize-quality-gate.json)
- [384x384 fine-tuning result](../evidence/smolvla-resize384-finetune-quality-gate.json)
- [Targeted INT8 result](../evidence/orin-smolvla-quantization.json)
- [Latency-injected LIBERO runs](../evidence/smolvla-orin-latency-envelope.json)

## What the results do not show

Orin runtime used synthetic observations, and the latency-injected task runs were performed in LIBERO simulation. There was no physical-robot closed-loop success test. The compile comparison's point estimate is within the preset +/-5-point range, but its confidence interval cannot establish non-inferiority or an improvement. The dataset checks cover parquet rows and shard identity, not decoded video pixels.
