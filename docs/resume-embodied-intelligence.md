# Resume bullets for embodied intelligence roles

- Checked three records of a fixed LeRobot LIBERO subset, confirming 432 episodes, 52,970 frames, and identical hashes for all 377 parquet shards; then trained SmolVLA LoRA policies with three seeds and measured 13.3% main-condition success with two-stage bootstrap intervals.
- Compared eager execution and `torch.compile(reduce-overhead)` on the same 50 official LIBERO episodes: 20.0% versus 22.0%, a +2.0-point difference. The point estimate is within the preset +/-5-point screening range, but the wide confidence interval does not establish non-inferiority.
- Tested 256x256 visual input on the same policy and episodes. Success fell from 20.0% at 512x512 to 0.0%, so I did not keep direct downsampling as a deployment option.
- Trained a 6000-step, 384x384 LoRA follow-up and evaluated the same 50 episodes. It reached 10.0% success versus 20.0% for the 512x512 reference, so I stopped before the planned three-seed, 20k-step expansion.
- Benchmarked targeted TorchAO INT8 weight-only and dynamic quantization on physical Orin. Both stayed within the chosen action-difference limits but increased first-action P50 from 1202 ms to 1416 and 5450 ms.
- Profiled SmolVLA on physical Orin and found that denoise VLM forward accounts for 60.6% of full inference and prefix embedding for 27.0%, focusing the next optimization work on kernel and export-level analysis.
- Replayed the 50-action scheduler on physical Orin. Raising the refill threshold from 0.5 to 0.8 increased fresh-action ticks from 68.7% to 86.7%, but stale ticks doubled from 100 to 201.
- Injected the measured Orin latency range into paired LIBERO closed-loop runs. Asynchronous success fell by 16 and 36 percentage points at the two tested thresholds, showing that scheduler tuning alone did not preserve task quality.
- Ran the selected SmolVLA PyTorch policy on Jetson AGX Orin and measured 1190 ms first-action P50, 1192 ms steady P50, and 0.837 Hz. Stage profiling and deadline replay showed that this path cannot support direct 30 Hz control.

Orin measurements used synthetic observations. The latency-injected task results come from LIBERO simulation, not physical-robot success tests.
