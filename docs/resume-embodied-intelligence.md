# Embodied intelligence resume entries

- Reconciled three original audits of a pinned LeRobot LIBERO subset (432 episodes, 52,970 frames, 377/377 identical parquet hashes), then trained SmolVLA LoRA policies across three seeds and reported 13.3% main-condition success with two-stage bootstrap intervals.
- Ran a paired 50-episode official LIBERO screen for the frozen seed-43 policy: eager 20.0% versus `torch.compile(reduce-overhead)` 22.0%, delta +2.0%; the point-estimate rule passed, but the wide CI does not establish statistical non-inferiority.
- Tested direct visual downsampling on the same frozen policy: 512x512 eager 20.0% versus 256x256 0.0%, delta -20.0%; rejected the candidate under a paired +/-5 percentage-point gate.
- Trained a 6000-step 384x384 LoRA recovery candidate and evaluated 50 paired LIBERO episodes; success remained 10.0% versus 20.0% eager, so stopped the planned three-seed/20k expansion at the frozen 15% gate.
- Benchmarked targeted TorchAO INT8 weight-only/dynamic paths on physical Orin; both met action-difference limits but regressed first-action P50 from 1202 ms to 1416/5450 ms and were rejected.
- Profiled the physical-Orin SmolVLA path and attributed 60.6% of full inference to denoise VLM forward and 27.0% to prefix embedding, narrowing the next gate to kernel/export-level analysis.
- Replayed the 50-action scheduler on physical Orin: raising refill threshold from 0.5 to 0.8 increased fresh-action ticks from 68.7% to 86.7% but doubled stale ticks from 100 to 201.
- Injected a measured Orin latency envelope into paired LIBERO closed-loop runs at two success thresholds; async success regressed by 16%/36%, so both scheduler variants were rejected instead of promoting latency-only gains.
- Deployed the frozen SmolVLA PyTorch path to Jetson AGX Orin and measured 1190 ms first-action P50, 1192 ms steady P50 and 0.837 Hz; used profiling and deadline replay to reject a misleading 30 Hz claim and define a kernel/export-level gate.

Interview boundary: Orin used synthetic observations; the latency-envelope task results are LIBERO simulation, not physical-robot success.
