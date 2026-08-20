# AI Systems and Embodied Intelligence Technical Report

This report follows two robot-learning workloads from official simulator evaluation and RTX 4090 optimization to deployment on Jetson AGX Orin. All experiments were completed by **2026-08-07**. Work on August 8-9 only synchronized, inventoried, and hash-checked existing files; it did not produce new experimental results. Every measured value can be traced through the [result and evidence index](docs/evidence-map.md).

## 1. Project overview

The two projects focus on different engineering questions:

1. **EdgeDiffusion:** evaluate a Diffusion Policy checkpoint in official Push-T simulation, compare eager execution, `torch.compile`, and TensorRT on one fixed RTX 4090, and deploy an FP16 TensorRT U-Net engine on Jetson AGX Orin.
2. **SmolVLA Edge:** evaluate a LoRA policy on official LIBERO Spatial tasks, measure it on physical Orin with synthetic observations, and test `torch.compile`, input resolution, INT8 quantization, and action-chunk scheduling.

The main EdgeDiffusion result is a reduction in 100-step denoising P50 from **566.71 ms** to **235.80 ms** on the same RTX 4090, a **2.40x** speedup. One U-Net call on Orin measured **3.37/4.41 ms P50/P95** with **0/500** misses against a 50 ms target. The complete scheduler-plus-U-Net loop, however, took **557.81/568.67 ms** at 100 steps. Reducing it to 6 steps reached **32.43/36.28 ms**, but the corresponding Push-T high-quality completion rate fell to 0%. None of the measured settings met both timing and task-quality requirements.

The main SmolVLA conclusion is that the current PyTorch path reaches only **0.837 Hz** and misses **500/500** measurements against a 30 Hz target. `torch.compile` reduces first-action latency on fixed synthetic input, but remains far from real-time control and the task-quality confidence interval is too wide to establish non-inferiority. Direct 256x256 resizing, a 6000-step 384x384 fine-tune, and targeted TorchAO INT8 also did not produce results worth carrying forward.

## 2. Questions addressed

| Question | Experiment | Where the result applies |
|---|---|---|
| How well does the policy perform in the official simulator? | Push-T and LIBERO task evaluations | Simulator success is not physical-robot success. |
| Can a different backend reduce latency for the same workload? | Eager, compile, and TensorRT on a fixed RTX 4090 | Numbers apply only to the specified model, input shape, step count, and node. |
| Can inference on Orin meet a local timing target? | U-Net, complete denoising-loop, and policy measurements | Each result states whether the scheduler is included; full robot I/O is excluded. |
| Does a faster option retain task quality? | Matched Push-T and LIBERO episode comparisons | Speed and task quality are measured separately and then considered together. |

The results should not be generalized to every VLA model, GPU, or physical robot system.

## 3. Experimental setup

- Diffusion Policy repository revision: `5ba07ac6661db573af695b419a7947ecb704690f`.
- Diffusion Policy checkpoint SHA256: `f804e16575e261fa0b7e981da3f67741fc8517817734320d550e43a4182bf876`.
- Target board: Jetson AGX Orin 64GB running L4T 36.5.0.
- EdgeDiffusion base quality run: 60 official-checkpoint Push-T episodes with unique environment seeds.
- EdgeDiffusion step comparison: 20 matched Push-T episodes for each of 100, 20, 10, 8, and 6 denoising steps.
- SmolVLA compile comparison: 10 official LIBERO Spatial tasks with 5 episodes each; eager and compile use the same episode IDs, selected seed-43 policy, and 50 action steps.
- SmolVLA resolution comparison: the same 50 episodes, changing only `resize_imgs_with_padding` from 512x512 to 256x256.
- SmolVLA 384px fine-tune: rank 32, learning rate `5e-4`, seed 43, and 6000 LoRA steps; expansion required at least 15% success.
- SmolVLA INT8: TorchAO v0.13.0 at revision `e318546f9be8c6dd1340157cd14e5cbc6ffa1f65`; 5 warmups and 20 measured synthetic-input iterations per variant on physical Orin; Q/V projections carrying LoRA adapters were excluded.
- SmolVLA scheduling: paired LIBERO simulation at thresholds 0.5 and 0.8 with physical-Orin P50/P95 latency endpoints; this is neither a physical-robot run nor a replay of every raw timestamp.
- Orin measurements record first-action and steady action-chunk timing separately and retain raw timing CSV files and telemetry.
- The public repository excludes credentials, private endpoints, billing data, interview Q&A, and private machine paths.

### 3.1 Environment versions

| Workload | Hardware and system | Software, model, and artifact versions |
|---|---|---|
| EdgeDiffusion cloud benchmark | NVIDIA GeForce RTX 4090; CUDA 12.4 | PyTorch 2.5.1+cu124; Torch-TensorRT 2.5.0+cu124; TensorRT 10.3.0; Diffusers 0.11.1 |
| EdgeDiffusion target board | Jetson AGX Orin 64GB; L4T 36.5.0 | TensorRT 10.11.0.33; PyTorch 2.8.0a0+5228986c39.nv25.06; engine `770a439e18d2...` |
| SmolVLA target board | Jetson AGX Orin 64GB; synthetic observations | LeRobot `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`; base model `c83c3163b8ca9b7e67c509fffd9121e66cb96205`; policy `435185f15d31...` |
| Push-T quality evaluation | Official simulator protocol `edge-evaluation-protocol-v1` | repository `5ba07ac6661db573af695b419a7947ecb704690f`; checkpoint `f804e16575e2...` |

## 4. EdgeDiffusion results

### 4.1 Push-T task quality

The official checkpoint achieved **95.0%** high-quality completion (57/60) with a mean score of **0.959**. A separate 20-episode rerun used the same seeds as its reference window and reproduced the same **90.0%** result. The 20-episode rerun checks repeatability; it should not be equated with the separate 60-episode aggregate.

### 4.2 Backend comparison on a fixed RTX 4090

All three backends used the same model, input shape, and 100-step denoising workload. Each setting ran two warmups followed by 30 measured loops, with power sampled every 0.1 seconds. Timing began after conditional tensors had been constructed.

| Backend | P50 / P95 | Throughput from P50 | Incremental peak allocation | Mean sampled GPU power | Maximum denoising error |
|---|---:|---:|---:|---:|---:|
| Eager | 566.71 / 570.36 ms | 1.765 /s | 20.37 MiB | 96.67 W | 0.000000 |
| `torch.compile` | 347.34 / 356.44 ms | 2.879 /s | 20.37 MiB | 109.62 W | 0.000066 |
| TensorRT | 235.80 / 237.93 ms | 4.241 /s | 0.01 MiB | 131.41 W | 0.000100 |

TensorRT is **2.40x** faster by the same-node P50 ratio. Numerical error, Push-T task quality, and latency were measured separately so that a faster backend would not be retained if its behavior had changed too much.

### 4.3 One U-Net call on Orin

The FP16 engine was built directly on physical Orin. It passed all 20 correctness cases with a maximum absolute error of **0.011283**. One U-Net invocation measured **3.37 ms P50** and **4.41 ms P95**, equivalent to **280.87 calls/s**, with no measurement above 50 ms in 500 iterations.

This number covers **one U-Net invocation only**. It excludes repeated scheduler calls, observation processing, simulation, cameras, actuators, networking, and safety logic. I also do not report total energy per action because the available telemetry did not expose a confirmed total-module rail that was independent of the component rails.

### 4.4 Denoising steps and task quality

The 100-step setting is the quality reference. The 20, 10, 8, and 6-step variants use the same 20 Push-T seeds. The preset limits were 0.05 for mean-score difference and 5 percentage points for completion-rate difference.

| Denoising steps | Controller P50 | Controller P95 | Mean score | High-quality completion | Task quality retained |
|---:|---:|---:|---:|---:|---|
| 100 | 1076.5 ms | 1120.3 ms | 0.922 | 90% | Yes |
| 20 | 217.6 ms | 236.2 ms | 0.119 | 0% | No |
| 10 | 88.4 ms | 96.8 ms | 0.093 | 0% | No |
| 8 | 80.4 ms | 90.5 ms | 0.067 | 0% | No |
| 6 | 65.2 ms | 81.4 ms | 0.104 | 0% | No |

Reducing the step count clearly improved latency, but all four shorter settings fell to **0%** high-quality completion. They show the speed-quality tradeoff, not a deployment option.

### 4.5 Complete denoising loop on Orin

The TensorRT U-Net was then placed inside the DDPM scheduler. Each setting ran three warmups followed by 20 measured loops.

| Denoising steps | Orin loop P50 | Orin loop P95 | Above 50 ms | Timing result | Conclusion with task quality included |
|---:|---:|---:|---:|---|---|
| 100 | 557.81 ms | 568.67 ms | 20/20 | Target not met | Retains task quality, but too slow |
| 20 | 110.80 ms | 121.96 ms | 20/20 | Target not met | Push-T quality loss is too large |
| 10 | 54.88 ms | 64.15 ms | 20/20 | Target not met | Push-T quality loss is too large |
| 8 | 43.68 ms | 50.01 ms | 1/20 | Target not met consistently | Push-T quality loss is too large |
| 6 | 32.43 ms | 36.28 ms | 0/20 | Target met | Push-T quality loss is too large |

The 6-step loop meets the 50 ms timing target, but the corresponding Push-T high-quality completion rate is 0%. The 100-step setting preserves quality but misses 50 ms every time. None of the measured step counts satisfies both conditions. The loop still uses synthetic, fixed-shape tensors and excludes observation, simulator, and robot I/O.

## 5. SmolVLA Edge results

### 5.1 Task quality and dataset scope

Three LoRA fine-tuning seeds produced **13.3%** success over 45 main-condition runs and **18.1%** over 105 secondary-condition runs. The secondary tasks were not used for model selection but were included in the final fine-tuning data, so I do not describe this as zero-shot transfer.

| Condition | Unique episodes / runs | Per-seed success | Two-stage bootstrap 95% CI |
|---|---:|---|---:|
| Main selection tasks | 15 / 45 | 42: 13.3%, 43: 6.7%, 44: 20.0% | [2.2%, 26.7%] |
| Secondary condition | 35 / 105 | 42: 8.6%, 43: 20.0%, 44: 25.7% | [7.6%, 29.5%] |

Three original records from August 4 each contain **432 selected episodes**, **52,970 frames**, and the same **377/377 parquet shard hashes**. File presence, frame index, task index, timestamps, and metadata counts had no missing files or failed rows. The `verify-only` operation at 2026-08-10T08:19:32+08:00 checked the synchronized archive and did not rerun the experiment. This check covers parquet rows and shard identity, not decoded video pixels or task success.

### 5.2 Orin runtime and profiling

The SmolVLA PyTorch path used synthetic observations on Orin. First-action P50 was **1190.17 ms**, steady-chunk P50 was **1191.74 ms**, and the resulting control rate was **0.837 Hz**. All 500 measurements exceeded the timing requirement for 30 Hz.

In a separate fixed-input comparison, eager first-action P50 was **1287.54 ms** and `torch.compile(reduce-overhead)` was **1005.55 ms**, a **21.9%** reduction. Peak RAM increased from **7813 MB** to **8912 MB**. Compile provides a meaningful improvement, but both settings remain far too slow for 30 Hz.

| Variant | First-action P50 | First-action P95 | Rate | 30 Hz misses | Peak RAM | Max action difference vs eager | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline-fixed` | 1287.54 ms | 1322.35 ms | 0.773 Hz | 20/20 | 7813 MB | 0.0000 | Reference |
| `amp-fixed` | 1242.39 ms | 1274.36 ms | 0.804 Hz | 20/20 | 7785 MB | 0.0225 | Small improvement, still not real time |
| `resize256-fixed` | 996.67 ms | 1033.76 ms | 0.998 Hz | 20/20 | 7694 MB | 2.0072 | Later task evaluation showed unacceptable quality loss |
| `compile-reduce-overhead-fixed` | 1005.55 ms | 1025.40 ms | 0.996 Hz | 20/20 | 8912 MB | Point estimate within range, but CI is inconclusive and timing is still insufficient |

A CUDA-event stage profile used 20 fixed-input calls and 10 denoising steps. The representative action matched the baseline exactly.

| Stage | Mean per inference | Share of full inference | Relationship |
|---|---:|---:|---|
| Prefix embedding | 332.45 ms | 27.04% | Top level |
| Prefix KV forward | 96.90 ms | 7.88% | Top level |
| Denoising-step total | 785.77 ms | 63.92% | Top level |
| Denoise VLM forward | 744.92 ms | 60.60% | Included within the denoising-step total |
| Unattributed | 14.16 ms | - | Remaining time |

Denoise VLM forward accounts for **60.60%** of full inference and prefix embedding for **27.04%**. Kernel and export-level analysis of these two stages is therefore the most useful next optimization step. Because Denoise VLM forward is included in the denoising-step total, the nested rows must not be added together.

### 5.3 Targeted TorchAO INT8

Post-training quantization covered 80 K/O/MLP linear layers in the SmolVLA action expert and excluded Q/V projections carrying LoRA. Both options stayed within the preset action-difference limits, but neither achieved the required 5% reduction in first-action latency.

| Variant | First-action P50/P95 | Steady P50/P95 | Rate | Peak RAM | Action max/mean absolute difference | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| Eager baseline | 1202.12/1245.03 ms | 1203.49/1236.71 ms | 0.829 Hz | 8250 MB | 0/0 | Reference |
| INT8 weight-only | 1415.67/1457.57 ms | 1417.84/1445.21 ms | 0.705 Hz | 8242 MB | 0.02460/0.00524 | Slower than the baseline |
| INT8 dynamic | 5450.43/5498.47 ms | 5427.35/5481.92 ms | 0.184 Hz | 8265 MB | 0.02534/0.00737 | Slower than the baseline |

By first-action P50, weight-only was **17.8% slower** and dynamic INT8 was **353.4% slower** than eager. This result applies only to the tested model partition, TorchAO revision, NVIDIA 25.06 iGPU container, and synthetic input. It does not imply that INT8 is generally slower on Orin.

### 5.4 Eager and compile task quality

Both settings used the same policy and the same 50 episodes:

- Eager: **10/50 = 20.0%**.
- `torch.compile(reduce-overhead)`: **11/50 = 22.0%**.
- Paired difference: **+2.0 percentage points**.
- Bootstrap 95% CI: **[-10.0%, +14.0%]**.
- Preset initial screening range: absolute difference no greater than 5 percentage points; the point estimate is inside this range.

The point estimate does not show an obvious regression, but the confidence interval extends beyond +/-5 percentage points. It therefore cannot establish statistical non-inferiority or superiority, and the absolute success rate remains low.

### 5.5 Direct resize to 256x256

The selected seed-43 policy, LeRobot revision, and episode IDs were unchanged; only `resize_imgs_with_padding` changed from 512x512 to 256x256. Success fell from **10/50 = 20.0%** at eager 512x512 to **0/50 = 0.0%** at 256x256. The paired difference was **-20.0 percentage points**, with a bootstrap 95% CI of **[-32.0%, -10.0%]**.

Lower resolution was faster in the fixed-input test, but the current checkpoint was not robust to this change. Direct 256x256 input is therefore not useful here. Continuing this direction would require retraining or distillation followed by the same matched task evaluation.

### 5.6 The 384x384 fine-tuning attempt

The candidate used rank 32, learning rate `5e-4`, seed 43, and 6000 LoRA steps at 384x384. On the same 50 LIBERO Spatial episodes, it achieved **5/50 = 10.0%** compared with **10/50 = 20.0%** for eager 512x512. The paired difference was **-10.0 percentage points**, with a bootstrap 95% CI of **[-24.0%, +2.0%]**.

Expanding to three seeds and 20k steps required at least **15.0%** success. The measured **10.0%** did not meet that condition, so I did not spend additional compute on the larger run.

### 5.7 Action-chunk scheduling

The 50-action chunk scheduler was replayed for 300 ticks on physical Orin with synthetic observations.

| Refill threshold | Chunk P50/P95 | Fresh-action ticks | Fallback/underflow ticks | Stale ticks |
|---:|---:|---:|---:|---:|
| 0.5 | 1302.48/1354.78 ms | 68.7% | 94 | 100 |
| 0.8 | 1325.58/1362.75 ms | 86.7% | 40 | 201 |

Raising the refill threshold increased fresh-action availability from **68.7%** to **86.7%** and reduced fallback ticks from **94** to **40**, but stale ticks rose from **100** to **201**. Buffering can trade queue underflow for action age, but it cannot increase the policy's output rate. This was a synthetic-input replay, not a physical-robot task-success test.

A separate offline 30 Hz replay produced **8/300** fresh-action ticks and **284** safe-fallback ticks. It used measured Orin timing records and was not a new simulator experiment.

### 5.8 LIBERO results with Orin latency injected

To observe how board-scale latency affects task quality with asynchronous scheduling, the physical Orin P50/P95 endpoints were injected into paired LIBERO closed-loop runs. Each threshold used 25 matched episodes.

| Success threshold | Synchronous success | Asynchronous success | Async minus sync | Fallback / queue-empty ticks | Conclusion |
|---:|---:|---:|---:|---:|---|
| 0.5 | 24% | 8% | -16% | 3618 | Quality loss exceeded the preset range |
| 0.8 | 40% | 4% | -36% | 4028 | Quality loss exceeded the preset range |

A higher refill threshold reduced underflow but increased action age, and the latency-injected LIBERO runs still lost task success. The experiment used P50/P95 summary endpoints rather than every raw per-chunk timestamp. It is neither a full raw-trace replay nor a physical-robot experiment.

## 6. Engineering tradeoffs

1. **Measure speed and task quality separately.** A faster backend is only worth keeping when its numerical and task behavior remain acceptable.
2. **Distinguish local inference from complete control.** U-Net and policy timing do not include all sensor, actuator, network, and safety-loop costs.
3. **Keep experiments that did not work.** SmolVLA misses 30 Hz, and lower step counts, lower resolution, and INT8 all showed clear problems. These results directly determine what to try next and what to stop.
4. **Do not estimate energy without suitable measurements.** Energy per action remains empty when there is no reliable total-module power rail.
5. **Repeat the key results.** Rebuilding the Orin engine and repeating the Push-T seed window checked that the core results could be reproduced.
6. **Stop based on intermediate evidence.** The 384x384 candidate stayed below the 15% continuation level, so the more expensive three-seed, 20k-step run was not started.
7. **Do not equate smaller weights with lower latency.** The tested INT8 options reduced estimated weight payload and CUDA allocation but increased measured end-to-end action latency.

For EdgeDiffusion, the next useful direction is a distilled or retrained low-step policy compared with the 100-step quality reference. For SmolVLA, denoise VLM forward and prefix embedding should receive kernel or export-level analysis before investing in TensorRT, ONNX, or quantization-aware training.

## 7. Reproducing the results

From the repository root, run:

```bash
python3 reproduction/package/reproduce.py
```

The script recalculates the summary from the included result files and checks timing replay, dataset records, hashes, and task completion. It does not rerun model inference and needs neither cloud credentials nor active AutoDL or Orin resources.

The public repository can also be checked with:

```bash
python3 scripts/verify_repository.py .
```

The current check covers **36/36** completed tasks, with no partial tasks and no invalid resource references. Every headline value, its measurement conditions, and its source file are listed in the [result and evidence index](docs/evidence-map.md).

## 8. Conclusion

This project does not show that a VLA already runs at 30 Hz on Orin. It demonstrates a more specific engineering workflow: evaluate robot-learning policies under fixed versions and explicit conditions, compare optimization backends on an RTX 4090, deploy the models to a physical Orin board, and use task quality, complete-loop timing, and unsuccessful experiments to decide what is worth continuing.

The final results include a **2.40x** TensorRT speedup and millisecond-scale timing for one U-Net invocation, as well as clear evidence that the current SmolVLA path is not real time and that fewer denoising steps, direct resizing, and the tested INT8 paths are not suitable. Those tradeoffs are the main systems-engineering outcome of the work.
