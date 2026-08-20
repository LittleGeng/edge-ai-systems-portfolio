# AI Systems and Embodied Intelligence Technical Report

**Report ID:** `job-portfolio-v1`  
**Status:** frozen evidence synthesis  
**Experiment execution through:** 2026-08-07  
**Materials refreshed:** 2026-08-09T11:06:59+08:00  
**Archive verification:** `autodl-release-20260808` at 2026-08-09T09:07:35+08:00 (`verify-only`; August 8-9 were synchronization and verification only)  
**Publication audit:** 36/36 completed tasks ready  
**Evidence policy:** measured values are tied to raw artifacts and SHA256 manifests; unsupported values remain explicit limitations.

## 1. Executive summary

This report evaluates two complementary systems paths for industrial AI systems and embodied-intelligence roles:

1. **EdgeDiffusion:** a frozen Diffusion Policy checkpoint was evaluated in official Push-T simulation, optimized on a fixed RTX 4090, and deployed as a board-native FP16 TensorRT U-Net engine on Jetson AGX Orin.
2. **SmolVLA Edge:** a frozen LoRA policy was evaluated on official LIBERO Spatial tasks, measured on physical Orin with synthetic observations, and screened with a paired `torch.compile(reduce-overhead)` task-quality gate.

The strongest positive systems result is a **2.40x reduction** in fixed-node 100-step denoising P50 (566.71 ms to 235.80 ms) plus a board-native Orin U-Net result of **3.37/4.41 ms P50/P95** with **0/500** misses at a 50 ms inference deadline. The corresponding Orin scheduler-plus-U-Net loop is much slower at 100 steps (**557.81/568.67 ms**); reducing it to 6 steps meets the local 50 ms timing gate but is rejected because the paired Push-T quality screen falls to 0% high-quality completion.

The most important negative result is that the current SmolVLA PyTorch path reaches only **0.837 Hz** and misses **500/500** 30 Hz deadlines. `torch.compile` lowers first-action P50 in a fixed synthetic-input strengthening run, but does not establish 30 Hz control or a task-quality win. Direct 256x256 downsampling failed, a 6000-step 384x384 recovery candidate still measured only **10.0%** paired success, and targeted TorchAO weight-only/dynamic INT8 regressed first-action P50 to **1415.67/5450.43 ms** versus **1202.12 ms** eager.

## 2. Questions and claim discipline

The experiments answer four bounded questions:

| Question | Answered by | Claim boundary |
|---|---|---|
| Does the frozen policy perform its task in the official simulator? | Push-T and LIBERO quality runs | Simulator quality is not physical-robot success. |
| Does a backend reduce the same workload latency? | Fixed RTX 4090 eager/compile/TensorRT comparison | Numbers apply to the pinned model, shape, step count and node. |
| Can a target-board engine meet a local inference deadline? | Orin correctness and timing runs | U-Net or policy inference scope is stated; full robot I/O is excluded. |
| Does optimization preserve task quality? | Paired 50-episode LIBERO gate | A passed screening gate is not proof of improvement or statistical superiority. |

No result is presented as a general claim about all VLA models, all GPUs, or a physical robot.

## 3. Frozen inputs and protocol

- Diffusion Policy repository revision: `5ba07ac6661db573af695b419a7947ecb704690f`.
- Diffusion Policy checkpoint SHA256: `f804e16575e261fa0b7e981da3f67741fc8517817734320d550e43a4182bf876`.
- Target board: Jetson AGX Orin 64GB, L4T 36.5.0 evidence path.
- Edge quality: 60 official-checkpoint Push-T episodes with unique environment seeds.
- EdgeDiffusion step quality: 20 paired Push-T episodes per tested denoising-step setting (100/20/10/8/6), using node-matched reference windows.
- SmolVLA quality: paired official LIBERO Spatial protocol, 10 tasks × 5 episodes, identical episode IDs for eager and compile, frozen seed-43 policy, 50 action steps.
- SmolVLA resize quality: the same 50 paired LIBERO Spatial episode IDs, with only `resize_imgs_with_padding` changed from 512x512 to 256x256.
- SmolVLA resize recovery: a 6000-step rank-32, learning-rate `5e-4`, seed-43 LoRA candidate at 384x384, evaluated on the same 50 paired episode IDs with a predeclared 15% minimum-success stop gate.
- SmolVLA targeted INT8: TorchAO v0.13.0 at pinned revision `e318546f9be8c6dd1340157cd14e5cbc6ffa1f65`, 5 warmups and 20 measured synthetic-input iterations per variant on physical Orin; Q/V projections carrying LoRA adapters were excluded.
- SmolVLA scheduling quality: paired LIBERO simulation at thresholds 0.5 and 0.8 with a physical-Orin P50/P95 latency envelope; this is not a physical-robot run or a raw timestamp replay.
- SmolVLA dataset audit: three original August 4 audits agree on 432 selected episodes, 52,970 frames and all 377 parquet shard hashes; every recorded row-level integrity gate passed.
- Orin runtime measurements separate first action from steady action chunks and retain raw timing CSV plus telemetry.
- Experiments ended by 2026-08-07. August 8-9 activity was limited to remote-to-local synchronization, inventory and hash verification; it produced no new experimental observations.
- All release artifacts are checked by `SHA256SUMS`; the public project is sanitized and excludes credentials, private endpoints, billing data and interview Q&A.

### 3.1 Environment and artifact identity

| Workload | Hardware/runtime | Frozen identities |
|---|---|---|
| EdgeDiffusion cloud benchmark | NVIDIA GeForce RTX 4090; CUDA 12.4 | PyTorch 2.5.1+cu124; Torch-TensorRT 2.5.0+cu124; TensorRT 10.3.0; Diffusers 0.11.1 |
| EdgeDiffusion target board | Jetson AGX Orin 64GB; L4T 36.5.0 | TensorRT 10.11.0.33; PyTorch 2.8.0a0+5228986c39.nv25.06; engine `770a439e18d2...` |
| SmolVLA target board | Jetson AGX Orin 64GB; synthetic observations | LeRobot `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`; base model `c83c3163b8ca9b7e67c509fffd9121e66cb96205`; policy `435185f15d31...` |
| Push-T quality | official simulator protocol `edge-evaluation-protocol-v1` | repository `5ba07ac6661db573af695b419a7947ecb704690f`; checkpoint `f804e16575e2...` |

## 4. EdgeDiffusion results

### 4.1 Task quality

The official checkpoint reached **95.0%** high-quality completion (57/60) with mean task score **0.959**. An independent same-seed 20-episode rerun reached **90.0%**, exactly matching its 20-episode baseline and passing the predeclared 2 percentage-point rerun tolerance. That tolerance is a paired-window check, not a claim that the 20-episode rate equals the separate 60-episode aggregate.

### 4.2 Fixed RTX 4090 optimization

The same model, input shape and 100-step denoising workload were measured on one fixed RTX 4090. The loop was measured for 30 iterations after two warmups; power was sampled every 0.1 s. The scope starts after condition-tensor construction.

| Backend | P50 / P95 | Throughput from P50 | Incremental peak allocation | Mean sampled GPU power | Max denoising error |
|---|---:|---:|---:|---:|---:|
| Eager | 566.71 / 570.36 ms | 1.765 /s | 20.37 MiB | 96.67 W | 0.000000 |
| `torch.compile` | 347.34 / 356.44 ms | 2.879 /s | 20.37 MiB | 109.62 W | 0.000066 |
| TensorRT | 235.80 / 237.93 ms | 4.241 /s | 0.01 MiB | 131.41 W | 0.000100 |

TensorRT speedup is **2.40x** by the same-node P50 ratio. Numerical correctness and Push-T quality gates were kept separate from timing so a faster but behavior-changing backend could not be promoted.

### 4.3 Orin board-native engine

The FP16 engine was built on physical Orin and passed **20/20** correctness cases, with maximum absolute error **0.011283**. The U-Net wall-time benchmark measured **3.37 ms P50**, **4.41 ms P95**, throughput **280.87 actions/s**, and **0/500** misses at 50 ms.

This is a **single U-Net invocation** timing result. It excludes scheduler repetition, observation processing, simulator, camera capture, actuator I/O, networking and safety logic. Total energy/action is intentionally not reported because the available telemetry did not prove a non-overlapping total-module power rail.

### 4.4 Denoising-step quality/latency sweep

The frozen checkpoint was evaluated on paired 20-episode Push-T windows at five denoising-step counts. The 100-step condition is the quality reference; each lower-step condition is compared with the same-node reference seed window using the predeclared limits of 0.05 mean-score delta and 0.05 completion-rate delta.

| Denoising steps | Controller P50 | Controller P95 | Mean score | High-quality completion | Quality gate |
|---:|---:|---:|---:|---:|---|
| 100 | 1076.5 ms | 1120.3 ms | 0.922 | 90% | PASS |
| 20 | 217.6 ms | 236.2 ms | 0.119 | 0% | REJECT |
| 10 | 88.4 ms | 96.8 ms | 0.093 | 0% | REJECT |
| 8 | 80.4 ms | 90.5 ms | 0.067 | 0% | REJECT |
| 6 | 65.2 ms | 81.4 ms | 0.104 | 0% | REJECT |

Only 100 steps passed the paired quality gate. The 20/10/8/6 settings reduced P50 to **217.6/88.4/80.4/65.2 ms**, respectively, but each produced **0%** high-quality completion in its 20-episode window. These are valid negative systems results and do not justify a lower-step deployment claim.

### 4.5 Complete denoising loop on physical Orin

The board-native TensorRT U-Net was also measured inside the frozen DDPM scheduler for 20 iterations per step count after three warmups:

| Denoising steps | Orin loop P50 | Orin loop P95 | 50 ms misses | Timing status | Cross-evidence deployment decision |
|---:|---:|---:|---:|---|---|
| 100 | 557.81 ms | 568.67 ms | 20/20 | MISS | quality reference; timing misses |
| 20 | 110.80 ms | 121.96 ms | 20/20 | MISS | REJECT: paired Push-T quality gate failed |
| 10 | 54.88 ms | 64.15 ms | 20/20 | MISS | REJECT: paired Push-T quality gate failed |
| 8 | 43.68 ms | 50.01 ms | 1/20 | MISS | REJECT: paired Push-T quality gate failed |
| 6 | 32.43 ms | 36.28 ms | 0/20 | PASS | REJECT: paired Push-T quality gate failed |

The 6-step loop reaches **32.43/36.28 ms** with zero 50 ms misses, but the separate paired Push-T screen reports **0%** high-quality completion. Conversely, the 100-step quality reference misses every Orin timing deadline. This is the central latency-quality conflict: no measured step count satisfies both gates. The board loop uses synthetic fixed-shape tensors and excludes observation, simulator and robot I/O.

## 5. SmolVLA Edge results

### 5.1 Quality and dataset scope

Three LoRA fine-tuning seeds produced **13.3%** main-condition success over 45 runs and **18.1%** on the secondary condition over 105 runs. The secondary tasks were excluded from model selection but were present in final fine-tuning data; this is **not** a zero-shot transfer claim.

| Condition | Unique episodes / runs | Per-seed success | Two-stage bootstrap 95% CI |
|---|---:|---|---:|
| Main selection tasks | 15 / 45 | 42: 13.3%, 43: 6.7%, 44: 20.0% | [2.2%, 26.7%] |
| Secondary condition | 35 / 105 | 42: 8.6%, 43: 20.0%, 44: 25.7% | [7.6%, 29.5%] |

The original VLA-01 audits were captured on August 4 and later synchronized into the local archive. Independent D/E/F records agree on **432 selected episodes**, **52,970 frames**, and **377/377 parquet shard hashes**. All file-presence, frame-index, task-index, timestamp and metadata-count gates passed, with no missing files or failed rows. The archive was verified on 2026-08-10T08:19:32+08:00 in `verify-only` mode; that synchronization did not rerun the experiment. The audit covers parquet rows and shard identity, not decoded video pixels or task success.

### 5.2 Orin runtime baseline and strengthening

The frozen PyTorch path on Orin with synthetic observations measured first-action P50 **1190.17 ms**, steady-chunk P50 **1191.74 ms**, control rate **0.837 Hz**, and **500/500** misses at 30 Hz.

In the fixed-input strengthening comparison, eager first-action P50 was **1287.54 ms** and `torch.compile(reduce-overhead)` was **1005.55 ms**, a **21.9%** reduction. Peak RAM increased from **7813 MB** to **8912 MB**. Both variants still missed the 30 Hz deadline in all strengthening iterations.

The full fixed-input sweep was:

| Variant | First-action P50 | First-action P95 | Rate | 30 Hz misses | Peak RAM | Max action delta vs eager | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline-fixed` | 1287.54 ms | 1322.35 ms | 0.773 Hz | 20/20 | 7813 MB | 0.0000 | reference | 
| `amp-fixed` | 1242.39 ms | 1274.36 ms | 0.804 Hz | 20/20 | 7785 MB | 0.0225 | modest gain; not sufficient | 
| `resize256-fixed` | 996.67 ms | 1033.76 ms | 0.998 Hz | 20/20 | 7694 MB | 2.0072 | rejected by paired task-quality gate | 
| `compile-reduce-overhead-fixed` | 1005.55 ms | 1025.40 ms | 0.996 Hz | 20/20 | 8912 MB | 0.0767 | screening point estimate passed; CI inconclusive; real-time failed | 

The resize-256 variant is faster but changes the representative action materially (**2.0072** max absolute difference), so it is not treated as an optimization win without a task-quality experiment.

The diagnostic stage profile used 20 fixed-input calls with 10 denoising steps and preserved the baseline representative action exactly. CUDA-event attribution identifies the next optimization targets:

| Stage | Mean per inference | Share of full inference | Nesting |
|---|---:|---:|---|
| Prefix embedding | 332.45 ms | 27.04% | top level |
| Prefix KV forward | 96.90 ms | 7.88% | top level |
| Denoise-step total | 785.77 ms | 63.92% | top level |
| Denoise VLM forward | 744.92 ms | 60.60% | nested inside denoise-step total |
| Unattributed | 14.16 ms | - | residual |

Because denoise VLM forward alone accounts for **60.60%** and prefix embedding another **27.04%**, the next step is kernel/export-level analysis of those stages, not another broad component guess. The profile is diagnostic: it uses CUDA events without per-stage synchronization and must not be summed across the nested denoise rows.

### 5.3 Targeted TorchAO INT8 on physical Orin

The post-training quantization study targeted 80 K/O/MLP linear layers in the SmolVLA action expert and excluded LoRA-bearing Q/V projections. Both candidates passed the predeclared action-difference gate, but neither passed the required 5% first-action latency-reduction gate.

| Variant | First-action P50/P95 | Steady P50/P95 | Rate | Peak RAM | Action max/mean abs | Decision |
|---|---:|---:|---:|---:|---:|---|
| Eager baseline | 1202.12/1245.03 ms | 1203.49/1236.71 ms | 0.829 Hz | 8250 MB | 0/0 | reference |
| INT8 weight-only | 1415.67/1457.57 ms | 1417.84/1445.21 ms | 0.705 Hz | 8242 MB | 0.02460/0.00524 | REJECT |
| INT8 dynamic | 5450.43/5498.47 ms | 5427.35/5481.92 ms | 0.184 Hz | 8265 MB | 0.02534/0.00737 | REJECT |

Weight-only was **17.8% slower** and dynamic INT8 **353.4% slower** than eager by first-action P50. The result is bounded to this model partition, TorchAO revision, NVIDIA 25.06 iGPU container and synthetic input. It does not show that INT8 is generally slow on Orin; it shows that these particular post-training paths add more runtime overhead than they remove.

### 5.4 Paired task-quality gate

Using the same 50 episode IDs and the same frozen policy:

- Eager: **10/50 = 20.0%**.
- `torch.compile(reduce-overhead)`: **11/50 = 22.0%**.
- Paired delta: **+2.0%**.
- Bootstrap 95% CI: **[-10.0%, +14.0%]**.
- Predeclared point-estimate gate: absolute delta ≤ 5 percentage points, **passed**.

The point estimate passes the predeclared screening rule, but the wide confidence interval crosses the +/-5 percentage-point region. This does **not** establish statistical non-inferiority or superiority, and the absolute success rate remains low.

### 5.5 Paired visual resize quality gate

Using the same frozen seed-43 policy, LeRobot revision and episode IDs, changing only `resize_imgs_with_padding` from 512x512 to 256x256 produced **0/50 = 0.0%** success versus **10/50 = 20.0%** for eager 512x512. The paired delta was **-20.0%**, with bootstrap 95% CI **[-32.0%, -10.0%]**. The predeclared absolute 5 percentage-point gate **failed**; all 10 eager-only successes were lost and there were zero resize256-only successes.

This is a useful negative systems result. The fixed-input runtime sweep showed a lower-resolution path can be faster, but the frozen checkpoint is not resolution-robust. Direct 256x256 deployment is rejected; a lower-resolution path needs retraining or distillation followed by the same paired task gate.

### 5.6 Fine-tuned 384px recovery stop gate

The recovery candidate used rank 32, learning rate `5e-4`, seed 43 and 6000 LoRA steps at 384x384. On the same 50 paired official LIBERO Spatial episode IDs it reached **5/50 = 10.0%**, versus **10/50 = 20.0%** for eager 512x512. The paired delta was **-10.0%** with bootstrap 95% CI **[-24.0%, +2.0%]**.

The predeclared recovery rule required at least **15.0%** success. Because the candidate achieved only **10.0%**, the planned three-seed and 20k-step expansion was stopped. This is a resource-governance result as well as a quality result: a failed paired gate prevented a much larger compute commitment.

### 5.7 Action-chunk scheduling on physical Orin

The frozen 50-action chunk scheduler was replayed for 300 ticks on physical Orin with synthetic observations:

| Refill threshold | Chunk P50/P95 | Fresh-action ticks | Fallback/underflow ticks | Stale ticks |
|---:|---:|---:|---:|---:|
| 0.5 | 1302.48/1354.78 ms | 68.7% | 94 | 100 |
| 0.8 | 1325.58/1362.75 ms | 86.7% | 40 | 201 |

Raising the refill threshold improves action availability from **68.7%** to **86.7%** and reduces fallback ticks from **94** to **40**, but stale ticks increase from **100** to **201**. Buffering can trade underflow for action age; it cannot create fresh policy outputs or prove closed-loop quality. This is a synthetic-input board replay, not physical-robot task success.

### 5.8 Scheduling and safety interpretation

The offline 30 Hz scheduler replay produced **8/300** fresh-action ticks and **284** safe-fallback ticks. This replay uses measured Orin timing traces; it is not a new simulator run and not physical-robot task success.

### 5.9 Latency-envelope closed-loop quality

To test whether asynchronous scheduling preserves simulator task quality under board-scale latency, the physical Orin P50/P95 endpoints were injected into paired LIBERO closed-loop runs. The two thresholds use 25 paired episodes each.

| Success threshold | Sync success | Async success | Async-minus-sync | Fallback / queue-empty ticks | Gate |
|---:|---:|---:|---:|---:|---|
| 0.5 | 24% | 8% | -16% | 3618 | REJECT |
| 0.8 | 40% | 4% | -36% | 4028 | REJECT |

Both settings violated the predeclared +/-5 percentage-point quality gate. Together with the physical Orin replay, the evidence shows that a higher refill threshold reduces underflow but increases action age, while the injected-latency LIBERO runs still lose task success. The simulator experiment uses P50/P95 summary endpoints rather than raw per-chunk timestamps, so it remains an envelope injection rather than a complete raw-trace replay.

The deployment decision is therefore to reject a direct 30 Hz control claim for the measured eager, compile and targeted post-training INT8 paths. Stage-level profiling has already localized most time to denoise VLM forward and prefix embedding; the next VLA gate is kernel/export-level profiling of those paths before choosing TensorRT/ONNX or quantization-aware retraining. For EdgeDiffusion, the step sweep keeps 100 steps as the quality reference; any lower-step optimization now requires a quality-preserving method such as distillation or retraining.

## 6. Cross-project engineering decisions

1. **Separate quality from speed.** A backend is not promoted from a latency table alone; numerical and task-level gates are required.
2. **Separate local inference from closed-loop control.** U-Net and policy timings do not include all sensor, actuator and safety-loop costs.
3. **Keep negative results.** SmolVLA does not meet the 30 Hz target; the failure is a useful deployment conclusion, not a missing placeholder.
4. **Do not invent energy numbers.** Missing total-module rail evidence produces `null`, not an estimated joules/action value.
5. **Use independent reruns.** REL-03 rebuilt the Orin engine and repeated the Push-T seed window; P50/P95 and task-success tolerances passed.
6. **Treat input resolution as a quality contract.** A 256x256 runtime shortcut was rejected by the paired LIBERO gate despite the fixed-input speed observation; performance-only resize changes are not deployable without retraining or distillation.
7. **Stop recovery work at a frozen gate.** A 6000-step 384px LoRA candidate missed the 15% minimum, so the planned three-seed/20k expansion was not executed.
8. **Do not equate smaller weights with lower latency.** Targeted INT8 reduced the estimated weight payload and CUDA allocation, but both measured variants regressed end-to-end action latency on this Orin stack.
9. **Do not confuse synchronization with experimentation.** August 8-9 activity only copied, inventoried and hash-verified previously generated evidence; each claim retains its original capture time.

## 7. Complete claim index

| Claim | Value | Scope | Evidence |
|---|---:|---|---|
| `EDGE-SUCCESS` | 0.95 ratio | 60 official-checkpoint simulator episodes | `evidence/edge-quality.json` |
| `EDGE-SUCCESS-RERUN` | 0.9 ratio | 20 same-seed official-checkpoint episodes; independent rerun | `evidence/independent-rerun.json` |
| `EDGE-4090-EAGER` | 566.709 ms | fixed RTX 4090 node | `evidence/edge-fixed-node.json` |
| `EDGE-4090-TRT` | 235.798 ms | fixed RTX 4090 node | `evidence/edge-fixed-node.json` |
| `EDGE-4090-SPEEDUP` | 2.40337 x | P50 ratio on same node | `evidence/edge-fixed-node.json` |
| `EDGE-STEPS-100-P50` | 1076.48 ms | 20 paired Push-T episodes; quality gate passed | `evidence/edge-step-quality.json` |
| `EDGE-STEPS-20-P50` | 217.602 ms | 20 paired Push-T episodes; quality gate rejected | `evidence/edge-step-quality.json` |
| `EDGE-STEPS-10-P50` | 88.3921 ms | 20 paired Push-T episodes; quality gate rejected | `evidence/edge-step-quality.json` |
| `EDGE-STEPS-8-P50` | 80.4068 ms | 20 paired Push-T episodes; quality gate rejected | `evidence/edge-step-quality.json` |
| `EDGE-STEPS-6-P50` | 65.2405 ms | 20 paired Push-T episodes; quality gate rejected | `evidence/edge-step-quality.json` |
| `EDGE-ORIN-P50` | 3.37094 ms | U-Net inference only; no simulator/robot I/O | `evidence/edge-orin.json` |
| `EDGE-ORIN-P95` | 4.41489 ms | U-Net inference only; no simulator/robot I/O | `evidence/edge-orin.json` |
| `EDGE-ORIN-MISS` | 0 count/500 | measured board inference | `evidence/edge-orin.json` |
| `EDGE-ORIN-LOOP-100-P50` | 557.809 ms | physical Orin, synthetic tensors, complete denoising loop; 50 ms timing gate missed | `evidence/edge-orin-denoising-steps.json` |
| `EDGE-ORIN-LOOP-6-P50` | 32.4274 ms | physical Orin timing gate passed; paired Push-T quality gate rejected | `evidence/edge-orin-denoising-steps.json` |
| `EDGE-ORIN-LOOP-6-MISS` | 0 count/20 | physical Orin timing only; paired Push-T quality gate rejected | `evidence/edge-orin-denoising-steps.json` |
| `VLA-MAIN` | 0.133333 ratio | 3 seeds, 45 runs; no zero-shot claim | `evidence/smolvla-quality.json` |
| `VLA-GEN` | 0.180952 ratio | tasks excluded from selection but included in final fine-tuning | `evidence/smolvla-quality.json` |
| `VLA-ORIN-P50` | 1191.74 ms | synthetic input, PyTorch path | `evidence/smolvla-orin.json` |
| `VLA-ORIN-RATE` | 0.836851 Hz | synthetic input, PyTorch path | `evidence/smolvla-orin.json` |
| `VLA-ORIN-MISS` | 500 count/500 | every measured inference missed | `evidence/smolvla-orin.json` |
| `VLA-ORIN-STRENGTHENING` | 1005.55 ms | fixed synthetic input; paired LIBERO task-quality gate reported separately | `evidence/orin-smolvla-strengthening.json` |
| `VLA-DATA-SHARD-AUDIT` | 377 count | D/E/F original audits agree; 432 episodes and 52,970 frames; archive synchronization did not rerun experiments | `evidence/smolvla-dataset-audit.json` |
| `VLA-STAGE-DENOISE-VLM` | 60.5977 percent | diagnostic CUDA-event stage profile; nested inside denoise-step total | `evidence/orin-smolvla-strengthening.json` |
| `VLA-STAGE-PREFIX-EMBED` | 27.0439 percent | diagnostic CUDA-event stage profile | `evidence/orin-smolvla-strengthening.json` |
| `VLA-CHUNK-05-FRESH` | 0.686667 ratio | 300-tick synthetic-input board replay; not task success | `evidence/orin-smolvla-chunk-scheduler-05.json` |
| `VLA-CHUNK-08-FRESH` | 0.866667 ratio | 300-tick synthetic-input board replay; not task success | `evidence/orin-smolvla-chunk-scheduler-08.json` |
| `VLA-CHUNK-08-STALE` | 201 count/300 | higher refill threshold trades fallback for action age | `evidence/orin-smolvla-chunk-scheduler-08.json` |
| `VLA-LIBERO-EAGER-SUCCESS` | 0.2 ratio | 50 official LIBERO Spatial episodes, frozen seed-43 policy | `evidence/smolvla-quality-gate.json` |
| `VLA-LIBERO-COMPILE-SUCCESS` | 0.22 ratio | 50 official LIBERO Spatial episodes, reduce-overhead | `evidence/smolvla-quality-gate.json` |
| `VLA-LIBERO-COMPILE-DELTA` | 0.02 ratio | paired episode delta; acceptance threshold +/-5 percentage points | `evidence/smolvla-quality-gate.json` |
| `VLA-LIBERO-RESIZE-SUCCESS` | 0 ratio | 50 official LIBERO Spatial episodes; frozen seed-43 policy; quality gate rejected | `evidence/smolvla-resize-quality-gate.json` |
| `VLA-LIBERO-RESIZE-DELTA` | -0.2 ratio | paired episode delta; absolute +/-5 percentage-point gate rejected | `evidence/smolvla-resize-quality-gate.json` |
| `VLA-LIBERO-RESIZE384-FT-SUCCESS` | 0.1 ratio | 6000-step rank-32 LoRA recovery candidate; 50 paired official LIBERO Spatial episodes; quality gate rejected | `evidence/smolvla-resize384-finetune-quality-gate.json` |
| `VLA-LIBERO-RESIZE384-FT-DELTA` | -0.1 ratio | paired episode delta; candidate required at least 15% success and was rejected at 10% | `evidence/smolvla-resize384-finetune-quality-gate.json` |
| `VLA-ORIN-INT8-WO-P50` | 1415.67 ms | physical Orin, synthetic input, TorchAO targeted action-expert linear layers; latency gate rejected | `evidence/orin-smolvla-quantization.json` |
| `VLA-ORIN-INT8-DYNAMIC-P50` | 5450.43 ms | physical Orin, synthetic input, TorchAO targeted action-expert linear layers; latency gate rejected | `evidence/orin-smolvla-quantization.json` |
| `VLA-ORIN-INT8-WO-ACTION-MAX` | 0.0245989 abs | correctness gate passed; promotion rejected because P50 was slower than baseline | `evidence/orin-smolvla-quantization.json` |
| `VLA-ORIN-INT8-DYNAMIC-ACTION-MAX` | 0.025344 abs | correctness gate passed; promotion rejected because P50 was slower than baseline | `evidence/orin-smolvla-quantization.json` |
| `VLA-TRACE-05-DELTA` | -0.16 ratio | paired LIBERO simulation with Orin latency envelope; not physical robot | `evidence/smolvla-orin-latency-envelope.json` |
| `VLA-TRACE-08-DELTA` | -0.36 ratio | paired LIBERO simulation with Orin latency envelope; not physical robot | `evidence/smolvla-orin-latency-envelope.json` |
| `VLA-TRACE-05-FALLBACK` | 3618 count | 25 paired LIBERO episodes; latency envelope injection | `evidence/smolvla-orin-latency-envelope.json` |
| `VLA-TRACE-08-FALLBACK` | 4028 count | 25 paired LIBERO episodes; latency envelope injection | `evidence/smolvla-orin-latency-envelope.json` |
| `VLA-REPLAY-FRESH` | 0.0266667 ratio | offline timing replay, not task execution | `evidence/control-replay.json` |

## 8. Reproduction and audit

The minimum offline path is:

```bash
cd .
python3 reproduction/package/reproduce.py
```

It verifies frozen aggregation, timing replay, the reconciled original dataset audit, claim checks, release hashes and the strict completed-task audit. It does **not** rerun model inference and does not require cloud credentials or active AutoDL/Orin resources.

The public sanitized package can be checked independently:

```bash
cd .
python3 scripts/verify_repository.py .
```

Current audit result: **36/36** tasks publication-ready, with **0** partial tasks and **0** invalid resource evidence references.

## 9. Interview-safe conclusion

The defensible conclusion is not “a VLA runs at 30 Hz on Orin.” It is: **a frozen robot-learning workload was evaluated under explicit quality gates, accelerated on a fixed RTX 4090, deployed to a physical Orin target, independently rerun, and stopped with a quantified negative real-time decision when the full control requirement was not met.**

That distinction is the central systems-engineering result of this portfolio.
