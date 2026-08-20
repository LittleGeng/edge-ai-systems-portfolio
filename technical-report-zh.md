# AI 系统与具身智能技术报告

**报告 ID：** `job-portfolio-v1`  
**状态：** 冻结证据综合报告  
**实验执行截止日期：** 2026-08-07  
**材料刷新时间：** 2026-08-09T11:06:59+08:00  
**归档校验：** `autodl-release-20260808`，完成于 2026-08-09T09:07:35+08:00（`verify-only`；8 月 8 日至 9 日仅进行同步与校验）  
**发布审计：** 36/36 个已完成任务可发布  
**证据原则：** 所有实测值均关联原始产物和 SHA256 清单；没有证据支持的数值保留为明确限制，不作推测。

## 1. 执行摘要

本报告面向工业界 AI 系统研发和具身智能岗位，评估两条互补的系统工程路线：

1. **EdgeDiffusion：** 在官方 Push-T 仿真中评测冻结的 Diffusion Policy checkpoint，在固定 RTX 4090 上优化，并在 Jetson AGX Orin 上部署板端原生 FP16 TensorRT U-Net 引擎。
2. **SmolVLA Edge：** 在官方 LIBERO Spatial 任务上评测冻结 LoRA 策略，使用合成观测在物理 Orin 上测量，并以配对 `torch.compile(reduce-overhead)` 任务质量门禁筛选优化候选。

最强的正向系统结果是：固定节点上 100 步去噪 P50 从 **566.71 ms** 降至 **235.80 ms**，加速 **2.40x**；板端原生 Orin U-Net 达到 **3.37/4.41 ms P50/P95**，在 50 ms 推理 deadline 下超期 **0/500** 次。但对应的 Orin scheduler + U-Net 完整循环在 100 步时显著更慢，为 **557.81/568.67 ms**；降至 6 步虽然通过本地 50 ms 时序门禁，却因配对 Push-T 质量筛选的高质量完成率降至 0% 而被拒绝。

最重要的负向结果是：当前 SmolVLA PyTorch 路径仅达到 **0.837 Hz**，在 30 Hz deadline 下超期 **500/500** 次。`torch.compile` 在固定合成输入补强实验中降低了首动作 P50，但既未实现 30 Hz 控制，也未证明任务质量收益。直接下采样至 256x256 失败；6000 步 384x384 恢复候选的配对成功率仍仅为 **10.0%**；定向 TorchAO weight-only/dynamic INT8 又将首动作 P50 从 eager 的 **1202.12 ms** 恶化至 **1415.67/5450.43 ms**。

## 2. 研究问题与声明边界

实验回答四个有明确边界的问题：

| 问题 | 证据来源 | 声明边界 |
|---|---|---|
| 冻结策略能否在官方仿真器中完成任务？ | Push-T 与 LIBERO 质量实验 | 仿真质量不等于物理机器人成功率。 |
| 后端能否降低同一工作负载的时延？ | 固定 RTX 4090 eager/compile/TensorRT 对比 | 数值仅适用于固定模型、形状、步数和节点。 |
| 目标板引擎能否满足局部推理 deadline？ | Orin 正确性与时序实验 | 明确限定为 U-Net 或策略推理，不含完整机器人 I/O。 |
| 优化后是否保持任务质量？ | 50 个 episode 的配对 LIBERO 门禁 | 通过筛选门禁不等于证明改进或统计优越性。 |

任何结果都不外推为对所有 VLA 模型、所有 GPU 或物理机器人的一般性结论。

## 3. 冻结输入与实验协议

- Diffusion Policy 仓库 revision：`5ba07ac6661db573af695b419a7947ecb704690f`。
- Diffusion Policy checkpoint SHA256：`f804e16575e261fa0b7e981da3f67741fc8517817734320d550e43a4182bf876`。
- 目标板：Jetson AGX Orin 64GB，L4T 36.5.0 证据路径。
- Edge 质量：60 个官方 checkpoint Push-T episode，使用互不重复的环境种子。
- EdgeDiffusion 步数质量：每个去噪步数设置（100/20/10/8/6）使用 20 个配对 Push-T episode，并使用同节点参考窗口。
- SmolVLA 质量：官方 LIBERO Spatial 配对协议，10 个任务 x 5 个 episode；eager 与 compile 使用相同 episode ID、冻结 seed-43 策略和 50 个动作步。
- SmolVLA 缩放质量：沿用相同 50 个配对 episode ID，仅将 `resize_imgs_with_padding` 从 512x512 改为 256x256。
- SmolVLA 缩放恢复：384x384、rank 32、学习率 `5e-4`、seed 43、6000 步 LoRA 候选；使用相同 50 个配对 episode ID，预先声明最低成功率 15% 的停止门禁。
- SmolVLA 定向 INT8：TorchAO v0.13.0，固定 revision `e318546f9be8c6dd1340157cd14e5cbc6ffa1f65`；物理 Orin 上每个变体预热 5 次、测量 20 次合成输入；排除带 LoRA adapter 的 Q/V projection。
- SmolVLA 调度质量：在阈值 0.5 和 0.8 下进行配对 LIBERO 仿真，并注入物理 Orin P50/P95 时延包络；不是物理机器人实验，也不是原始时间戳回放。
- SmolVLA 数据集审计：8 月 4 日的三份原始审计均为 432 个选定 episode、52,970 帧和全部 377 个 parquet 分片哈希；所有已记录行级完整性门禁通过。
- Orin 运行时测量区分首动作与稳定动作 chunk，并保留原始时序 CSV 和遥测。
- 实验在 2026-08-07 前结束。8 月 8 日至 9 日仅进行远端到本地同步、清点和哈希校验，没有产生新的实验观测。
- 所有本地发布产物由 `SHA256SUMS` 校验；公开项目经过脱敏，不含凭据、私有端点、计费数据和面试问答。

### 3.1 环境与产物身份

| 工作负载 | 硬件/运行时 | 冻结身份 |
|---|---|---|
| EdgeDiffusion 云端基准 | NVIDIA GeForce RTX 4090；CUDA 12.4 | PyTorch 2.5.1+cu124；Torch-TensorRT 2.5.0+cu124；TensorRT 10.3.0；Diffusers 0.11.1 |
| EdgeDiffusion 目标板 | Jetson AGX Orin 64GB；L4T 36.5.0 | TensorRT 10.11.0.33；PyTorch 2.8.0a0+5228986c39.nv25.06；engine `770a439e18d2...` |
| SmolVLA 目标板 | Jetson AGX Orin 64GB；合成观测 | LeRobot `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`；base model `c83c3163b8ca9b7e67c509fffd9121e66cb96205`；policy `435185f15d31...` |
| Push-T 质量 | 官方仿真协议 `edge-evaluation-protocol-v1` | repository `5ba07ac6661db573af695b419a7947ecb704690f`；checkpoint `f804e16575e2...` |

## 4. EdgeDiffusion 实验结果

### 4.1 任务质量

官方 checkpoint 的高质量完成率为 **95.0%**（57/60），平均任务得分 **0.959**。独立同种子 20-episode 复跑达到 **90.0%**，与其 20-episode 基线完全一致，并通过预先声明的 2 个百分点复跑容差。该容差是配对窗口检查，不表示 20-episode 成功率与另一次 60-episode 汇总值相等。

### 4.2 固定 RTX 4090 优化

在同一固定 RTX 4090 上测量相同模型、输入形状和 100 步去噪工作负载。预热 2 次后测量 30 次循环；每 0.1 秒采样一次功耗。计时范围从条件张量构造完成后开始。

| 后端 | P50 / P95 | 按 P50 计算吞吐 | 增量峰值分配 | GPU 平均采样功耗 | 最大去噪误差 |
|---|---:|---:|---:|---:|---:|
| Eager | 566.71 / 570.36 ms | 1.765 /s | 20.37 MiB | 96.67 W | 0.000000 |
| `torch.compile` | 347.34 / 356.44 ms | 2.879 /s | 20.37 MiB | 109.62 W | 0.000066 |
| TensorRT | 235.80 / 237.93 ms | 4.241 /s | 0.01 MiB | 131.41 W | 0.000100 |

按同节点 P50 比值，TensorRT 加速为 **2.40x**。数值正确性、Push-T 质量门禁和时延测量相互独立，防止速度更快但行为已经改变的后端被错误晋级。

### 4.3 Orin 板端原生引擎

FP16 引擎在物理 Orin 上构建，并通过 **20/20** 个正确性用例，最大绝对误差 **0.011283**。U-Net wall-time 基准为 **3.37 ms P50**、**4.41 ms P95**，吞吐 **280.87 actions/s**，50 ms deadline 超期 **0/500** 次。

这是**单次 U-Net 调用**的时延，不含 scheduler 重复、观测处理、仿真器、相机采集、执行器 I/O、网络与安全逻辑。由于现有遥测不能证明存在不重叠的整机总功率轨，报告有意不计算总能耗/action。

### 4.4 去噪步数质量/时延扫描

冻结 checkpoint 在五种去噪步数下分别使用 20 个配对 Push-T episode。100 步是质量参考；每个较低步数与同节点参考种子窗口比较，预先声明的平均得分差和完成率差上限均为 0.05。

| 去噪步数 | 控制器 P50 | 控制器 P95 | 平均得分 | 高质量完成率 | 质量门禁 |
|---:|---:|---:|---:|---:|---|
| 100 | 1076.5 ms | 1120.3 ms | 0.922 | 90% | 通过 |
| 20 | 217.6 ms | 236.2 ms | 0.119 | 0% | 拒绝 |
| 10 | 88.4 ms | 96.8 ms | 0.093 | 0% | 拒绝 |
| 8 | 80.4 ms | 90.5 ms | 0.067 | 0% | 拒绝 |
| 6 | 65.2 ms | 81.4 ms | 0.104 | 0% | 拒绝 |

只有 100 步通过配对质量门禁。20/10/8/6 步分别将 P50 降至 **217.6/88.4/80.4/65.2 ms**，但各自的 20-episode 窗口高质量完成率均为 **0%**。这些是有效的负向系统结果，不能据此声明较低步数可部署。

### 4.5 物理 Orin 上的完整去噪循环

将板端原生 TensorRT U-Net 放入冻结 DDPM scheduler 中，每个步数设置预热 3 次后测量 20 次：

| 去噪步数 | Orin 循环 P50 | Orin 循环 P95 | 50 ms 超期 | 时序状态 | 跨证据部署决策 |
|---:|---:|---:|---:|---|---|
| 100 | 557.81 ms | 568.67 ms | 20/20 | 超期 | 质量参考；时序超期 |
| 20 | 110.80 ms | 121.96 ms | 20/20 | 超期 | 拒绝：配对 Push-T 质量门禁失败 |
| 10 | 54.88 ms | 64.15 ms | 20/20 | 超期 | 拒绝：配对 Push-T 质量门禁失败 |
| 8 | 43.68 ms | 50.01 ms | 1/20 | 超期 | 拒绝：配对 Push-T 质量门禁失败 |
| 6 | 32.43 ms | 36.28 ms | 0/20 | 通过 | 拒绝：配对 Push-T 质量门禁失败 |

6 步循环达到 **32.43/36.28 ms**，50 ms 超期为零，但独立的配对 Push-T 筛选报告高质量完成率为 **0%**。相反，100 步质量参考在 Orin 上每次都错过时序 deadline。这是核心的时延与质量冲突：没有任何实测步数同时通过两个门禁。目标板循环使用合成固定形状张量，不含观测、仿真器和机器人 I/O。

## 5. SmolVLA Edge 实验结果

### 5.1 质量与数据集范围

三个 LoRA 微调种子在 45 次主条件运行中获得 **13.3%** 成功率，在 105 次次要条件运行中获得 **18.1%** 成功率。次要任务未参与模型选择，但存在于最终微调数据中，因此这**不是**零样本迁移声明。

| 条件 | 唯一 episode / 运行次数 | 各训练种子成功率 | 两阶段 bootstrap 95% CI |
|---|---:|---|---:|
| 主选择任务 | 15 / 45 | 42: 13.3%, 43: 6.7%, 44: 20.0% | [2.2%, 26.7%] |
| 次要条件 | 35 / 105 | 42: 8.6%, 43: 20.0%, 44: 25.7% | [7.6%, 29.5%] |

VLA-01 原始审计在 8 月 4 日采集，随后同步到本地归档。D/E/F 三份独立记录均为 **432 个选定 episode**、**52,970 帧**和 **377/377 个 parquet 分片哈希**。文件存在性、frame index、task index、时间戳与元数据计数门禁全部通过，没有文件缺失或失败行。归档在 2026-08-10T08:19:32+08:00 以 `verify-only` 模式校验；该同步没有重跑实验。审计覆盖 parquet 行和分片身份，不覆盖解码后视频像素或任务成功率。

### 5.2 Orin 运行时基线与补强

冻结 PyTorch 路径在 Orin 上使用合成观测，首动作 P50 为 **1190.17 ms**，稳定 chunk P50 为 **1191.74 ms**，控制率 **0.837 Hz**，30 Hz deadline 超期 **500/500** 次。

固定输入补强对比中，eager 首动作 P50 为 **1287.54 ms**，`torch.compile(reduce-overhead)` 为 **1005.55 ms**，降低 **21.9%**。峰值 RAM 从 **7813 MB** 增至 **8912 MB**。两个变体在所有补强迭代中仍均错过 30 Hz deadline。

完整固定输入扫描如下：

| 变体 | 首动作 P50 | 首动作 P95 | 频率 | 30 Hz 超期 | 峰值 RAM | 相对 eager 最大动作差 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline-fixed` | 1287.54 ms | 1322.35 ms | 0.773 Hz | 20/20 | 7813 MB | 0.0000 | 参考 | 
| `amp-fixed` | 1242.39 ms | 1274.36 ms | 0.804 Hz | 20/20 | 7785 MB | 0.0225 | 小幅收益；仍不充分 | 
| `resize256-fixed` | 996.67 ms | 1033.76 ms | 0.998 Hz | 20/20 | 7694 MB | 2.0072 | 配对任务质量门禁拒绝 | 
| `compile-reduce-overhead-fixed` | 1005.55 ms | 1025.40 ms | 0.996 Hz | 20/20 | 8912 MB | 0.0767 | 筛选点估计通过；CI 不确定；实时性失败 | 

resize-256 变体速度更快，但代表动作发生显著变化，最大绝对差为 **2.0072**；在完成任务质量实验前不能视为优化成功。

诊断性分阶段剖析使用 20 次固定输入调用和 10 个去噪步骤，并精确保留基线代表动作。CUDA event 归因确定了下一步优化目标：

| 阶段 | 单次推理平均耗时 | 占完整推理比例 | 嵌套关系 |
|---|---:|---:|---|
| Prefix embedding | 332.45 ms | 27.04% | 顶层 |
| Prefix KV forward | 96.90 ms | 7.88% | 顶层 |
| 去噪步骤总计 | 785.77 ms | 63.92% | 顶层 |
| Denoise VLM forward | 744.92 ms | 60.60% | 嵌套于去噪步骤总计 |
| 未归因 | 14.16 ms | - | 残差 |

Denoise VLM forward 单独占 **60.60%**，prefix embedding 又占 **27.04%**，因此下一步应对这些阶段做 kernel/export 级分析，而不是继续猜测宽泛组件。该剖析是诊断性的：使用未逐阶段同步的 CUDA event，嵌套去噪行不能相加。

### 5.3 物理 Orin 上的定向 TorchAO INT8

后训练量化针对 SmolVLA action expert 中 80 个 K/O/MLP 线性层，并排除带 LoRA 的 Q/V projection。两个候选均通过预先声明的动作差门禁，但都未通过首动作时延至少降低 5% 的门禁。

| 变体 | 首动作 P50/P95 | 稳定态 P50/P95 | 频率 | 峰值 RAM | 动作最大/平均绝对差 | 决策 |
|---|---:|---:|---:|---:|---:|---|
| Eager 基线 | 1202.12/1245.03 ms | 1203.49/1236.71 ms | 0.829 Hz | 8250 MB | 0/0 | 参考 |
| INT8 weight-only | 1415.67/1457.57 ms | 1417.84/1445.21 ms | 0.705 Hz | 8242 MB | 0.02460/0.00524 | 拒绝 |
| INT8 dynamic | 5450.43/5498.47 ms | 5427.35/5481.92 ms | 0.184 Hz | 8265 MB | 0.02534/0.00737 | 拒绝 |

按首动作 P50，weight-only 比 eager **慢 17.8%**，dynamic INT8 **慢 353.4%**。该结论仅适用于此模型分区、TorchAO revision、NVIDIA 25.06 iGPU 容器与合成输入。它不说明 INT8 在 Orin 上普遍较慢，只说明这些具体后训练路径增加的运行时开销超过了其节省。

### 5.4 配对任务质量门禁

在相同 50 个 episode ID 和相同冻结策略下：

- Eager：**10/50 = 20.0%**。
- `torch.compile(reduce-overhead)`：**11/50 = 22.0%**。
- 配对差值：**+2.0%**。
- Bootstrap 95% CI：**[-10.0%, +14.0%]**。
- 预先声明的点估计门禁：绝对差值不超过 5 个百分点，**通过**。

点估计通过预设筛选规则，但宽置信区间越过 +/-5 个百分点区域。这**不能**建立统计非劣效性或优越性，且绝对成功率仍然较低。

### 5.5 配对视觉缩放质量门禁

保持冻结 seed-43 策略、LeRobot revision 和 episode ID 不变，仅将 `resize_imgs_with_padding` 从 512x512 改为 256x256，成功率为 **0/50 = 0.0%**；eager 512x512 为 **10/50 = 20.0%**。配对差值 **-20.0%**，bootstrap 95% CI 为 **[-32.0%, -10.0%]**。预设绝对 5 个百分点门禁**失败**；eager 独有的 10 次成功全部丢失，resize256 独有成功为零。

这是有价值的负向系统结果。固定输入扫描表明低分辨率路径可以更快，但冻结 checkpoint 不具备分辨率鲁棒性。拒绝直接部署 256x256；低分辨率路线需要重训练或蒸馏，并再次接受同一配对任务门禁。

### 5.6 微调 384px 恢复停止门禁

恢复候选使用 rank 32、学习率 `5e-4`、seed 43，在 384x384 下训练 6000 个 LoRA 步骤。在相同 50 个配对官方 LIBERO Spatial episode ID 上达到 **5/50 = 10.0%**；eager 512x512 为 **10/50 = 20.0%**。配对差值 **-10.0%**，bootstrap 95% CI 为 **[-24.0%, +2.0%]**。

预先声明的恢复规则要求成功率至少 **15.0%**。候选仅达到 **10.0%**，因此停止原计划的三种子、20k 步扩展。这既是质量结果，也是资源治理结果：失败的配对门禁避免了更大的算力投入。

### 5.7 物理 Orin 上的动作 chunk 调度

在物理 Orin 上以合成观测回放冻结的 50-action chunk scheduler，共 300 个 tick：

| 补充阈值 | Chunk P50/P95 | 新鲜动作 tick | Fallback/underflow tick | 过期 tick |
|---:|---:|---:|---:|---:|
| 0.5 | 1302.48/1354.78 ms | 68.7% | 94 | 100 |
| 0.8 | 1325.58/1362.75 ms | 86.7% | 40 | 201 |

提高补充阈值将新鲜动作可用率从 **68.7%** 提升至 **86.7%**，fallback tick 从 **94** 降至 **40**，但过期 tick 从 **100** 增至 **201**。缓冲可以在 underflow 与动作年龄之间权衡，但不能生成新的策略输出，也不能证明闭环质量。这是目标板合成输入回放，不是物理机器人任务成功率。

### 5.8 调度与安全解释

离线 30 Hz scheduler 回放产生 **8/300** 个新鲜动作 tick 和 **284** 个安全 fallback tick。该回放使用实测 Orin 时序 trace，不是新的仿真实验，也不是物理机器人任务成功率。

### 5.9 时延包络下的闭环质量

为检验板端时延下异步调度能否保持仿真任务质量，将物理 Orin P50/P95 端点注入配对 LIBERO 闭环运行。两个阈值各使用 25 个配对 episode。

| 成功阈值 | 同步成功率 | 异步成功率 | 异步减同步 | Fallback / 队列空 tick | 门禁 |
|---:|---:|---:|---:|---:|---|
| 0.5 | 24% | 8% | -16% | 3618 | 拒绝 |
| 0.8 | 40% | 4% | -36% | 4028 | 拒绝 |

两个设置都违反预设的 +/-5 个百分点质量门禁。结合物理 Orin 回放可见，更高补充阈值减少 underflow，却增加动作年龄；注入时延的 LIBERO 运行仍损失任务成功率。仿真实验使用 P50/P95 汇总端点而非逐 chunk 原始时间戳，因此是时延包络注入，而非完整原始 trace 回放。

部署决策是：拒绝对实测 eager、compile 和定向后训练 INT8 路径作出直接 30 Hz 控制声明。分阶段剖析已将主要耗时定位到 denoise VLM forward 和 prefix embedding；下一项 VLA 门禁应先对这些路径进行 kernel/export 级剖析，再选择 TensorRT/ONNX 或量化感知重训练。对于 EdgeDiffusion，步数扫描保留 100 步作为质量参考；任何较低步数优化都必须采用蒸馏或重训练等质量保持方法。

## 6. 跨项目工程决策

1. **质量与速度分离。** 不能仅凭时延表晋级后端；必须同时通过数值和任务级门禁。
2. **局部推理与闭环控制分离。** U-Net 和策略时延不包含全部传感器、执行器与安全循环开销。
3. **保留负向结果。** SmolVLA 未达到 30 Hz；这是有效部署结论，不是待填占位符。
4. **不虚构能耗数值。** 缺少整机功率轨证据时保留 `null`，不估算 joules/action。
5. **使用独立复跑。** REL-03 重建 Orin 引擎并重复 Push-T 种子窗口；P50/P95 与任务成功率容差均通过。
6. **把输入分辨率视为质量契约。** 尽管固定输入下速度更快，256x256 路径仍被配对 LIBERO 门禁拒绝；没有重训练或蒸馏时，纯性能缩放不可部署。
7. **在冻结门禁处停止恢复实验。** 6000 步 384px LoRA 候选未达到 15% 下限，因此不执行三种子、20k 步扩展。
8. **不把更小权重等同于更低时延。** 定向 INT8 降低估算权重载荷和 CUDA 分配，但两个实测变体均使该 Orin 软件栈上的端到端动作时延恶化。
9. **不混淆同步与实验。** 8 月 8 日至 9 日仅复制、清点和哈希校验既有证据；每项声明保留原始采集时间。

## 7. 完整 Claim 索引

| Claim | 数值 | 适用范围 | 证据 |
|---|---:|---|---|
| `EDGE-SUCCESS` | 0.95 ratio | 官方 checkpoint，60 个仿真 episode | `evidence/edge-quality.json` |
| `EDGE-SUCCESS-RERUN` | 0.9 ratio | 官方 checkpoint，同种子 20 个 episode；独立复跑 | `evidence/independent-rerun.json` |
| `EDGE-4090-EAGER` | 566.709 ms | 固定 RTX 4090 节点 | `evidence/edge-fixed-node.json` |
| `EDGE-4090-TRT` | 235.798 ms | 固定 RTX 4090 节点 | `evidence/edge-fixed-node.json` |
| `EDGE-4090-SPEEDUP` | 2.40337 x | 同一节点上的 P50 比值 | `evidence/edge-fixed-node.json` |
| `EDGE-STEPS-100-P50` | 1076.48 ms | 20 个配对 Push-T episode；质量门禁通过 | `evidence/edge-step-quality.json` |
| `EDGE-STEPS-20-P50` | 217.602 ms | 20 个配对 Push-T episode；质量门禁拒绝 | `evidence/edge-step-quality.json` |
| `EDGE-STEPS-10-P50` | 88.3921 ms | 20 个配对 Push-T episode；质量门禁拒绝 | `evidence/edge-step-quality.json` |
| `EDGE-STEPS-8-P50` | 80.4068 ms | 20 个配对 Push-T episode；质量门禁拒绝 | `evidence/edge-step-quality.json` |
| `EDGE-STEPS-6-P50` | 65.2405 ms | 20 个配对 Push-T episode；质量门禁拒绝 | `evidence/edge-step-quality.json` |
| `EDGE-ORIN-P50` | 3.37094 ms | 仅 U-Net 推理；不含仿真器或机器人 I/O | `evidence/edge-orin.json` |
| `EDGE-ORIN-P95` | 4.41489 ms | 仅 U-Net 推理；不含仿真器或机器人 I/O | `evidence/edge-orin.json` |
| `EDGE-ORIN-MISS` | 0 count/500 | 目标板实测推理 | `evidence/edge-orin.json` |
| `EDGE-ORIN-LOOP-100-P50` | 557.809 ms | 物理 Orin、合成张量、完整去噪循环；未通过 50 ms 时序门禁 | `evidence/edge-orin-denoising-steps.json` |
| `EDGE-ORIN-LOOP-6-P50` | 32.4274 ms | 物理 Orin 时序门禁通过；配对 Push-T 质量门禁拒绝 | `evidence/edge-orin-denoising-steps.json` |
| `EDGE-ORIN-LOOP-6-MISS` | 0 count/20 | 仅物理 Orin 时序；配对 Push-T 质量门禁拒绝 | `evidence/edge-orin-denoising-steps.json` |
| `VLA-MAIN` | 0.133333 ratio | 3 个种子、45 次运行；不作零样本声明 | `evidence/smolvla-quality.json` |
| `VLA-GEN` | 0.180952 ratio | 任务未参与模型选择，但包含在最终微调数据中 | `evidence/smolvla-quality.json` |
| `VLA-ORIN-P50` | 1191.74 ms | 合成输入，PyTorch 路径 | `evidence/smolvla-orin.json` |
| `VLA-ORIN-RATE` | 0.836851 Hz | 合成输入，PyTorch 路径 | `evidence/smolvla-orin.json` |
| `VLA-ORIN-MISS` | 500 count/500 | 每次实测推理均超期 | `evidence/smolvla-orin.json` |
| `VLA-ORIN-STRENGTHENING` | 1005.55 ms | 固定合成输入；配对 LIBERO 任务质量门禁单独报告 | `evidence/orin-smolvla-strengthening.json` |
| `VLA-DATA-SHARD-AUDIT` | 377 count | D/E/F 原始审计一致：432 个 episode、52,970 帧；归档同步未重跑实验 | `evidence/smolvla-dataset-audit.json` |
| `VLA-STAGE-DENOISE-VLM` | 60.5977 percent | 诊断性 CUDA event 分阶段剖析；嵌套于去噪步骤总耗时 | `evidence/orin-smolvla-strengthening.json` |
| `VLA-STAGE-PREFIX-EMBED` | 27.0439 percent | 诊断性 CUDA event 分阶段剖析 | `evidence/orin-smolvla-strengthening.json` |
| `VLA-CHUNK-05-FRESH` | 0.686667 ratio | 目标板 300 tick 合成输入回放；不是任务成功率 | `evidence/orin-smolvla-chunk-scheduler-05.json` |
| `VLA-CHUNK-08-FRESH` | 0.866667 ratio | 目标板 300 tick 合成输入回放；不是任务成功率 | `evidence/orin-smolvla-chunk-scheduler-08.json` |
| `VLA-CHUNK-08-STALE` | 201 count/300 | 更高补充阈值以动作时效性换取更少 fallback | `evidence/orin-smolvla-chunk-scheduler-08.json` |
| `VLA-LIBERO-EAGER-SUCCESS` | 0.2 ratio | 50 个官方 LIBERO Spatial episode，冻结 seed-43 策略 | `evidence/smolvla-quality-gate.json` |
| `VLA-LIBERO-COMPILE-SUCCESS` | 0.22 ratio | 50 个官方 LIBERO Spatial episode，reduce-overhead | `evidence/smolvla-quality-gate.json` |
| `VLA-LIBERO-COMPILE-DELTA` | 0.02 ratio | 配对 episode 差值；验收阈值为 +/-5 个百分点 | `evidence/smolvla-quality-gate.json` |
| `VLA-LIBERO-RESIZE-SUCCESS` | 0 ratio | 50 个官方 LIBERO Spatial episode；冻结 seed-43 策略；质量门禁拒绝 | `evidence/smolvla-resize-quality-gate.json` |
| `VLA-LIBERO-RESIZE-DELTA` | -0.2 ratio | 配对 episode 差值；绝对 +/-5 个百分点门禁拒绝 | `evidence/smolvla-resize-quality-gate.json` |
| `VLA-LIBERO-RESIZE384-FT-SUCCESS` | 0.1 ratio | 6000 步 rank-32 LoRA 恢复候选；50 个配对官方 LIBERO Spatial episode；质量门禁拒绝 | `evidence/smolvla-resize384-finetune-quality-gate.json` |
| `VLA-LIBERO-RESIZE384-FT-DELTA` | -0.1 ratio | 配对 episode 差值；候选要求至少 15% 成功率，实测 10% 后拒绝 | `evidence/smolvla-resize384-finetune-quality-gate.json` |
| `VLA-ORIN-INT8-WO-P50` | 1415.67 ms | 物理 Orin、合成输入、TorchAO 定向量化 action expert 线性层；时延门禁拒绝 | `evidence/orin-smolvla-quantization.json` |
| `VLA-ORIN-INT8-DYNAMIC-P50` | 5450.43 ms | 物理 Orin、合成输入、TorchAO 定向量化 action expert 线性层；时延门禁拒绝 | `evidence/orin-smolvla-quantization.json` |
| `VLA-ORIN-INT8-WO-ACTION-MAX` | 0.0245989 abs | 正确性门禁通过；因 P50 慢于基线而拒绝晋级 | `evidence/orin-smolvla-quantization.json` |
| `VLA-ORIN-INT8-DYNAMIC-ACTION-MAX` | 0.025344 abs | 正确性门禁通过；因 P50 慢于基线而拒绝晋级 | `evidence/orin-smolvla-quantization.json` |
| `VLA-TRACE-05-DELTA` | -0.16 ratio | 注入 Orin 时延包络的配对 LIBERO 仿真；不是物理机器人 | `evidence/smolvla-orin-latency-envelope.json` |
| `VLA-TRACE-08-DELTA` | -0.36 ratio | 注入 Orin 时延包络的配对 LIBERO 仿真；不是物理机器人 | `evidence/smolvla-orin-latency-envelope.json` |
| `VLA-TRACE-05-FALLBACK` | 3618 count | 25 个配对 LIBERO episode；时延包络注入 | `evidence/smolvla-orin-latency-envelope.json` |
| `VLA-TRACE-08-FALLBACK` | 4028 count | 25 个配对 LIBERO episode；时延包络注入 | `evidence/smolvla-orin-latency-envelope.json` |
| `VLA-REPLAY-FRESH` | 0.0266667 ratio | 离线时序回放，不是任务执行 | `evidence/control-replay.json` |

## 8. 复现与审计

最小离线复现路径：

```bash
cd .
python3 reproduction/package/reproduce.py
```

它会校验冻结聚合、时序回放、已对账原始数据集审计、claim 检查、发布哈希和严格的已完成任务审计。它**不会**重跑模型推理，也不需要云端凭据或仍在运行的 AutoDL/Orin 资源。

公开脱敏包可独立校验：

```bash
cd .
python3 scripts/verify_repository.py .
```

当前审计结果：**36/36** 个任务可发布，部分完成任务 **0** 个，无效资源证据引用 **0** 个。

## 9. 面试表述结论

可辩护的结论不是“VLA 在 Orin 上以 30 Hz 运行”，而是：**在明确质量门禁下评测冻结的机器人学习工作负载，在固定 RTX 4090 上完成加速，部署到物理 Orin 目标板并独立复跑；当完整控制要求未满足时，以量化负向结果终止错误的实时部署声明。**

这种边界意识是本项目组合最核心的系统工程结果。
