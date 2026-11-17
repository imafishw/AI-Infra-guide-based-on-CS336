# AI 基础设施：训练资源估算 / AI Infrastructure: Training Resource Estimates

用粗略估算（back-of-the-envelope calculation / napkin math）回答两个问题：训练需要多久，以及显存能容纳多大的模型。

Use rough estimates to answer two questions: how long training takes, and how large a model fits in GPU memory.

## 1. 基础概念 / Basic concepts

| 概念 / Term | 中文解释 | English explanation |
|---|---|---|
| Parameter（参数） | 训练时调整的数值，例如矩阵中的权重。70B = 700 亿参数。 | A learned value, such as a matrix weight. 70B means 70 billion parameters. |
| Token（词元） | 文本处理单位，可以是词、词的一部分或标点，不一定等于一个字。 | A unit of text, such as a word, word fragment, or punctuation. |
| Training tokens（训练词元数） | 训练累计处理的 token 数，不一定都是独立、无重复的数据。 | Total tokens processed during training, not necessarily unique tokens. |
| FLOPs | 浮点运算总次数，表示计算工作量。 | Total floating-point operations: the amount of computation. |
| FLOP/s | 每秒浮点运算次数，表示计算速度。 | Floating-point operations per second: compute throughput. |
| GPU memory（显存） | 保存权重、梯度、优化器状态及中间结果的空间。 | Storage for weights, gradients, optimizer states, and intermediate results. |
| MFU | 模型有效计算吞吐占理论峰值的比例，不等于监控面板上的 GPU 使用率。 | Model FLOPs Utilization: useful model throughput divided by theoretical peak throughput. |

科学计数法 / Scientific notation: `70e9 = 70 × 10⁹`，`15e12 = 15 × 10¹²`。B = billion（十亿），T = trillion（万亿）。

## 2. 用 1024 张 H100 训练 70B 模型、处理 15T tokens，需要多久？

How long does it take to train a 70B model on 15T tokens using 1024 H100 GPUs?

### 总计算量 / Total computation

稠密 Transformer 训练的常用近似公式为 **C ≈ 6ND**：N 是参数数量，D 是训练 token 总数。前向传播约需 2ND，反向传播约需 4ND。

For dense Transformer training, a common approximation is **C ≈ 6ND**, where N is the parameter count and D is the training token count. The forward pass costs roughly 2ND and the backward pass roughly 4ND.

- 前向传播 / Forward pass：根据输入生成预测 / Produce predictions from inputs.
- 反向传播 / Backward pass：计算用于更新参数的梯度 / Compute gradients for parameter updates.

```python
total_flops = 6 * 70e9 * 15e12  # 6.3e24 FLOPs
```

该公式是近似值，没有完整计算注意力等额外开销；长序列下偏差可能更明显。

This approximation does not fully capture costs such as attention, which can matter more for long sequences.

### 有效计算速度 / Effective throughput

```python
h100_flop_per_sec = 1979e12 / 2  # Dense FP16/BF16 throughput, H100 SXM
mfu = 0.5
num_gpus = 1024
seconds_per_day = 60 * 60 * 24
flops_per_day = h100_flop_per_sec * mfu * num_gpus * seconds_per_day
days = total_flops / flops_per_day  # Approximately 144 days
```

- **1979e12 / 2**：这里采用 H100 SXM 的 FP16/BF16 Tensor Core 算力口径。1979 TFLOP/s 是特定稀疏条件下的峰值，除以 2 得到约 989.5 TFLOP/s 的稠密峰值。不同 H100 型号的规格不同。
  **1979e12 / 2** converts the sparse FP16/BF16 peak to approximately 989.5 TFLOP/s of dense throughput for the H100 SXM.
- **MFU = 0.5**：假设有效模型计算达到理论峰值的 50%，实际效率受通信、内存访问和等待等影响。
  **MFU = 0.5** assumes useful model computation achieves 50% of peak throughput; communication, memory access, and stalls affect efficiency.
- **86400**：一天的秒数，把每秒吞吐转换为每天吞吐。
  **86400** converts throughput per second into throughput per day.

每天有效计算量约为 **4.38 × 10²² FLOPs**，所以训练时间约为 **144 天**。

Effective daily computation is approximately **4.38 × 10²² FLOPs**, giving a training time of approximately **144 days**.

这是假设集群持续维持该效率的估计；停机、评估、保存检查点等可能增加实际日历时间。

This assumes sustained cluster efficiency; downtime, evaluation, and checkpointing can increase elapsed time.

规格参考 / Hardware reference: [NVIDIA H100 GPU Whitepaper](https://resources.nvidia.com/en-us-hopper-architecture).

## 3. 用 AdamW 在 8 张 80 GB H100 上，最多能训练多大的模型？

What is the largest model that can be trained with AdamW on eight 80 GB H100 GPUs?

AdamW 是使用梯度的一阶矩、二阶矩并采用解耦权重衰减的优化器。训练需要保存参数以及更新参数所需的状态。

AdamW uses gradient moments and decoupled weight decay. Training stores parameters together with the states needed to update them.

| 存储项 / Item | 作用 / Purpose | 本例每参数占用 / Bytes per parameter |
|---|---|---:|
| Parameters（参数） | 当前模型权重 / Current model weights | 2 |
| Gradients（梯度） | 损失对参数的变化率 / Rate of change of loss with respect to parameters | 2 |
| First moment（一阶矩） | 梯度的移动平均 / Moving average of gradients | 4 |
| Second moment（二阶矩） | 梯度平方的移动平均 / Moving average of squared gradients | 4 |
| **合计 / Total** | | **12** |

16 位浮点数占 2 字节，32 位浮点数占 4 字节。这里的精度分配是估算假设，不是所有 AdamW 实现的固定要求。

A 16-bit float uses 2 bytes and a 32-bit float uses 4 bytes. These storage precisions are assumptions, not universal AdamW requirements.

```python
h100_bytes = 80e9
bytes_per_parameter = 2 + 2 + (4 + 4)
num_parameters = (h100_bytes * 8) / bytes_per_parameter
# Approximately 53.3e9 parameters = 53.3B = 约 533 亿参数
```

总显存约 640 GB，因此 **640 × 10⁹ / 12 ≈ 53.3B 参数**。这是模型状态存储的理论上限，不能直接当作实际可训练规模。

Total nominal memory is 640 GB, giving **640 × 10⁹ / 12 ≈ 53.3B parameters**. This is a model-state-only upper bound, not a practical training capacity.

### 限制条件 / Limitations

- **激活值未计入 / Activations are excluded.** 前向传播的中间结果通常需要用于反向传播；批量大小（batch size）和序列长度（sequence length）越大，通常占用越多显存。 Intermediate results consume memory, typically increasing with batch size and sequence length.
- **可能需要 FP32 主权重 / FP32 master weights may be required.** 某些混合精度实现额外保留一份 32 位权重，每参数再加 4 字节；此时上限降为 640 / 16 = **40B**。 Some mixed-precision implementations add 4 bytes per parameter, reducing the model-state-only limit to **40B**.
- **必须合理分片 / Model states must be sharded.** 直接把 8 张卡的显存相加，假设参数、梯度和优化器状态可以分布到各卡；每卡保存完整副本时不能这样计算。 Adding GPU capacities assumes states are distributed across GPUs rather than fully replicated on each GPU.
- **还有运行开销 / Runtime overhead also consumes memory.** 通信缓冲区、临时工作空间等也需预留显存。 Communication buffers and temporary workspaces require additional memory.

优化器参考 / Optimizer reference: [PyTorch AdamW interface](https://docs.pytorch.org/docs/main/generated/torch.optim.adamw.adamw_function.html).

## 4. 70B 模型到底有多大？ / How large is a 70B model?

70B 表示 **700 亿个参数**。参数数量描述模型规模；存储大小还取决于每个参数使用的精度。

70B means **70 billion parameters**. Parameter count describes model scale; storage size also depends on precision.

| 精度 / Precision | 每参数字节数 / Bytes per parameter | 70B 权重大小 / 70B weights only |
|---|---:|---:|
| FP32 | 4 | 280 GB |
| FP16 / BF16 | 2 | 140 GB |
| INT8 | 1 | 70 GB |
| INT4 | 0.5 | 35 GB |

表中使用十进制 GB（10⁹ 字节），仅估算权重。量化格式还需要缩放系数等元数据，实际大小会有所增加；推理运行时还需要 KV cache 等额外空间。

These decimal GB estimates cover weights only. Quantized formats need additional metadata, and inference also requires runtime memory such as the KV cache.

按前面的每参数 12 字节假设，70B 模型训练状态需要 **70 × 10⁹ × 12 = 840 GB**，尚未包含激活值和临时空间。

Under the 12-byte assumption, training states for a 70B model require **840 GB**, excluding activations and temporary memory.

第二题的 53.3B 模型，FP16/BF16 权重本身约为 **107 GB**。

For the 53.3B model in the second estimate, FP16/BF16 weights alone occupy approximately **107 GB**.

## 5. 记住两个公式 / Two formulas to remember

```text
训练时间（秒）≈ 6 × 参数数量 × 训练 token 总数 / (单卡峰值 FLOP/s × MFU × GPU 数量)
Training time (seconds) ≈ 6 × parameters × training tokens / (peak FLOP/s per GPU × MFU × GPU count)

参数数量上限 ≈ 可用总显存（字节）/ 每参数训练状态占用（字节）
Parameter-count upper bound ≈ total available memory (bytes) / training-state bytes per parameter
```

前者估算计算时间，后者估算存储上限；显存装得下，并不代表训练速度足够快。

The first estimates compute time, while the second estimates storage capacity. Fitting in memory does not guarantee fast training.

## 6. CS336 Lecture 2：优化器为什么需要保存状态？

本节按官方 `lecture_02.py` 的 `optimizer()`、`AdaGrad`、`gradient_accumulation()` 与 `activation_checkpointing()` 整理。来源：[官方讲义代码](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。公式是课件的资源概算，以下特别标明补充解释与实现边界。

### 6.1 参数、梯度、激活、优化器状态是不同的东西

| 项目 | 保存什么 | 生命周期与作用 |
|---|---|---|
| 参数 parameter | 当前权重 W | 跨训练步保留，每次更新改变其值 |
| 梯度 gradient | 当前损失对 W 的导数 g | 反向产生；可跨 microbatch 累积，更新后通常清零 |
| 激活 activation | 前向的中间结果，如 h1 | 供本次反向使用；图释放后通常不再保留 |
| 优化器状态 optimizer state | 历史梯度的统计量 | 跨训练步保留，用于决定之后如何更新参数 |

**状态不是另一套模型参数，也不是当前梯度的同义词。** 例如同一个当前梯度，在历史梯度不同的情况下，AdaGrad 或 Adam 可以给出不同更新。

### 6.2 课件明确说了什么，哪些是补充解释？

课件明确列出：Momentum 使用梯度的指数平均；AdaGrad 累计梯度平方；RMSProp 改用梯度平方的指数平均；Adam 结合一阶、二阶统计。课件还明确说明，累计这些统计量时通常使用 FP32 来提高稳定性。下面对状态数量、公式与原因作展开解释。

记第 t 步梯度为 g_t，所有平方、除法等均逐元素执行：

| 优化器 | 每个参数需要保留的主要历史状态 | FP32 状态占用 |
|---|---|---:|
| 普通 SGD，无 momentum | 不保留历史梯度统计 | 0 bytes/参数 |
| Momentum | 一份梯度指数平均 m | 4 bytes/参数 |
| AdaGrad | 一份累计梯度平方 s | 4 bytes/参数 |
| RMSProp，基础形式 | 一份梯度平方指数平均 v | 4 bytes/参数 |
| Adam / AdamW | 梯度指数平均 m，以及梯度平方指数平均 v | 8 bytes/参数 |

表格忽略步数等很小的元数据，也不包含可选实现状态、FP32 主权重。带 momentum/centered 等选项的 RMSProp 会增加状态。

**AdaGrad：一份状态记录每个参数过去收到的梯度大小。**

```text
s_t = s_(t-1) + g_t²
w_t = w_(t-1) - learning_rate × g_t / sqrt(s_t + epsilon)
```

这是课件实现的形式。每个权重对应一个 s，因此若 W 有 N 个元素，s 也有 N 个元素。累计平方大的位置，后续有效步长更小。课件的 s 是累计和，不是除以步数后的算术平均；不同库的 epsilon 放置位置可能不同。

**Adam：需要两份状态，因为更新方向和梯度尺度分别估计。**

```text
m_t = beta1 × m_(t-1) + (1-beta1) × g_t
v_t = beta2 × v_(t-1) + (1-beta2) × g_t²
```

m 保留梯度方向的平滑信息，v 估计梯度平方的尺度。标准 Adam 使用偏差校正后的 m_hat、v_hat 更新；v 是未中心化二阶矩，不能直接称为梯度方差。AdamW 的解耦权重衰减通常不要求再加一份逐参数历史数组。

**为什么状态使用 FP32？（课件给出稳定性理由，这里展开）** 状态跨很多步递推，累计量可能越来越大，而新的增量很小。低精度的舍入可能让小增量消失；BF16 虽然指数范围大，但有效精度仍较低。FP32 可以减轻这些累计误差。它不是所有实现自动遵守的规定，而是课件内存估算采用的配置。

### 6.3 课件代码与内存公式的精度不同

课件 `torch.randn(...)` 默认生成 FP32 参数，梯度通常也是 FP32；其自定义 AdaGrad 用 `torch.zeros_like(grad)` 创建状态，状态跟随梯度的 dtype，并没有显式转换成 FP32。

但后面的内存核算假设：**参数、梯度、激活 BF16；优化器状态 FP32。** 因此那段公式是一个混合精度预算，不是前面代码实测的字节数。仅把参数改成 BF16，也不会让该自定义优化器自动拥有 FP32 状态，需要另行实现转换。一般的 autocast 也不能理解为把所有持久参数和状态统一改成 BF16。

## 7. Memory：一次训练需要存哪些东西？

课件示例：B = 2，D = 4，L = 3；每层一个 D×D 权重矩阵，然后做 ReLU，没有 bias。

```python
num_parameters = D * D * L  # 4 × 4 × 3 = 48
parameter_memory = 2 * num_parameters
gradient_memory = 2 * num_parameters
optimizer_state_memory = 4 * num_parameters  # AdaGrad
activation_memory = 2 * B * D * L
```

每个参数的梯度占一个元素。ReLU 没有可训练参数。

| 项目 | 课件公式，单位 bytes | 本例结果 |
|---|---|---:|
| 参数 | 2N | 96 |
| 梯度 | 2N | 96 |
| AdaGrad 状态 | 4N | 192 |
| 激活 | 2BDL | 48 |
| **合计** | **8N + 2BDL** | **432 bytes** |

若换成 Adam，状态变为 8N = 384 bytes，总计 **12N + 2BDL = 624 bytes**。差别来自 Adam 多存了一份 N 个 FP32 元素的历史统计量。

**为什么激活与 B 有关，参数与 B 无关？** 一个 batch 的样本共用权重，但每个样本的中间结果不同。这里每层约存 B×D 个激活，共 L 层。

这些是简化预算：没有逐项统计输入、损失、ReLU 所需保存的具体张量、临时工作空间、分配器预留等，不能视为实际峰值显存。实际保存哪些张量取决于自动求导与算子实现。

## 8. Compute for one step：为什么是 6BN？

这里 B 是一个训练步处理的样本数，D 是隐藏维度，L 是层数，N = LD² 是参数量。注意本文前面的 6ND 使用 D 表示训练 token 总量；本节的 D 改为课件的隐藏维度。

对一层 Y = XW：

| 阶段 | 矩阵运算 | 主导 FLOPs |
|---|---|---:|
| 前向 | XW | 2BD² |
| 反向：权重梯度 | Xᵀ × Y.grad | 2BD² |
| 反向：输入梯度 | Y.grad × Wᵀ | 2BD² |

前面的 2 是乘法与加法各计一次；反向的两个乘法分别产生权重梯度和传回前层的梯度。推广到 L 层：

```text
前向 ≈ 2BLD² = 2BN
反向 ≈ 4BLD² = 4BN
单步 ≈ 6BLD² = 6BN
```

代入课件 B=2、D=4、L=3、N=48：前向约 192 FLOPs，反向约 384 FLOPs，合计 **576 FLOPs**。

这是主导矩阵运算的概算，不是完整指令计数。原始输入不需要梯度时，首层会少一次输入梯度矩阵乘法；ReLU、损失、优化器更新另有计算。小型演示中这些差异比例可能较大；大模型估算时通常先抓主导项。

训练顺序：前向得到 loss → `loss.backward()` 产生梯度 → `optimizer.step()` 更新参数和优化器状态 → `optimizer.zero_grad(set_to_none=True)` 清理梯度。**清理梯度不会清理 AdaGrad/Adam 的历史状态**，否则历史学习信息就丢失了。

## 9. Gradient accumulation：用小批次累积出大批次梯度

### 9.1 课件数字示例

课件取 B=64、D=1024、L=16，采用 BF16 激活估算：

```text
一次处理 64 个样本：2 × 64 × 1024 × 16 = 2,097,152 bytes = 2 MiB
改成每次处理 16 个：2 × 16 × 1024 × 16 =   524,288 bytes = 0.5 MiB
累计次数：64 / 16 = 4
```

依次对四个 microbatch 做前向和反向，将梯度累加到同一份 `.grad`；最后只更新一次。每次反向完成后释放对应计算图，不必同时保留四份激活，因此激活预算降到约四分之一。

### 9.2 补充：可执行流程的写法

以下展开课件的流程说明，假设四个 microbatch 大小相同，损失使用 mean，且各样本计算相互独立：

```python
accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)
for x_micro, y_micro in four_microbatches:
    loss = loss_fn(model(x_micro), y_micro)
    (loss / accumulation_steps).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

经验与易错点：

- 节省的是峰值激活内存，不会把参数、梯度数组或优化器状态缩小四倍。
- mean loss 要按样本数正确加权。等大小时除以 4；大小不等时按各 microbatch 的样本数占比加权。按 token 平均的任务应使用有效 token 数。
- 不要每个 microbatch 都清零梯度或更新参数，否则不再是预期的大 batch 累积。
- 四次反向通常不需要 `retain_graph=True`，因为每次前向建立各自的图。
- 总样本计算量大致不变，但小矩阵的数据复用、GPU 利用率可能更差，启动次数也更多；省显存不保证加速。
- 有 BatchNorm、随机性或跨样本计算时，不保证与完整大 batch 的结果严格一致。

## 10. Activation checkpointing：少存激活，反向时重算

### 10.1 为什么可以重算？

例如 Y=XW 的权重梯度是 Xᵀ×Y.grad，反向需要前向的 X。普通训练把必要的中间张量保留下来；checkpointing 只保存部分边界输入，反向时重新执行局部前向，恢复所需张量。

这与把模型保存到磁盘的训练 checkpoint 不同：这里的 checkpoint 是供反向重计算使用的激活边界。

### 10.2 课件数字与演示

课件设置 B=64、D=1024、L=16：普通激活概算同样为 **2 MiB**（BF16 假设）。演示输入显式使用 `requires_grad=True`，并通过 `get_max_memory_usage(...)` 比较普通与 checkpoint 版本的一次前向和反向。

课件的模型每层包含“线性变换 + ReLU”。它在循环中把：

```python
x = layer(x)
```

换成：

```python
x = torch.utils.checkpoint.checkpoint(layer, x)
```

以上保留课件写法。核心效果是保留层边界、减少层内部中间张量的保存。**每层包 checkpoint 并不等于整网只存一个激活，也不自动得到 O(sqrt(L)) 的内存复杂度。** 具体测量数字取决于设备、dtype、实现与运行环境，课件源码没有可作为通用结论的固定实测值。

### 10.3 保存间隔的取舍

课件进一步用线性链网络说明：

| 策略 | 激活内存随层数变化 | 重计算代价 |
|---|---|---|
| 保存所有层需要的激活 | O(L) | 无额外前向重算 |
| 不存中间激活，每次从起点恢复 | 理想化 O(1) | 朴素策略 O(L²) |
| 每约 sqrt(L) 层保存一个边界 | O(sqrt(L)) | 额外 O(L) 重计算 |

补充推导：每 k 层一段，约存 L/k 个边界；反向处理一段时，暂存该段约 k 层激活，因此内存约与 **L/k + k** 成正比，在 k≈sqrt(L) 时平衡。L=16 时可按约 4 层分段；这是复杂度直觉，不能据此断言总显存恰好下降四倍。

经验：checkpointing 用额外计算换激活内存；它不直接缩小参数或优化器状态。省下的空间可能允许更大的 microbatch，从而改善利用率，但不能无条件认为它会提速。重计算必须与原前向保持一致，含随机操作、状态变更的模块需要特别注意。

## 11. 一页回顾：根据占用来源选择方法

| 当前瓶颈 | 优先理解的办法 | 代价或边界 |
|---|---|---|
| batch 太大导致激活占用高 | 梯度累积 | microbatch 变小，吞吐可能下降 |
| 深层网络保存中间结果太多 | 激活 checkpointing | 反向阶段增加前向重计算 |
| 优化器状态占用高 | 核对状态份数与精度，再考虑分片等方案 | 上述两种激活优化不能直接解决状态占用 |
| 想估算一个训练步 | 参数 N，激活 BDL，计算约 6BN | 区分估算假设与实际峰值测量 |

两种激活优化可以一起用：先把大 batch 分成 microbatch，再对每个 microbatch 的模型执行激活 checkpointing。复习时始终问：**保存的是什么、保存多久、改变哪个量才能减少它、代价是什么？**
