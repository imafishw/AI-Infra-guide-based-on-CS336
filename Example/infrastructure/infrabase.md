# CS336 Lecture 2 学习笔记：计算、显存与性能瓶颈

> 学习目标：从张量形状推导 FLOPs 与存储量，判断计算/带宽瓶颈，并解释梯度累积与激活重算的取舍。

内容依据 [官方 Lecture 2](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。
**阅读主线：形状与 FLOPs → 带宽与 Roofline → 训练状态与内存 → 优化策略 → 资源估算 → 工程综合题。** 课件假设与补充解释分别标明；所有资源数字都是概算，不是实际设备测量。

## 目录

**知识点（由浅入深）**

- [1. 符号与计量：先分清容量、工作量和速度](#section-1)
- [2. Compute for one step：为什么是 6BN？](#section-2)
- [3. 从 Arithmetic intensity 到 Roofline，再到 MFU](#section-3)
- [4. CS336 Lecture 2：优化器为什么需要保存状态？](#section-4)
- [5. Memory：一次训练需要存哪些东西？](#section-5)
- [6. Gradient accumulation：用小批次累积出大批次梯度](#section-6)
- [7. Activation checkpointing：少存激活，反向时重算](#section-7)
- [8. 一页回顾：根据占用来源选择方法](#section-8)

**例题（应用知识点）**

- [9. 用 1024 张 H100 训练 70B 模型、处理 15T tokens，需要多久？](#section-9)
- [10. 用 AdamW 在 8 张 80 GB H100 上，最多能训练多大的模型？](#section-10)
- [11. 工程综合例题：显存够，为什么推理仍然慢？](#section-11)

<a id="section-1"></a>

## 1. 符号与计量：先分清容量、工作量和速度

> 自测提示：FLOPs 与 FLOP/s、显存容量与带宽分别有什么区别？

| 符号 / 术语 | 含义 | 单位或说明 |
|---|---|---|
| N | 参数数量 | 权重元素数，不含 batch 维 |
| B | 一次算子的 batch 大小 | 不要与 70B 中的 billion 混淆 |
| D、K、L | 输入维度、输出维度、层数 | 本文用 T 表示训练 token 总数，避免 D 一符多义 |
| T | 累计训练 token 数 | token 不一定等于一个字，累计数不一定去重 |
| F | 运算量 | FLOPs |
| Q | 数据搬运量 | bytes；重复读取需重复计算 |
| P、W | 峰值算力、显存带宽 | FLOP/s、bytes/s |
| I、A | 算术强度、加速器强度 | 都是 FLOPs/byte |
| MFU | 有效模型吞吐 / 硬件峰值 | 不等于监控面板 GPU 使用率 |

张量逻辑存储量为：

$$
M=\operatorname{numel}(X)\times\text{每元素字节数}
$$

FP32 为 4 bytes，FP16/BF16 为 2 bytes。BF16 的指数范围接近 FP32，但有效精度较低。GB 使用十进制 $10^9$ bytes；MiB 使用 $2^{20}$ bytes。**容量回答放不放得下，带宽回答搬得够不够快。**

<a id="section-2"></a>

## 2. Compute for one step：为什么是 6BN？

> 自测提示：为什么前向是 2BN，反向约为 4BN？

### 从输出元素数推导 FLOPs

对 $X\in\mathbb R^{B\times D}$、$W\in\mathbb R^{D\times K}$，输出 $Y=XW$ 的形状为 $B\times K$。无 bias 时参数量为 $N=DK$。

$$
Y_{ik}=\sum_{j=1}^{D}X_{ij}W_{jk}
$$

每个输出需要 D 次乘法、D−1 次加法，共有 BK 个输出：

$$
F=BK(2D-1)\approx2BDK=2BN
$$

**前面的 2 来自乘法和加法，不是来自前向与反向。** 通用口诀：$(m,k)@(k,n)$ 约需 $2mkn$ FLOPs。

### 反向为什么有两次矩阵乘法？

设 $G=\partial\mathcal L/\partial Y$ 为后面传回的梯度：

$$
\frac{\partial\mathcal L}{\partial W}=X^\top G,
\qquad
\frac{\partial\mathcal L}{\partial X}=GW^\top
$$

前者更新权重所需，后者把梯度传给前一层。例如课件 `h2 = h1 @ w2` 的 `h1_grad = h2.grad @ w2.T`；同一路径的导数相乘，不同输出路径的贡献相加。


这里 B 是一个训练步处理的样本数，D 是隐藏维度，L 是层数，N = LD² 是参数量。

对一层 Y = XW：

| 阶段 | 矩阵运算 | 主导 FLOPs |
|---|---|---:|
| 前向 | XW | 2BD² |
| 反向：权重梯度 | Xᵀ × Y.grad | 2BD² |
| 反向：输入梯度 | Y.grad × Wᵀ | 2BD² |

前面的 2 是乘法与加法各计一次；反向的两个乘法分别产生权重梯度和传回前层的梯度。推广到 L 层：

$$
F_{\mathrm{forward}}\approx2BN,\qquad
F_{\mathrm{backward}}\approx4BN,\qquad
F_{\mathrm{step}}\approx6BN
$$

代入课件 B=2、D=4、L=3、N=48：前向约 192 FLOPs，反向约 384 FLOPs，合计 **576 FLOPs**。

这是主导矩阵运算的概算，不是完整指令计数。原始输入不需要梯度时，首层会少一次输入梯度矩阵乘法；ReLU、损失、优化器更新另有计算。小型演示中这些差异比例可能较大；大模型估算时通常先抓主导项。

训练顺序：前向得到 loss → `loss.backward()` 产生梯度 → `optimizer.step()` 更新参数和优化器状态 → `optimizer.zero_grad(set_to_none=True)` 清理梯度。**清理梯度不会清理 AdaGrad/Adam 的历史状态**，否则历史学习信息就丢失了。

<a id="section-3"></a>

## 3. 从 Arithmetic intensity 到 Roofline，再到 MFU

> 自测提示：任务位于屋顶拐点哪一侧？实测 MFU 为什么可能低于屋顶上限？

前面的显存核算回答“能不能放下”；本节回答“放下后能跑多快”。标准术语是 **memory-bound** 和 **compute-bound**，不是 memory bond / computing bond。来源：[课件 `arithmetic_intensity()` 与 `roofline_plots()`](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)、[课件引用的 Roofline 说明](https://jax-ml.github.io/scaling-book/roofline/)。

### 3.1 两种瓶颈与两个 intensity

记 F 为任务计算量（FLOPs），Q 为实际数据搬运量（bytes），P 为硬件峰值算力（FLOP/s），W 为显存带宽（bytes/s）。这里主要关注 GPU 显存与片上计算资源之间的数据搬运。

```text
Arithmetic intensity   I = F / Q    单位 FLOPs/byte：任务每搬一个字节做多少计算
Accelerator intensity  A = P / W    单位 FLOPs/byte：硬件算力与带宽的比例

理想计算时间 = F / P
理想搬运时间 = Q / W
总耗时下限   = max(F/P, Q/W)        假设计算与搬运充分重叠
```

| 判断 | 时间关系 | 瓶颈 | 优化方向 |
|---|---|---|---|
| I < A | F/P < Q/W | Memory-bound：数据供给跟不上 | 减少读写、提高复用、融合算子、增加带宽 |
| I > A | F/P > Q/W | Compute-bound：计算限制吞吐 | 提高有效算力、改善计算实现、减少必要计算 |
| I ≈ A | 两者接近 | 接近平衡点 | 结合实际测量判断 |

**容量和带宽不同：**OOM 表示空间不足；memory-bound 表示搬运速度限制性能。Q 也不等于张量占用空间：同一数据多次从显存读取，要重复计入搬运量。算术强度高并不表示耗时短，任务总计算量仍可能很大。

### 3.2 按课件计算：ReLU、点积、矩阵乘法

课件使用 BF16，每元素 2 bytes；硬件假设为 H100 的稠密 BF16 峰值与显存带宽：

```python
h100_flop_per_sec = 1979e12 / 2
h100_bytes_per_sec = 3.35e12
accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec
# ≈ 295.37 FLOPs/byte
```

| 操作 | F：计算量 | Q：理想读写量 | I = F/Q | 按课件模型判断 |
|---|---:|---:|---:|---|
| ReLU，n 个元素 | n 次比较 | 2n + 2n | 0.25 | Memory-bound |
| 两个长度 n 向量点积 | 2n−1 | 2n + 2n + 2 | ≈ 0.5 | Memory-bound |
| 向量 × n×n 矩阵 | n(2n−1) | 2n + 2n² + 2n | ≈ 1 | Memory-bound |
| 两个 n×n 矩阵相乘 | n²(2n−1) | 2n² + 2n² + 2n² | ≈ n/3 | n=1024 时约 341.17，高于拐点 |

以 ReLU 为例，读取 x、比较、写回 y：

```python
n = 1024 * 1024
bytes_moved = 2 * n + 2 * n
flops = n
arithmetic_intensity = flops / bytes_moved  # 0.25
memory_time = bytes_moved / h100_bytes_per_sec  # ≈ 1.25 μs
compute_time = flops / h100_flop_per_sec       # ≈ 0.00106 μs
```

因此该简化模型中搬运时间远大于计算时间。增大 n 会同时增加 F 和 Q，不会提高 ReLU 的算术强度。矩阵乘法则可复用同一个权重处理多个输入，算术强度可能随规模提高。

**核算边界：**课件将比较计作一次操作，并用统一峰值作教学比较；真实 ReLU 不使用矩阵 Tensor Core 峰值，精确分析应采用对应执行单元的吞吐。表中假设输入各读一次、输出写一次；缓存、分块和重复读取会改变真实 Q。小算子也可能主要受启动开销限制，不能只按运算名称分类。

### 3.3 `roofline_plots()`：图怎么读？

![课件引用的 Roofline 图：算术强度与理论吞吐上限](https://jax-ml.github.io/scaling-book/assets/img/roofline-improved-1400.webp)

图片来自课件引用的 [How To Scale Your Model — Rooflines](https://jax-ml.github.io/scaling-book/roofline/)。它是概念图，下面的 H100 数字是另行代入课程假设的例子；图片需联网加载。

- 横轴是任务的算术强度 I（FLOPs/byte），不是数据总量。
- 纵轴是吞吐率（FLOP/s），不是总 FLOPs，也不是耗时。
- 原图使用双对数坐标。左边上升线受带宽限制，右边水平线受峰值算力限制。

```text
吞吐率
  ↑
P |                 ┌──────────────── 计算屋顶：P
  |               ／
  |             ／  带宽约束：W × I
  |           ／
  +────────────────────────────────→ 算术强度 I
                  P/W
       memory-bound │ compute-bound
```

屋顶公式：**R_roof = min(P, W × I)**。斜线与水平线相交时 W×I=P，因此拐点横坐标恰好是 accelerator intensity，A=P/W。

读图步骤：先算任务 I → 在横轴定位 → 向上找到对应硬件屋顶 → 纵坐标就是理想吞吐上限。实际测量点通常落在屋顶下方；曲线不是保证能达到的速度。

原图 BW1、BW2 对应不同带宽：带宽更高，斜线更高，拐点左移；峰值算力相同则水平屋顶不变。提高数据复用相当于将任务向右移动；只增加算力会抬高水平屋顶，不能直接解决左侧任务的带宽瓶颈。

### 3.4 MFU：实际有效吞吐占峰值多少？

后面的训练时间例题会使用 MFU=0.5；先明确它的含义与限制。

```text
单卡实测 MFU = 有效模型 FLOPs / (实测耗时 × 单卡对应精度的峰值 FLOP/s)
多卡实测 MFU = 有效模型 FLOPs / (实测耗时 × GPU 数量 × 单卡峰值 FLOP/s)
```

例如单卡每步有效模型计算量为 100 TFLOPs，耗时 0.2 秒，峰值为 1000 TFLOP/s：实际有效吞吐为 500 TFLOP/s，MFU=50%。它不等于监控工具显示的 GPU 利用率；设备一直忙，也可能主要忙于搬运。

课件 `roofline_plots()` 写出的关系，在理想模型下是：

```text
MFU_ideal = R_roof / P
          = min(P, W×I) / P
          = min(1, I/A)
```

这表示屋顶模型允许的利用率上限，不是所有程序的实测 MFU 恒等式。真实值还受到算子执行效率、启动、通信等影响；使用峰值时要匹配设备、精度、稀疏性与计算单元。

按课程 ReLU 的统一算力教学口径，I/A≈0.25/295.37≈0.000846，即约 **0.0846%**；这不是对实际 ReLU 做出的精确 MFU 测量。n=1024 的方阵乘法 I≈341.17，屋顶模型上限为 100%，实际不保证达到。

### 3.5 与梯度累积、激活 checkpointing 的联系

- 梯度累积降低同时存在的激活，但更小的 microbatch 可能降低矩阵乘法的数据复用，因而降低算术强度和吞吐。有效 batch 大，不代表每次算子看到的矩阵也大。
- Checkpointing 减少激活保存，却增加重计算。标准模型 MFU 的有效 FLOPs 通常不把额外重计算当作新增模型工作，因此重计算更多不意味着 MFU 更高。
- 两者可能通过释放容量允许更合适的 batch 配置。最终应比较每步耗时或 tokens/s，并检查训练语义一致，不能只看显存下降或 GPU 忙碌程度。


本节公式卡：

$$
I=\frac FQ,\quad A=\frac PW,\quad
 t\ge\max\left(\frac FP,\frac QW\right)
$$
$$
R_{\mathrm{roof}}=\min(P,WI),\qquad
\mathrm{MFU}_{\mathrm{ideal}}=\min\left(1,\frac IA\right)
$$
$$
\mathrm{MFU}_{\mathrm{measured}}=
\frac{F_{\mathrm{model}}}{t\,n_{\mathrm{GPU}}\,P}
$$

<a id="section-4"></a>

## 4. CS336 Lecture 2：优化器为什么需要保存状态？

> 自测提示：清零梯度时为什么不能一起清零优化器状态？

本节按官方 `lecture_02.py` 的 `optimizer()`、`AdaGrad`、`gradient_accumulation()` 与 `activation_checkpointing()` 整理。来源：[官方讲义代码](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。公式是课件的资源概算，以下特别标明补充解释与实现边界。

### 4.1 参数、梯度、激活、优化器状态是不同的东西

| 项目 | 保存什么 | 生命周期与作用 |
|---|---|---|
| 参数 parameter | 当前权重 W | 跨训练步保留，每次更新改变其值 |
| 梯度 gradient | 当前损失对 W 的导数 g | 反向产生；可跨 microbatch 累积，更新后通常清零 |
| 激活 activation | 前向的中间结果，如 h1 | 供本次反向使用；图释放后通常不再保留 |
| 优化器状态 optimizer state | 历史梯度的统计量 | 跨训练步保留，用于决定之后如何更新参数 |

**状态不是另一套模型参数，也不是当前梯度的同义词。** 例如同一个当前梯度，在历史梯度不同的情况下，AdaGrad 或 Adam 可以给出不同更新。

### 4.2 课件明确说了什么，哪些是补充解释？

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

$$
s_t=s_{t-1}+g_t^2,\qquad
w_t=w_{t-1}-\eta\frac{g_t}{\sqrt{s_t+\epsilon}}
$$

这是课件实现的形式。每个权重对应一个 s，因此若 W 有 N 个元素，s 也有 N 个元素。累计平方大的位置，后续有效步长更小。课件的 s 是累计和，不是除以步数后的算术平均；不同库的 epsilon 放置位置可能不同。

**Adam：需要两份状态，因为更新方向和梯度尺度分别估计。**

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

m 保留梯度方向的平滑信息，v 估计梯度平方的尺度。标准 Adam 使用偏差校正后的 m_hat、v_hat 更新；v 是未中心化二阶矩，不能直接称为梯度方差。AdamW 的解耦权重衰减通常不要求再加一份逐参数历史数组。

**为什么状态使用 FP32？（课件给出稳定性理由，这里展开）** 状态跨很多步递推，累计量可能越来越大，而新的增量很小。低精度的舍入可能让小增量消失；BF16 虽然指数范围大，但有效精度仍较低。FP32 可以减轻这些累计误差。它不是所有实现自动遵守的规定，而是课件内存估算采用的配置。

### 4.3 课件代码与内存公式的精度不同

课件 `torch.randn(...)` 默认生成 FP32 参数，梯度通常也是 FP32；其自定义 AdaGrad 用 `torch.zeros_like(grad)` 创建状态，状态跟随梯度的 dtype，并没有显式转换成 FP32。

但后面的内存核算假设：**参数、梯度、激活 BF16；优化器状态 FP32。** 因此那段公式是一个混合精度预算，不是前面代码实测的字节数。仅把参数改成 BF16，也不会让该自定义优化器自动拥有 FP32 状态，需要另行实现转换。一般的 autocast 也不能理解为把所有持久参数和状态统一改成 BF16。

<a id="section-5"></a>

## 5. Memory：一次训练需要存哪些东西？

> 自测提示：增加 batch 会增加哪些存储，不会增加哪些存储？

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

<a id="section-6"></a>

## 6. Gradient accumulation：用小批次累积出大批次梯度

> 自测提示：四个 microbatch 怎样得到一次大 batch 更新？

### 6.1 课件数字示例

课件取 B=64、D=1024、L=16，采用 BF16 激活估算：

```text
一次处理 64 个样本：2 × 64 × 1024 × 16 = 2,097,152 bytes = 2 MiB
改成每次处理 16 个：2 × 16 × 1024 × 16 =   524,288 bytes = 0.5 MiB
累计次数：64 / 16 = 4
```

依次对四个 microbatch 做前向和反向，将梯度累加到同一份 `.grad`；最后只更新一次。每次反向完成后释放对应计算图，不必同时保留四份激活，因此激活预算降到约四分之一。

### 6.2 补充：可执行流程的写法

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

<a id="section-7"></a>

## 7. Activation checkpointing：少存激活，反向时重算

> 自测提示：为什么重算能节省内存？它与保存模型到磁盘有什么区别？

### 7.1 为什么可以重算？

**这里的激活检查点主要用于训练，不是普通推理的必需步骤。** “选择激活”指选择保存哪些中间结果，不是选择训练哪些层，也不是跳过其他层。前向仍然经过全部层，模型参数量不变。

激活就是数据经过某一层产生的中间结果。例如 `h = relu(x @ W)` 中的 h；本例每层输出约有 B×D 个数。参数 W 是模型要学习的数，激活 h 则随输入数据而变化。

| 对比 | 训练 | 普通推理（不计算梯度） |
|---|---|---|
| 是否需要反向传播 | 需要，用于计算参数梯度 | 不需要 |
| 激活的用途 | 计算后续输出，还要供反向使用 | 计算后续输出 |
| 何时可以释放 | 相关反向完成后，或采用重算策略 | 后续计算不再需要时 |
| 激活 checkpointing | 少保存激活，反向时重算，以计算换显存 | 通常没有必要 |

推理也会产生激活，只是不必为了反向传播保留各层结果。对简单顺序网络，可以逐层释放不再使用的结果；有残差连接时仍需保留后续会用到的张量，Transformer 自回归推理还可能保存 KV cache，不能理解成推理只占权重内存。

例如 Y=XW 的权重梯度是 Xᵀ×Y.grad，反向需要前向的 X。普通训练把必要的中间张量保留下来；checkpointing 只保存部分边界输入，反向时重新执行局部前向，恢复所需张量。

**直观例子：9 层网络，每 3 层保存一个边界。**

```text
前向经过：输入 → h1 → h2 → h3 → h4 → h5 → h6 → h7 → h8 → h9 → loss
长期保留：输入             h3             h6             h9
```

这里是分段策略的示意，不是自动求导实际保存张量的精确清单。h1、h2 等仍然在前向时算过，只是不一直保存到反向。

1. 反向处理第 9～7 层：从保存的 h6 出发，重算这一段前向，恢复所需中间结果，然后完成这一段反向。
2. 释放这一段不再需要的临时结果，再从 h3 出发，重算并反向处理第 6～4 层。
3. 最后从输入出发，重算并反向处理第 3～1 层。

就像只保留解题过程的几个关键步骤，需要细节时从最近的关键步骤重新推导。节省的是同时保存的激活，而不是模型权重或参数梯度的数量。

这与把模型保存到磁盘的训练 checkpoint 不同：这里的 checkpoint 是供反向重计算使用的激活边界。

### 7.2 课件数字与演示

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

### 7.3 保存间隔的取舍

课件进一步用线性链网络说明：

| 策略 | 激活内存随层数变化 | 重计算代价 |
|---|---|---|
| 保存所有层需要的激活 | O(L) | 无额外前向重算 |
| 不存中间激活，每次从起点恢复 | 理想化 O(1) | 朴素策略 O(L²) |
| 每约 sqrt(L) 层保存一个边界 | O(sqrt(L)) | 额外 O(L) 重计算 |

补充推导：每 k 层一段，约存 L/k 个边界；反向处理一段时，暂存该段约 k 层激活，因此内存约与 **L/k + k** 成正比，在 k≈sqrt(L) 时平衡。L=16 时可按约 4 层分段；这是复杂度直觉，不能据此断言总显存恰好下降四倍。

经验：checkpointing 用额外计算换激活内存；它不直接缩小参数或优化器状态。省下的空间可能允许更大的 microbatch，从而改善利用率，但不能无条件认为它会提速。重计算必须与原前向保持一致，含随机操作、状态变更的模块需要特别注意。


分段激活预算可概括为：

$$
M_{\mathrm{activation}}\propto\frac{L}{k}+k,
\qquad k\approx\sqrt L
$$

<a id="section-8"></a>

## 8. 一页回顾：根据占用来源选择方法

> 自测提示：面对 OOM 和低吞吐，分别先检查什么？

| 当前瓶颈 | 优先理解的办法 | 代价或边界 |
|---|---|---|
| batch 太大导致激活占用高 | 梯度累积 | microbatch 变小，吞吐可能下降 |
| 深层网络保存中间结果太多 | 激活 checkpointing | 反向阶段增加前向重计算 |
| 优化器状态占用高 | 核对状态份数与精度，再考虑分片等方案 | 上述两种激活优化不能直接解决状态占用 |
| 显存够但吞吐低 | 用 F/Q 与 P/W 判断瓶颈，参考 Roofline 与实测 MFU | 低 MFU 不能单独证明带宽受限 |
| 想估算一个训练步 | 参数 N，激活 BDL，计算约 6BN | 区分估算假设与实际峰值测量 |

两种激活优化可以一起用：先把大 batch 分成 microbatch，再对每个 microbatch 的模型执行激活 checkpointing。复习时始终问：**保存的是什么、保存多久、改变哪个量才能减少它、代价是什么？**

<a id="section-9"></a>

## 9. 用 1024 张 H100 训练 70B 模型、处理 15T tokens，需要多久？

> 自测提示：为什么不能直接用峰值算力估算训练时间？


### 总计算量 / Total computation

稠密 Transformer 训练的常用近似公式为 **C ≈ 6NT**：N 是参数数量，T 是训练 token 总数。前向传播约需 2NT，反向传播约需 4NT。


- 前向传播 / Forward pass：根据输入生成预测 / Produce predictions from inputs.
- 反向传播 / Backward pass：计算用于更新参数的梯度 / Compute gradients for parameter updates.

```python
total_flops = 6 * 70e9 * 15e12  # 6.3e24 FLOPs
```

该公式是近似值，没有完整计算注意力等额外开销；长序列下偏差可能更明显。


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
- **MFU = 0.5**：假设有效模型计算达到理论峰值的 50%，实际效率受通信、内存访问和等待等影响。
- **86400**：一天的秒数，把每秒吞吐转换为每天吞吐。

每天有效计算量约为 **4.38 × 10²² FLOPs**，所以训练时间约为 **144 天**。


这是假设集群持续维持该效率的估计；停机、评估、保存检查点等可能增加实际日历时间。


规格参考 / Hardware reference: [NVIDIA H100 GPU Whitepaper](https://resources.nvidia.com/en-us-hopper-architecture).



$$
t_{\mathrm{train}}\approx\frac{6NT}{n_{\mathrm{GPU}}P\,\mathrm{MFU}}
$$

<a id="section-10"></a>

## 10. 用 AdamW 在 8 张 80 GB H100 上，最多能训练多大的模型？

> 自测提示：为什么把八张 GPU 的显存相加需要额外假设？


AdamW 是使用梯度的一阶矩、二阶矩并采用解耦权重衰减的优化器。训练需要保存参数以及更新参数所需的状态。


| 存储项 / Item | 作用 / Purpose | 本例每参数占用 / Bytes per parameter |
|---|---|---:|
| Parameters（参数） | 当前模型权重 / Current model weights | 2 |
| Gradients（梯度） | 损失对参数的变化率 / Rate of change of loss with respect to parameters | 2 |
| First moment（一阶矩） | 梯度的移动平均 / Moving average of gradients | 4 |
| Second moment（二阶矩） | 梯度平方的移动平均 / Moving average of squared gradients | 4 |
| **合计 / Total** | | **12** |

16 位浮点数占 2 字节，32 位浮点数占 4 字节。这里的精度分配是估算假设，不是所有 AdamW 实现的固定要求。


```python
h100_bytes = 80e9
bytes_per_parameter = 2 + 2 + (4 + 4)
num_parameters = (h100_bytes * 8) / bytes_per_parameter
# Approximately 53.3e9 parameters = 53.3B = 约 533 亿参数
```

总显存约 640 GB，因此 **640 × 10⁹ / 12 ≈ 53.3B 参数**。这是模型状态存储的理论上限，不能直接当作实际可训练规模。


### 限制条件 / Limitations

- **激活值未计入 / Activations are excluded.** 前向传播的中间结果通常需要用于反向传播；批量大小（batch size）和序列长度（sequence length）越大，通常占用越多显存。 Intermediate results consume memory, typically increasing with batch size and sequence length.
- **可能需要 FP32 主权重 / FP32 master weights may be required.** 某些混合精度实现额外保留一份 32 位权重，每参数再加 4 字节；此时上限降为 640 / 16 = **40B**。 Some mixed-precision implementations add 4 bytes per parameter, reducing the model-state-only limit to **40B**.
- **必须合理分片 / Model states must be sharded.** 直接把 8 张卡的显存相加，假设参数、梯度和优化器状态可以分布到各卡；每卡保存完整副本时不能这样计算。 Adding GPU capacities assumes states are distributed across GPUs rather than fully replicated on each GPU.
- **还有运行开销 / Runtime overhead also consumes memory.** 通信缓冲区、临时工作空间等也需预留显存。 Communication buffers and temporary workspaces require additional memory.

优化器参考 / Optimizer reference: [PyTorch AdamW interface](https://docs.pytorch.org/docs/main/generated/torch.optim.adamw.adamw_function.html).



### 参数量与权重大小：70B 的直觉

70B 为 700 亿参数：仅权重 FP32 为 280 GB，BF16 为 140 GB；按每参数 12 bytes 的训练状态假设则为 840 GB。INT8/INT4 的原始权重载荷约为 70/35 GB，但量化元数据与运行时缓存需另计。

$$N_{\max}\lesssim\frac{M_{\mathrm{available}}}{\text{每参数训练状态字节数}}$$

<a id="section-11"></a>

## 11. 工程综合例题：显存够，为什么推理仍然慢？

### 题目：先自己列式，再查看解答

```python
X.shape = (B, 4096)
W.shape = (4096, 4096)
Y = X @ W
```

数据均为 BF16。GPU 峰值算力为 100 TFLOP/s，带宽为 1 TB/s；假设输入、权重各读一次，输出写一次，忽略启动与其他算子开销。当前 B=1。

1. FLOPs 和搬运量分别是多少？
2. 两个 intensity 是多少，属于哪种瓶颈？
3. 换成算力翻倍但带宽不变的 GPU，能快一倍吗？
4. B=128 时是否跨过 Roofline 拐点？为什么不能只根据吞吐决定线上 batch？

### 解答 1：从一个输出开始数 FLOPs

例如 $y=x_1w_1+x_2w_2+x_3w_3$ 需要 3 次乘法、2 次加法，共 5 FLOPs，近似记为 6。

本题输出有 $B\times4096$ 个数，每个是长度 4096 的点积：

$$
F=B\times4096\times(2\times4096-1)
\approx2B\times4096^2
$$

后续统一使用乘加近似。B=1 时：

$$F\approx33{,}554{,}432\ \mathrm{FLOPs}=33.55\ \mathrm{MFLOPs}$$

**记忆方法：输出元素个数 × 每个输出需要的乘加次数。**

### 解答 2：读输入、读权重、写输出

| 操作 | 字节数 |
|---|---:|
| 读 X | $2B\times4096$ |
| 读 W | $2\times4096^2$ |
| 写 Y | $2B\times4096$ |

$$Q=2(4096^2+2B\times4096)$$

B=1 时，Q=33,570,816 bytes，约 33.57 MB；绝大部分是权重。

$$
I=\frac FQ\approx1\ \mathrm{FLOP/byte},\qquad
A=\frac{100\times10^{12}}{10^{12}}=100\ \mathrm{FLOPs/byte}
$$

$I<A$，因此 **memory-bound**。容量占用小不能说明带宽开销小。

### 解答 3：比较时间，而不是只比较峰值算力

$$
t_{\mathrm{compute}}=\frac{33{,}554{,}432}{100\times10^{12}}
\approx0.336\ \mu s
$$
$$
t_{\mathrm{memory}}=\frac{33{,}570{,}816}{10^{12}}
\approx33.57\ \mu s
$$

算力翻倍只能把 0.336 μs 降到 0.168 μs，搬运仍需 33.57 μs，因此理想模型下总时间基本不变，不会快一倍。

### 解答 4：增大 batch 提高权重复用

$$
I(B)=\frac{2B\times4096^2}{2(4096^2+2B\times4096)}
=\frac{4096B}{4096+2B}
$$

$$I(128)\approx120.47>100$$

B=128 时已经越过理想拐点，变为 compute-bound。同一份权重为 128 行输入服务，所以每搬一个权重所完成的计算增加了。

| B | Arithmetic intensity | 理想瓶颈 |
|---:|---:|---|
| 1 | 约 1 FLOP/byte | Memory-bound |
| 128 | 约 120.47 FLOPs/byte | Compute-bound |

**工程判断：**大 batch 可能提高吞吐，但等待请求凑批会增加排队延迟，激活占用也会上升。线上应同时检查 tokens/s、响应延迟及尾延迟、显存和流量条件。这里的结论依赖题目“一次读权重”等理想假设，真实性能需测量。

**最后复述：我能否独立按“形状 → FLOPs → bytes → intensity → 瓶颈 → 优化代价”的顺序完成分析？**
