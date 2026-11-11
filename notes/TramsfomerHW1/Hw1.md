# Transformer HW1：从分词到模型组件

本文按实现顺序组织：先准备环境与数据，再训练和加载 tokenizer，最后理解 Embedding、RMSNorm、多头自注意力及其矩阵运算。

## 1. 环境准备

### 1.1 安装项目依赖

准备 Python 3.12 或更高版本以及 uv，在terminal中执行：

```
cd AI-Infra-guide-based-on-CS336
uv sync
```

后续使用 `uv run python` 运行脚本，以使用项目环境。

### 1.2 配置 minbpe 导入路径

本项目的包位于 `src/minbpe`，将其父目录 `src` 加入 Python 模块搜索路径：

macOS / Linux（bash、zsh）：

```bash
cd AI-Infra-guide-based-on-CS336
export PYTHONPATH="$PWD/src"
```

Windows PowerShell：

```powershell
cd AI-Infra-guide-based-on-CS336
$env:PYTHONPATH = "$PWD\src"
```

Windows CMD：

```bat
cd AI-Infra-guide-based-on-CS336
set PYTHONPATH=%CD%\src
```

`$PWD`（或 `%CD%`）是当前目录，所以应先进入项目根目录。设置仅对当前终端及其启动的进程有效，新开终端需要重新执行。无需复制 minbpe 或另装同名包。

验证导入（各平台通用）：

```bash
uv run python -c "from minbpe import RegexTokenizer; print(RegexTokenizer)"
```

脚本中直接使用：

```python
from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
```

Notebook 内核也需要能找到 `src`；在另一个终端设置环境变量不会改变已经运行的内核环境。

## 2. 数据与分词

### 2.1 准备 TinyStories

约定将训练集放在 `data/TinyStories-train.txt`，验证集放在 `data/TinyStories-valid.txt`。这些数据需要另行准备。生成的 tokenizer 文件保存在 `models/`，保存前先创建目录。

只在训练集上学习 BPE，验证集使用同一份训练好的 tokenizer 编码。先用少量文本和较小词表跑通；当前 minbpe 每次合并都会遍历文本块，全量训练较慢且占内存。

现有 `src/testexamples/trainTokenizer.py` 使用另一份示例语料，需先修改数据路径和训练配置，再从项目根目录运行：

```bash
uv run python src/testexamples/trainTokenizer.py
```

### 2.2 训练与保存 BPE

- 调用 `RegexTokenizer.train(text, vocab_size=...)` 学习合并规则。
- 若语料以 `<|endoftext|>` 分隔故事，训练前将该标记替换为换行，避免学习标记本身；训练后用 `register_special_tokens()` 分配独立 ID。
- 例如总词表为 1024 时，训练 1023 个普通 token，再将 EOS 注册为 1023。
- 调用 `tokenizer.save("models/tinystories")`，生成 `.model` 和 `.vocab`；前者用于加载，后者用于人工查看。
- 文本含注册过的 EOS 时，用 `encode(text, allowed_special={"<|endoftext|>"})` 编码。
- 语料太少、词表目标太大时，当前训练实现可能因无可合并 pair 而报错，需要减小目标、增加语料或添加提前终止判断。

### 2.3 加载并接入 LM

当前父类 `load()` 不会重新编译正则，加载后同步派生状态：

```python
import regex
from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.load("models/tinystories.model")
tokenizer.compiled_pattern = regex.compile(tokenizer.pattern)
tokenizer.register_special_tokens(tokenizer.special_tokens)

all_ids = set(tokenizer.vocab) | set(tokenizer.special_tokens.values())
vocab_size = max(all_ids) + 1
```

完整流程：

```text
文本 → tokenizer.encode() → token ID → torch.long 张量
     → nn.Embedding → Transformer → 输出词表 logits
生成的 ID → 同一个 tokenizer.decode() → 文本
```

Embedding 和 LM 输出层均使用包含特殊 token 的 `vocab_size`。训练时输入与目标错开一位，让模型预测下一个 token。BPE 规则固定，不参与反向传播。

tokenizer 的 `.model` 与 LM checkpoint 是不同文件，必须配套保存和加载；即使词表大小相同，不同 tokenizer 的 ID 含义也可能不同。

## 3. 模型组件与数学原理

对应代码：`Example/Pytorch/TransfomerHW1/model.py`。当前 Embedding 与 RMSNorm 已实现，多头注意力正在实现中；下文结合数学公式给出组件说明与参考实现。

### 3.1 统一符号与数据流

| 符号 | 含义 |
|---|---|
| $B$ | 批量大小，即一次处理的样本数量 |
| $T$ | 每个样本的 token 数量 |
| $V$ | 词表大小，包含特殊 token |
| $D$ | 隐藏维度，对应 `d_model` |

```text
token ID [B, T]
  → Embedding [B, T, D]
  → TransformerBlock × 层数 [B, T, D]
  → 最后的 RMSNorm [B, T, D]
  → 词表投影 [B, T, V]
```

Embedding 将离散 ID 转成可学习的向量；RMSNorm 调整向量的尺度，不改变形状。

### 3.2 Embedding：从 token ID 到向量

#### 参数与数学公式

Embedding 维护一个可学习的矩阵：

$$
E \in \mathbb{R}^{V \times D}
$$

每一行对应一个 token，每一列对应一个特征。对于输入 ID 矩阵 $I$，输出为：

$$
X_{b,t,:} = E_{I_{b,t},:},
\qquad X \in \mathbb{R}^{B \times T \times D}
$$

冒号表示取出这一行的全部特征。ID 只是索引，不表示大小关系或语义距离；例如 ID 8 和 ID 9 不一定语义接近。

#### 当前实现

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weights, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weights[token_ids]
```

- `torch.empty(V, D)` 分配空间，但没有初始化有效的参数值，因此随后必须初始化。
- `nn.Parameter` 将矩阵注册为模型的可学习参数，优化器可以更新它。
- `normal_` 原地进行正态分布初始化。这里每个参数服从 $\mathcal{N}(0, 0.02^2)$，其中 `0.02` 是标准差，不是方差。这是当前代码的选择，并不是 Embedding 必须采用的数值。
- `self.weights[token_ids]` 根据 ID 取行；ID 通常使用 `torch.long`，有效范围是 $0 \leq I_{b,t} < V$。

例如输入 `[[8, 12, 8]]` 的形状是 `[1, 3]`，输出形状就是 `[1, 3, D]`。两个 ID 8 取出同一个向量；位置差异由后续的位置机制处理。

查表操作也能参与反向传播：梯度更新被选中的权重行，同一个 ID 多次出现时，对应梯度会累积。整数 ID 本身不参与梯度更新。

### 3.3 RMSNorm：按 token 调整特征尺度

#### 数学公式

对某个 token 的特征向量 $x=(x_1,\ldots,x_D)$，均方根（Root Mean Square）为：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2}
$$

加入数值稳定项和可学习的缩放参数后，RMSNorm 为：

$$
y_i = \gamma_i\frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}}
$$

| 项 | 含义 |
|---|---|
| $\frac{1}{D}\sum_j x_j^2$ | 一个 token 的所有特征的平方平均值 |
| $\epsilon$ | 很小的正数，防止分母为零；这里采用 $10^{-5}$ |
| $\gamma_i$ | 第 $i$ 个特征的可学习缩放参数，初始化为 1 |

每个 token 独立计算自己的均方根，不跨样本或 token 求平均；所有 token 共享同一组 $D$ 个缩放参数。

#### 参考实现

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        x_float = x.float()

        mean_square = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(mean_square + self.eps)

        return (normalized * self.weight).to(original_dtype)
```

#### 代码与公式如何对应

1. `original_dtype` 保存输入类型；`x.float()` 使用 float32 计算平方和均值，降低低精度计算的数值风险，不会切断梯度。
2. `square()` 对每个元素平方；`mean(dim=-1)` 沿最后的特征维度 $D$ 求平均。`keepdim=True` 保留该维度，因此 `[B, T, D]` 变为 `[B, T, 1]`。
3. `torch.rsqrt(a)` 表示 $1/\sqrt{a}$，所以乘以它等价于除以均方根分母。
4. `self.weight` 对应 $\gamma$，形状为 `[D]`；它允许模型在归一化后学习各个特征的合适尺度。
5. `.to(original_dtype)` 恢复输入的数据类型，最终输出形状仍为 `[B, T, D]`。

例如 $x=[3,4]$，暂时忽略 $\epsilon$，并令 $\gamma=[1,1]$：

$$
\operatorname{RMS}(x)=\sqrt{(3^2+4^2)/2}=\sqrt{12.5}
$$

$$
y=[3/\sqrt{12.5},\;4/\sqrt{12.5}]\approx[0.849,\;1.131]
$$

#### 容易混淆的地方

- RMSNorm 不减均值，因此输出不保证均值为 0；它与 LayerNorm 的计算不同。
- RMSNorm 不是将向量的欧氏长度变为 1。忽略 $\epsilon$ 且 $\gamma$ 全为 1 时，输出的均方根为 1，而欧氏长度为 $\sqrt{D}$。
- 乘上训练后的 $\gamma$ 后，输出的均方根也不必为 1。

### 3.4 在 Transformer block 中如何使用

采用 Pre-Norm 结构时，Attention 和 FFN 前各放一个独立的 RMSNorm：

$$
h=x+\operatorname{Attention}(\operatorname{RMSNorm}_1(x))
$$

$$
y=h+\operatorname{SwiGLU}(\operatorname{RMSNorm}_2(h))
$$

两处加法是残差连接。两个 RMSNorm 的缩放参数独立学习；完整语言模型通常还在最后一个 block 后、词表投影前使用一次 RMSNorm。这里说明的是当前骨架所对应的现代 decoder-only 结构，不是所有 Transformer 都采用同一种归一化配置。

### 3.5 多头自注意力：让 token 交换信息

本节使用直接传入参数的接口，不依赖 TransformerConfig。先理解不含 RoPE 的因果自注意力，再接入位置编码。以下是教学参考实现，不代表 model.py 中已经全部完成。

#### 参数、符号与目标

| 符号或参数 | 含义 | 示例 |
|---|---|---|
| $B$ | 样本数量，从输入获取 | 2 |
| $T$ | 一个样本的序列长度，即 token 数量 | 10 |
| $D$ / d_model | 每个 token 的特征维度 | 256 |
| $H$ / n_heads | 注意力头的数量 | 8 |
| $d_h$ / head_dim | 每个头的特征维度，$D/H$ | 32 |
| $t$ | 某个 token 的位置索引，$0\le t<T$ | 3 表示第 4 个 token |

大写 $T$ 是数量，小写 $t$ 是索引；$T$ 不是特征维度。构造函数接收 d_model 和 n_heads；forward 接收输入 x，并从其形状获取 B、T、D。

输入和输出均为 [B, T, D]。注意力的目标是让每个位置根据匹配程度，汇总其他可见位置的信息。

#### 第一步：定义头数与每头维度

下面的代码片段按顺序组成一个类；第十步之后也提供完整实现。

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model 和 n_heads 必须为正数")
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
```

super().__init__() 初始化 nn.Module 的参数管理机制。每个头分配的维度为：

$$
d_h=D/H,\qquad D=H d_h
$$

这里必须用整数除法 //，因为 reshape 的维度必须是整数。256 维拆为 8 个头后，每头为 32 维，总维度没有增加。

#### 第二步：定义 Q、K、V 与输出投影

在构造函数内继续定义：

```python
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
```

Q 是查询表示，K 是供查询匹配的键，V 是实际被汇总的内容。三者来自同一个输入，因此叫自注意力；使用不同参数，让模型分别学习如何查询、如何被匹配以及传递什么信息。

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

这里数学上采用行向量右乘矩阵，$W_Q,W_K,W_V\in\mathbb{R}^{D\times D}$。PyTorch 的 Linear 保存 [out_features, in_features] 权重，实际计算 x @ weight.T；只是权重记号方向不同。bias=False 表示不加偏置。类名为 nn.Linear，注意大写 L。

输出投影用于混合各头的结果：

$$
Y=\operatorname{Concat}(O_1,\ldots,O_H)W_O
$$

#### 第三步：输入投影

```python
    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
```

三个输出都是 [B, T, D]。Linear 作用于最后一个特征维度，此时每个 token 只变换自己的特征，还没有与其他 token 交换信息。D 必须与构造时的 d_model 相同。

#### 第四步：拆头，再按头组织 token

```python
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
```

分开看是两步：

```text
[B, T, D] → [B, T, H, head_dim] → [B, H, T, head_dim]
```

reshape 将投影后的 D 维分成 H 份。每个头仍能使用完整输入的 D 个特征，因为先做了投影：

$$
W_Q=[W_Q^{(1)}|\cdots|W_Q^{(H)}],\qquad
W_Q^{(h)}\in\mathbb{R}^{D\times d_h},\qquad Q_h=XW_Q^{(h)}
$$

这等价于各头分别投影，而不是让各头只能看原始输入的一小部分维度。

**为什么交换 T 和 H？** 转换前按 token 组织所有头；转换后按头组织所有 token。注意力需要每个头独立比较不同 token，而矩阵乘法作用于最后两个维度，所以要让最后两维成为 [T, head_dim]。

例如 B=1、T=3、H=2、head_dim=2，忽略 batch：

```text
转换前 [T, H, head_dim]：
token 0：head 0 [1, 2]， head 1 [3, 4]
token 1：head 0 [5, 6]， head 1 [7, 8]
token 2：head 0 [9, 10]，head 1 [11, 12]

转换后 [H, T, head_dim]：
head 0：token 0 [1, 2]，token 1 [5, 6]，token 2 [9, 10]
head 1：token 0 [3, 4]，token 1 [7, 8]，token 2 [11, 12]
```

数值归属没有变化。设转换前为 Q，转换后为 Q'：

$$
Q'[b,h,t,r]=Q[b,t,h,r]
$$

transpose 通常只改变张量的步幅和访问方式，不立即复制数据。它与 reshape 的用途不同，不能直接 reshape 成 [B,H,T,head_dim] 来代替这次交换。

#### 第五步：计算匹配分数

```python
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
```

每个头计算：

$$
S_h=\frac{Q_hK_h^\top}{\sqrt{d_h}},\qquad
S_{b,h,i,j}=\frac{1}{\sqrt{d_h}}\sum_{r=0}^{d_h-1}Q_{b,h,i,r}K_{b,h,j,r}
$$

行 i 是发起查询的位置，列 j 是被关注的位置。输出为 [B,H,T,T]。负数维度索引从右向左数，-1 是 head_dim，-2 是 T，因此 transpose(-2,-1) 只转置每个头内部的矩阵。

在分量独立、均值为 0、方差为 1 的简化假设下，点积方差约为 $d_h$，除以 $\sqrt{d_h}$ 后方差约为 1。这有助于避免分数尺度过大导致 softmax 过度集中。高维乘法的具体算例见 3.6。

#### 第六步：因果遮罩，阻止关注未来

```python
        future_mask = torch.ones(
            T, T, device=x.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(future_mask, float("-inf"))
```

自回归语言模型在位置 i 只能看 j≤i 的位置。triu(diagonal=1) 标记主对角线上方，也就是未来位置：

```text
T=4，True 表示屏蔽：
False True  True  True
False False True  True
False False False True
False False False False
```

数学上等价于加上遮罩：

$$
\widetilde S_{i,j}=S_{i,j}+M_{i,j},\qquad
M_{i,j}=\begin{cases}0&j\le i\\-\infty&j>i\end{cases}
$$

[T,T] 遮罩广播到所有 batch 和头；device=x.device 保证设备一致。本例只处理因果遮罩，没有另外处理 padding。

#### 第七步：softmax 将分数变为权重

```python
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
```

$$
A_{i,j}=\frac{\exp(\widetilde S_{i,j})}{\sum_{m=0}^{T-1}\exp(\widetilde S_{i,m})},
\qquad \sum_j A_{i,j}=1
$$

最后一维是被关注的位置，所以 dim=-1 表示每个查询位置在所有可见位置之间分配权重。未来位置满足 exp(-∞)=0，权重为 0。形状仍是 [B,H,T,T]。

float() 用 float32 计算 softmax，然后恢复为 V 的类型；这提高 softmax 本身的数值稳定性，但不能修复此前点积已经出现的溢出。

#### 第八步：加权汇总 V

```python
        out = weights @ v
```

$$
O_h=A_hV_h,\qquad o_{h,i}=\sum_{j=0}^{T-1}A_{h,i,j}v_{h,j}
$$

形状为 [B,H,T,T] @ [B,H,T,head_dim] → [B,H,T,head_dim]。Q、K 决定关注谁，V 决定取回什么；这一步才真正融合不同 token 的信息。

#### 第九步：合头

```python
        out = out.transpose(1, 2).reshape(B, T, D)
```

先从 [B,H,T,head_dim] 换回 [B,T,H,head_dim]，再把同一个 token 的各头特征拼接成 D 维：

$$
z_i=[o_{1,i}|\cdots|o_{H,i}]
$$

不能省略 transpose，否则会把头和 token 的排列混淆。转置后内存可能不连续，reshape 会在必要时复制以得到正确布局。

#### 第十步：输出投影

```python
        return self.out_proj(out)
```

$$
Y=ZW_O\in\mathbb{R}^{B\times T\times D}
$$

输出投影学习如何混合不同头的信息。残差连接由外层 TransformerBlock 完成，见 3.4。

#### 完整基础参考实现

```python
import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model 和 n_heads 必须为正数")
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 后续在这里接入 RoPE，对 q、k 加入位置信息。
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        future_mask = torch.ones(
            T, T, device=x.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(future_mask, float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = weights @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


attention = MultiHeadSelfAttention(d_model=256, n_heads=8)
x = torch.randn(2, 10, 256)
y = attention(x)
print(y.shape)  # torch.Size([2, 10, 256])
```

整体形状链：

```text
输入             [B, T, D]
Q/K/V 投影       [B, T, D]
拆头             [B, T, H, head_dim]
按头组织         [B, H, T, head_dim]
QK 分数          [B, H, T, T]
mask + softmax   [B, H, T, T]
加权汇总 V       [B, H, T, head_dim]
交换维度并合头   [B, T, D]
输出投影         [B, T, D]
```

### 3.6 矩阵乘法详解：从二维到四维

#### 二维：一行与一列相乘求和

```text
[m, n] @ [n, p] → [m, p]
```

中间的 n 是被求和的维度，结果保留 m、p。@ 是矩阵乘法，* 是逐元素相乘。

$$
C_{i,j}=\sum_r A_{i,r}B_{r,j}
$$

例如：

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\quad
B=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
AB=\begin{bmatrix}4&5\\10&11\end{bmatrix}
$$

左上角为 $1\times1+2\times0+3\times1=4$；右下角为 $4\times0+5\times1+6\times1=11$。

#### QK 转置：一次得到所有 token 对的分数

只看一个样本、一个头，令 T=3、head_dim=2：

$$
Q=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
K=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}
$$

Q、K 的每一行是一个 token 的向量。想让 Q 的每一行与 K 的每一行做点积，就将 K 的行转成列：

$$
K^\top=\begin{bmatrix}1&3&5\\2&4&6\end{bmatrix},\qquad
QK^\top=\begin{bmatrix}1&3&5\\2&4&6\\3&7&11\end{bmatrix}
$$

例如索引 [2,1] 的分数是 $[1,1]\cdot[3,4]=7$。输出 [T,T] 的行对应查询 token，列对应被查询 token；这里为说明乘法，省略缩放和遮罩。

#### 三维：每个样本分别计算

若 q、k 都是 [B,T,head_dim]，则：

```python
scores = q @ k.transpose(-2, -1)
# 等价于对每个 b 分别计算：
# scores[b] = q[b] @ k[b].T
```

每组 [T,head_dim] @ [head_dim,T] 得到 [T,T]，B 组结果组成 [B,T,T]。不同样本之间不会互相匹配。

#### 四维：每个样本中的每个头分别计算

对注意力中的四维张量，最后两维是矩阵，前两维是分组：

```text
[B, H | T, head_dim] @ [B, H | head_dim, T]
 分组      矩阵          分组      矩阵
                        ↓
                 [B, H | T, T]
```

当 B=2、H=8、T=10、head_dim=32 时，相当于同时计算 16 组 [10,32] @ [32,10]。每组得到 [10,10]，总结果是 [2,8,10,10]。

```python
# 说明等价关系的写法；实际实现直接用 @，不需要循环。
scores = torch.empty(B, H, T, T, device=q.device, dtype=q.dtype)
for b in range(B):
    for h in range(H):
        scores[b, h] = q[b, h] @ k[b, h].T
```

这里 q[b,h] 和 k[b,h] 已是二维矩阵，才能直接用 .T。四维张量应使用 transpose(-2,-1)，不能用 .T 代替。

一般而言，torch.matmul（@）会对最后两维做矩阵乘法，前面的维度按广播规则匹配。本例前导维度完全相同，因此就是相同 b、h 的矩阵一一配对，没有跨样本或跨头的求和。

如果没有交换 T 和 H，就直接计算相同形式的乘法：

```text
[B,T,H,head_dim] @ [B,T,head_dim,H] → [B,T,H,H]
```

此时比较的是每个 token 内不同头的向量，得到 H×H 分数表，而我们需要的是每个头内部的 T×T token 分数表。这就是拆头后需要 transpose(1,2) 的原因。

#### weights @ V：在 token 维度上加权求和

假设一个查询位置的权重是 [0.2,0.3,0.5]，V 为：

$$
V=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}
$$

则该位置输出：

$$
[0.2,0.3,0.5]V
=[0.2\times1+0.3\times3+0.5\times5,\;
0.2\times2+0.3\times4+0.5\times6]
=[3.6,4.6]
$$

也就是 $0.2v_0+0.3v_1+0.5v_2$。这个例子可对应最后一个位置，它可以看到全部三个 token。

| 运算 | 形状（单个头） | 沿哪个维度求和 | 作用 |
|---|---|---|---|
| $QK^\top$ | [T,head_dim] @ [head_dim,T] → [T,T] | 特征维度 head_dim | 计算 token 间的匹配分数 |
| $AV$ | [T,T] @ [T,head_dim] → [T,head_dim] | token 维度 T | 按权重汇总 token 的内容 |

理解高维乘法时，先固定 b、h，只看一张二维矩阵；理解这一组后，再把结果放回所有组。

### 3.7 RoPE：用旋转将 token 的位置融入特征

#### 先明确：位置是 token 级的，旋转在特征对上完成

一个 token 只有一个位置 p。拆成多个头不会改变这个位置。RoPE 把同一个位置作用到该 token 在所有头中的特征上。

```text
一个 token，位置 p=2
├── head 0：第 0 对用角度 2×ω₀，第 1 对用角度 2×ω₁
└── head 1：第 0 对用角度 2×ω₀，第 1 对用角度 2×ω₁
```

这里不是给每对特征分配一个新位置：各对共享 token 的位置，只是频率不同。标准 RoPE 中，所有 token、所有头共享同一套频率规则。

| 哪个量 | 由什么决定 |
|---|---|
| 位置 p | token 在序列中的位置，通常从 0 开始 |
| 频率 ωᵣ | 每个头内部第 r 对特征的编号 |
| 角度 φₚ,ᵣ | 位置 × 频率 |
| 旋转后的值 | 原始特征、位置和频率共同决定 |

#### 直觉：让位置控制指针转动

把两个特征 [a,b] 看成一根箭头尖端的横、纵坐标。规定位置每增加 1，箭头就多转一个固定角度。于是位置改变，向量方向也随之改变。

例如仅为直观演示，规定每个位置转 30°：位置 2 转 60°，位置 5 转 150°，角度差为 90°。将两者同时后移到位置 3 和 6，各自转 90°、180°，角度差仍是 90°。共同移动不改变相对距离，也不改变由位置引入的角度差。

这就是旋转适合表达相对位置的原因。实际计算使用弧度，而非这里举例的角度。

旋转后的向量不是附带一个能直接读出的“位置数字”。原始内容也有自己的方向；最终方向同时受到内容和位置影响。后续点积可以感受到这种变化，不需要先解码位置。零向量无论怎样旋转仍是零向量，因此不能只看任意一个输出就唯一恢复位置。

#### 为什么配对，为什么使用多个频率？

一维实数只有数轴上的位置，没有二维平面的方向。将两个数配成 [a,b]，才能连续旋转方向并保持长度。配对并非因为两个特征必然有某种语义关系，而是为了提供二维旋转坐标。

只给一个数乘 cos 并不等价：当 cos 为 0 时，所有数都变为 0，信息丢失。二维旋转会在两个坐标之间重新分配信息，已知角度后可以逆向恢复。

同频率也可以编码位置，但所有特征会以同一种速度响应距离。不同频率提供快慢不同的位置变化：快频率对小距离变化明显，慢频率在更大范围内缓慢变化。类似同时观察秒针和分针，组合能减少单一周期带来的混淆。

例如每步转 90°时，位置 0 和 4 的旋转相同；再增加每步转 10°的一对特征，位置 4 在这一对上转了 40°，就与位置 0 不同。这个例子说明多频率的作用，不意味着实际浮点实现可以唯一表示无限位置，也不意味着距离越远点积必然越小。

#### 数学公式：先频率，再角度，最后旋转

对每头维度 d_k，采用相邻特征配对：

```text
[x₀,x₁,x₂,x₃,...] → [x₀,x₁]、[x₂,x₃]、...
```

第 r 对的频率、位置 p 对应的角度为：

$$
\omega_r=\theta^{-2r/d_k},\qquad
\phi_{p,r}=p\omega_r,\qquad r=0,\ldots,d_k/2-1
$$

**这里的 $d_k$ 是拆头后每个头的特征维度，不是总维度，也不是头数。** 在本项目的实现中：

```python
d_k = head_dim = d_model // n_heads
```

例如 `d_model=256`、`n_heads=8`，则每个头有 `d_k=32` 个特征，组成 `32 // 2 = 16` 对。公式中的特征对编号 $r$ 从 0 到 15，分母代入 **32**，不能代入总维度 256 或头数 8。不同头都按各自内部的这 16 对应用同一套频率规则。

例如 θ=10000、d_k=4：

$$
\omega_0=10000^0=1,\qquad
\omega_1=10000^{-2/4}=\frac{1}{\sqrt{10000}}=0.01
$$

位置 p=2 时，两对分别旋转 2、0.02 弧度。

对于一对特征 [a,b]，逆时针旋转 φ：

$$
a'=a\cos\phi-b\sin\phi,\qquad
b'=a\sin\phi+b\cos\phi
$$

也可以写成列向量矩阵乘法：

$$
\begin{bmatrix}a'\\b'\end{bmatrix}
=R(\phi)\begin{bmatrix}a\\b\end{bmatrix},\qquad
R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}
$$

旋转保持长度：$a'^2+b'^2=a^2+b^2$。对任意两组列向量 u、v，在同一频率下：

$$
(R(i\omega)u)^\top(R(j\omega)v)
=u^\top R((j-i)\omega)v
$$

这说明点积中的位置影响通过相对位置 j−i 表现出来，内容 u、v 仍参与计算。

#### 参数不要混淆

| 参数 | 含义 | 示例 |
|---|---|---|
| d_model | 一个 token 的总特征维度 | 256 |
| n_heads | 头数 | 8 |
| d_k / head_dim | 每个头的维度，d_model // n_heads | 32 |
| T | 当前输入的 token 数 | 10 |
| max_seq_len | 本实现预计算的位置数量 | 2048 |
| theta | 控制频率分布的基数 | 10000 |

max_seq_len 不是特征维度，也不要求每次都输入这么多 token。它只是本实现的缓存容量；超出后需要扩展缓存或另行处理，增大缓存也不自动保证模型的长上下文能力。

上例中，Q 为 [B,8,10,32]，完整 cos/sin 缓存各为 [2048,16]，当前按位置取出的 cos/sin 各为 [10,16]。旋转后的 Q 仍为 [B,8,10,32]。

### 3.8 RoPE 代码逐段解释

以下实现约定 x 为 [B,H,T,d_k]，positions 为所有样本共享的 [T] 整数位置。它采用相邻特征配对，不包含 KV cache。

#### 第一步：初始化频率与缓存

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k 必须是正偶数")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta 和 max_seq_len 必须为正数")
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_dims = torch.arange(0, d_k, 2, dtype=torch.float32)
        frequencies = theta ** (-pair_dims / d_k)  # [d_k/2]
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]
        # angles: [max_seq_len,d_k/2]
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)
```

- arange(0,d_k,2) 得到 0、2、4……，已经对应公式中的 2r，不要再次乘 2。
- positions[:,None] 从 [max_seq_len] 变为 [max_seq_len,1]；frequencies[None,:] 变为 [1,d_k/2]。
- 两者逐元素广播相乘，不是矩阵乘法，得到每个位置、每对特征的角度。
- sin、cos 使用弧度。先用 float32 计算这些表。
- register_buffer 保存无需训练的数据，模块移动设备时缓存跟随移动。persistent=False 表示不写入 state_dict，不是说数据不存在。

#### 第二步：根据 token 位置查表

下面继续写在同一个类内：

```python
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError("输入最后一维必须等于 d_k")
        if positions.ndim != 1 or positions.shape[0] != x.shape[-2]:
            raise ValueError("positions 必须为 [T]")
        if positions.dtype != torch.long:
            raise ValueError("positions 必须使用 torch.long")
        if torch.any(positions < 0) or torch.any(positions >= self.max_seq_len):
            raise ValueError("位置超出缓存范围")

        cos = self.cos_cached[positions].to(x.dtype)  # [T,d_k/2]
        sin = self.sin_cached[positions].to(x.dtype)
```

Attention 调用 rope(q,positions) 时，RoPE 内的参数 x 就是传入的 Q，名字不同但数据含义相同。

positions 是 token 的位置，不是 token ID。比如 ID 为 [153,27]，位置仍可为 [0,1]。使用 torch.arange(T,device=x.device) 是因为有 T 个 token 需要编号，不是给头或特征编号。

位置索引应非负且小于缓存长度，缓存、输入、位置保持同一设备。教学实现中的范围检查会在 GPU 上产生同步开销，性能优化时可统一在调用边界检查。

#### 第三步：分出每对特征的两个坐标，再旋转

```python
        a = x[..., 0::2]  # [B,H,T,d_k/2]
        b = x[..., 1::2]  # [B,H,T,d_k/2]
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos
```

省略号表示保留前面全部维度。0::2 取下标 0、2、4……，1::2 取 1、3、5……。

```text
原始：[x₀,x₁,x₂,x₃]
a：  [x₀,x₂]
b：  [x₁,x₃]
```

cos/sin 的 [T,d_k/2] 广播到所有 batch 和头。所有头使用同一套规则，但因为原始特征不同，旋转后的值仍不同。

#### 第四步：stack 配对，flatten 恢复特征维度

```python
        pairs = torch.stack((rotated_a, rotated_b), dim=-1)
        # [B,H,T,d_k/2,2]
        return pairs.flatten(-2)  # [B,H,T,d_k]
```

stack 在末尾新增一个维度，把对应元素配成一对。flatten(-2) 从倒数第二维开始合并到最后一维，不动 B、H、T。

```text
rotated_a = [10,30]
rotated_b = [20,40]
       ↓ stack(dim=-1)
[[10,20],
 [30,40]]
       ↓ flatten(-2)
[10,20,30,40]
```

不能直接 cat 两组，那会得到 [10,30,20,40]，顺序不同。旋转前后的形状完全相同。

### 3.9 两个头、两个 token：完整数字流转

#### 示例 A：d_model=4，每头只有一对特征

这里 token1 指第一个 token，位置为 0；token2 指第二个 token，位置为 1。以下数值已经是 Q 投影后的特征。

```text
Q 投影结果 [B,T,D] = [1,2,4]：
token1：[1,2,3,4]
token2：[5,6,7,8]

拆头后：
                  head 0     head 1
token1，位置0      [1,2]      [3,4]
token2，位置1      [5,6]      [7,8]
```

每头 d_k=2，因此只有 r=0，频率 θ⁰=1。两个头对应特征对都使用这个频率：

```text
token1 的两个头：角度都是 0×1=0
token2 的两个头：角度都是 1×1=1 弧度
```

旋转后：

| token | head 0 | head 1 |
|---|---|---|
| token1，位置 0 | [1,2] | [3,4] |
| token2，位置 1 | [-2.3473,7.4492] | [-2.9497,10.2127] |

例如 [5,6] 变为 $[5\cos1-6\sin1,\;5\sin1+6\cos1]$。两个头分别计算，不把 head 0 与 head 1 的特征混在一起。

可以单独运行：

```python
import torch

torch.set_printoptions(precision=4, sci_mode=False)
q = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.]]])
q = q.reshape(1, 2, 2, 2).transpose(1, 2)  # [B,H,T,d_k]
# head 0：[[1,2],[5,6]]；head 1：[[3,4],[7,8]]
positions = torch.arange(2)
frequencies = torch.tensor([1.0])
angles = positions[:, None] * frequencies[None, :]  # [[0],[1]]
cos, sin = angles.cos(), angles.sin()
a, b = q[..., 0::2], q[..., 1::2]
out = torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2)
print(out)
# tensor([[[[ 1.0000,  2.0000],
#           [-2.3473,  7.4492]],
#          [[ 3.0000,  4.0000],
#           [-2.9497, 10.2127]]]])
```

这里 H=T=2，transpose 前后的形状数字恰好相同，但维度含义和数值排列不同，不能省略转置。

#### 如何读 q[0,1,0,:]？

四维 Q 的索引顺序是 q[样本,头,token,特征]，从 0 开始，冒号表示全部取出。

| 表达式（上例旋转前） | 含义 | 值 |
|---|---|---|
| q[0,0,0,:] | 样本0、头0、第一个 token 的所有特征 | [1,2] |
| q[0,1,0,:] | 样本0、头1、第一个 token 的所有特征 | [3,4] |
| q[0,0,1,:] | 样本0、头0、第二个 token 的所有特征 | [5,6] |
| q[0,1,1,:] | 样本0、头1、第二个 token 的所有特征 | [7,8] |
| q[0,1,0,0] | 上述向量的第一个特征 | 3 |

这些表达式只是在取数据，不会执行位置编码。真正的旋转由 rope(q,positions) 完成。

#### 示例 B：d_model=8，每头两对特征

总维度增至 8，两个头各 4 维：

```text
token1：[X1,X2,X3,X4, X5,X6,X7,X8]
token2：[Y1,Y2,Y3,Y4, Y5,Y6,Y7,Y8]

head 0：第0对 [Y1,Y2]，第1对 [Y3,Y4]
head 1：第0对 [Y5,Y6]，第1对 [Y7,Y8]
```

每个头内部重新从 r=0 编号。θ=10000 时，两对频率为 [1,0.01]。

| token 位置 | head 0 两对的角度 | head 1 两对的角度 |
|---|---|---|
| 0 | [0,0] | [0,0] |
| 1 | [1,0.01] | [1,0.01] |
| 2（若增加第三个 token） | [2,0.02] | [2,0.02] |

例如位置 1 的向量为 [1,0,2,0,3,0,4,0]：

```text
head 0：[1,0] 转1弧度 → [0.5403,0.8415]
        [2,0] 转0.01弧度 → [1.9999,0.0200]
head 1：[3,0] 转1弧度 → [1.6209,2.5244]
        [4,0] 转0.01弧度 → [3.9998,0.0400]
```

如果是每头 8 维，则每头四对的频率为 [1,0.1,0.01,0.001]。频率由每头维度计算，不能拿总 d_model 替代 d_k。

### 3.10 在 Attention 中接入的位置

3.5 的基础实现未接位置编码，扩展它只需以下步骤。

#### 构造函数增加参数并创建 RoPE

```python
# MultiHeadSelfAttention 的构造函数参数：
def __init__(self, d_model: int, n_heads: int,
             theta: float = 10000.0, max_seq_len: int = 2048):
    # 保留原来的 super、参数检查与 Q/K/V/out 投影初始化。
    ...
    self.rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=self.head_dim,
        max_seq_len=max_seq_len,
    )
```

这段是插入位置示意，不是替换完整构造函数。每头维度应为正偶数。

#### 拆头后、QK 点积前应用

```python
q = q.transpose(1, 2)  # [B,H,T,head_dim]
k = k.transpose(1, 2)
v = v.transpose(1, 2)

positions = torch.arange(T, device=x.device)  # [T]，默认 torch.long
q = self.rope(q, positions)  # 形状不变
k = self.rope(k, positions)  # 形状不变

scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
# 然后继续原来的因果遮罩、softmax、weights @ v、合头和输出投影。
```

V 通常不旋转。RoPE 不替代因果遮罩：位置旋转影响匹配方式，遮罩限制不能关注未来。采用这种结构时，在每层 Attention 的 Q/K 上应用 RoPE，通常不必再给输入加一份绝对位置向量。

不是所有 Transformer 都在注意力内部加入位置：有的在输入上加位置向量，有的在分数上加相对位置偏置。这里采用的是旋转 Q/K 的方式。

### 3.11 RoPE 易混点与自查

| 疑问 | 正确理解 |
|---|---|
| 是 q + position 吗？ | 不是。位置控制角度，旋转已有特征；不额外拼一个位置数字。 |
| 位置编码不是 token 级的吗？ | 是 token 级的位置，特征对旋转是把位置融入向量的实现方法。 |
| 同一 token 的不同头频率相同吗？ | 标准实现共享同一套频率，对应特征对的角度相同。 |
| 第一对、第二对指不同 token 吗？ | 指同一头内部的不同特征对，不是不同 token。 |
| 同一头的所有特征都转一样的角度吗？ | 每对共同旋转，不同对频率不同；位置0时所有角度恰好都为0。 |
| max_seq_len 与 d_model 一样吗？ | 前者是位置缓存容量，后者是每 token 特征数。 |
| 缓存必须和 Q 一样大吗？ | 不必。当前 cos/sin 为 [T,d_k/2]，广播到所有样本、头；输出与 Q 同形状。 |
| 为什么按 T 生成 positions？ | 要给 T 个 token 编号；不是给头或特征编号。 |
| 能从旋转后向量直接读出位置吗？ | 一般不能，输出同时含内容和位置作用；后续比较通过旋转关系感知位置。 |

实现检查：

- 使用 nn.Linear，head_dim 用 // 计算；RoPE 的 d_k 为每头维度且为偶数。
- 特征配对规则必须一致。本节相邻配对，不可混用另一种前后半段配对实现。
- angles 用 float32 计算，三角函数接收弧度；低精度缓存会影响数值精度。
- positions 使用整数且不越界，模块与输入在同一设备。
- 带 KV cache 的增量生成要使用实际位置偏移，不能每次都从 0 开始。
- 输出形状应不变；位置 0 的输出应与输入一致；每对特征平方和应近似保持。
- Attention 中修改未来输入不应改变较早位置输出，可用来检查因果遮罩。
- 合头前交换维度；最终应为 weights @ v，再调用 out_proj，而不是 weights @ out_proj。



test task 

```python
attention = MultiHeadSelfAttention(
    d_model=8,
    n_heads=2,
    max_seq_len=16,
)

x = torch.randn(1, 3, 8, requires_grad=True)
out = attention(x)

assert out.shape == x.shape
assert torch.isfinite(out).all()

# 修改未来 token，前两个位置的输出应该不变
changed_x = x.detach().clone()
changed_x[:, 2, :] += 10

changed_out = attention(changed_x)
torch.testing.assert_close(out[:, :2], changed_out[:, :2])

# 检查能否反向传播
out.square().mean().backward()
assert torch.isfinite(x.grad).all()

print("输出形状、因果遮罩和反向传播检查通过")

```
