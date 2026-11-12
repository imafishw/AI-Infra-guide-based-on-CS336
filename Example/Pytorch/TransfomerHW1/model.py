import torch
import torch.nn as nn
from torch import Tensor

from config import TransformerConfig

# 已经完成 
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # TODO: Initialize embedding weights
        self.weights = nn.Parameter(torch.empty(vocab_size, d_model))
        # 正态分布初始化参数
        nn.init.normal_(self.weights, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:
        # TODO: [B, T] -> [B, T, d_model]
        return self.weights[token_ids]

# RMS normolization
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        x_float = x.float()  # 用 float32 计算，提高数值稳定性

        mean_square = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(mean_square + self.eps)

        return (normalized * self.weight).to(original_dtype)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()

        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k 必须是正偶数")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta 和 max_seq_len 必须为正数")

        self.d_k = d_k
        self.max_seq_len = max_seq_len  # 新增
        # 1. 每两个特征组成一对，给每对分配一个旋转频率
        # 例如 d_k=4：pair_indices = [0, 2]
        pair_indices = torch.arange(0, d_k, 2, dtype=torch.float32)
        frequencies = theta ** (-pair_indices / d_k)  # [d_k/2]

        # 2. 提前计算所有位置的旋转角度
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # [max_seq_len, 1] * [1, d_k/2]
        # -> [max_seq_len, d_k/2]
        angles = positions[:, None] * frequencies[None, :]

        # 3. 保存 sin、cos 表，forward 时直接按位置取出
        # buffer 不参与训练，会随模块一起移动到 GPU
        self.register_buffer(
            "cos_cached", torch.cos(angles), persistent=False
        )
        self.register_buffer(
            "sin_cached", torch.sin(angles), persistent=False
        )

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        if positions.ndim != 1 or positions.shape[0] != x.shape[-2]:
            raise ValueError("positions 必须是 [T]，与输入序列长度一致")

        if positions.dtype != torch.long:
            raise ValueError("positions 必须使用 torch.long")

        if torch.any(positions < 0) or torch.any(positions >= self.max_seq_len):
            raise ValueError("位置超出 RoPE 缓存范围")
        # x: [B, H, T, d_k]
        # positions: [T]，整数位置，例如 [0, 1, 2]
        if x.shape[-1] != self.d_k:
            raise ValueError("输入最后一维必须等于 d_k")

        # 4. 根据位置查表：[T, d_k/2]
        cos = self.cos_cached [positions].to(dtype=x.dtype) # type: ignore
        sin = self.sin_cached[positions].to(dtype=x.dtype) # type: ignore

        # 5. 把每一对中的第一个数、第二个数分别取出来
        # [B, H, T, d_k] -> [B, H, T, d_k/2]
        a = x[..., 0::2]
        b = x[..., 1::2]

        # 6. 二维旋转公式；cos、sin 自动广播到所有样本和头
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos

        # 7. 将每对特征交错拼回
        # -> [B, H, T, d_k/2, 2]
        pairs = torch.stack((rotated_a, rotated_b), dim=-1)

        # -> [B, H, T, d_k]
        return pairs.flatten(start_dim=-2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int ,n_heads: int, theta: float = 10000.0,
    max_seq_len: int = 2048,):
        super().__init__()
        # TODO: Q, K, V and output projections
        self.d_model = d_model
        self.n_heads = n_heads
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")    
        self.head_dim = d_model // n_heads
        self.rope = RotaryPositionalEmbedding(
            theta,
            self.head_dim,
            max_seq_len,
        )
        # 初始化位置编码 
        self.Q_proj = nn.Linear(d_model,d_model,bias = False)
        self.K_proj = nn.Linear(d_model,d_model,bias = False)
        self.V_proj = nn.Linear(d_model,d_model,bias = False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.Q_proj(x)
        k = self.K_proj(x)
        v = self.V_proj(x)
        # 此时q k v 的维度 还是 [Bacthsize Token Dmodel]
        q = q.reshape(B, T, self.n_heads, self.head_dim)
        k = k.reshape(B, T, self.n_heads, self.head_dim)
        v = v.reshape(B, T, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # H 为头数，head_dim 为每头维度；K 转置后为 [B, H, head_dim, T]
        # 新增：为 T 个 token 生成位置编号
        positions = torch.arange(T, device=x.device)  # [T]
        # 新增：按 token 的位置旋转 Q、K，形状不变
        q = self.rope(q, positions)
        k = self.rope(k, positions) 
        scores = q @ k.transpose(-2, -1)  # [B,H,T,head_dim] @ [B,H,head_dim,T] -> [B,H,T,T]
        scores = scores / (self.head_dim ** 0.5)  # 缩放分数，形状仍为 [B,H,T,T]
        future_mask = torch.ones(
            T, T,
            device=x.device,
            dtype=torch.bool,
        ).triu(diagonal=1)  # [T,T] 布尔遮罩，主对角线上方（未来位置）为 True
        # [T,T] 遮罩广播到所有 B、H；填充后仍为 [B,H,T,T]
        scores = scores.masked_fill(future_mask, float("-inf"))
        # 沿最后的 T（被关注位置）归一化；类型转换不改变形状
        weights = torch.softmax(
                    scores.float(), dim=-1
                ).to(v.dtype)  # weights: [B,H,T,T]
        out = weights @ v                 # [B,H,T,T] @ [B,H,T,head_dim]
                                  # -> [B,H,T,head_dim]
        out = out.transpose(1, 2)          # -> [B,T,H,head_dim]
        '''
        token 0
            head 0 [1, 2]
            head 1 [10, 20]
        token 1
            head 0[3, 4]
            head 1[30, 40]
        token 2
            head 0[5, 6]
            head 1[50, 60]
        '''
        # TOKEN 0 [1，2，10，20] 最终变回 B T D
        out = out.reshape(B, T, D)         # -> [B,T,D]，拼接各头
        return self.out_proj(out)         # -> [B,T,D]，混合各头特征
        

# FFNSwiGLU
class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff: int):
        super().__init__()
          # 两条分支：分别生成门控信号和待调节的特征
        self.gate_proj = nn.Linear(d_model,d_ff,bias = False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
          # 将隐藏特征投影回 d_model，便于残差相加
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        # x: [B,T,d_model]
        gate = self.gate_proj(x) # B T d_ff
        up = self.up_proj(x)
        # SiLU(z) = z * sigmoid(z)
        gate = gate * torch.sigmoid(gate)
        hidden = gate*up # 逐个元素门控 [B,T,d_ff]
        return self.down_proj(hidden)   # [B, T, d_model]


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
         # 对每个 token 的 d_model 个特征做归一化
        self.attn_norm = RMSNorm(config.d_model)

        # 接收和输出的形状都是 [B, T, d_model]
        self.attn = MultiHeadSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            theta=config.theta,
            max_seq_len=config.max_seq_len,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
        )
    def forward(self, x: Tensor) -> Tensor:
        attention_output = self.attn(self.attn_norm(x))
        x = x + attention_output  # 残差连接 resnet
        feedforward_output = self.ffn(self.ffn_norm(x))
        x = x + feedforward_output  # 残差连接 resnet
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len

        self.embedding = Embedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )
         # 每次循环都创建一个新 Block，各层拥有独立参数。
        # ModuleList 让 PyTorch 正确管理这些层的参数。
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        # [B, T, d_model] -> [B, T, vocab_size]
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids 必须是 [B, T]")
        
        if token_ids.shape[1] > self.max_seq_len:
            raise ValueError("输入序列长度超过 max_seq_len")

        x = self.embedding(token_ids)   # [B, T, d_model]

        for layer in self.layers:
            x = layer(x)               # [B, T, d_model]

        x = self.final_norm(x)          # [B, T, d_model]
        logits = self.lm_head(x)        # [B, T, vocab_size]

        return logits

if __name__ == "__main__":
    torch.manual_seed(42)

    config = TransformerConfig(
        vocab_size=1024,
        d_model=256,
        n_heads=8,
        d_ff=704,
        n_layers=2,
        max_seq_len=128,
    )
    device = torch.device(config.device)
    model = TransformerLM(config).to(device)

    # 每条序列 17 个 token，用前 16 个预测后 16 个
    tokens = torch.randint(0, config.vocab_size, (2, 17), device=device)
    inputs = tokens[:, :-1]    # [2, 16]
    targets = tokens[:, 1:]    # [2, 16]

    logits = model(inputs)    # [2, 16, 1024]

    loss = nn.functional.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )

    loss.backward()

    print("logits shape:", logits.shape)
    print("loss:", loss.item())
    print("反向传播完成")
