# author WilliamQiu
import torch.nn as nn
from torch import Tensor
import torch

# easy 实现 
class Rotraypostionembedding(nn.Module):
    def __init__(self, head_dim: int, max_seqlen: int, rotary_dim: int, theta: float = 10000.0):
        super().__init__()
        # 考虑 Attention [B,T,H,D]
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        # 每一个注意力头有 几个向量位置编码
        self.rotary_dim = rotary_dim
        # 基础theta
        self.theta = theta
        # 计算的旋转频率 
        # 2i 意味着我们只需要取一半 
        inv_freq = 1.0 / (theta ** torch.arange(0,rotary_dim,2))
    def forward(self, x: Tensor) -> Tensor:
        return x