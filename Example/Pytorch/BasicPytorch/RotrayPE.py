# author WilliamQiu
import torch.nn as nn
from torch import Tensor
import torch

# easy 实现 
class Rotraypostionembedding(nn.Module):
    def __init__(self, head_dim: int, max_posEmbeding: int, rotary_dim: int, theta: float = 10000.0):
        super().__init__()
        # 考虑 Attention [B,T,H,D]
        self.head_dim = head_dim
        self.max_posEmbeding = max_posEmbeding
        # 每一个注意力头有 几个向量位置编码
        self.rotary_dim = rotary_dim
        self.half = rotary_dim // 2
        # 基础theta
        self.theta = theta
        # 计算的旋转频率 
        # 2i 意味着我们只需要取一半 
        inv_freq = 1.0 / (theta ** torch.arange(0,rotary_dim,2))
        torch_emb = torch.arange(max_posEmbeding, dtype=torch.float)
        # cache 大小有讲究 
        # [
        #   [1,2,3,4,] 位置1 向量的 位置编码
        #   [5,6,7,8]  位置2 向量的 位置编码 假设   
        #          ]   我们一次性把这些存了 
        cache = torch.empty(max_posEmbeding,rotary_dim)
        # max_posEmbeding,rotary 矩阵的 cache[:,:half] 相当于取原矩阵同样位置的一半
        freqs = torch.outer(torch.arange(0,max_posEmbeding,dtype = torch.float),inv_freq)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos(), freqs.sin()), dim=-1),
            persistent=False,
        )
        def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        # 理解到 rotate这里 后续手撕继续刷 
    def forward(self, postions: torch.Tensor, query: Tensor,key: Tensor) -> tuple[Tensor,Tensor]:
        flat = postions.reshape(-1)
        return 


if __name__ == "__main__":
    num = torch.arange(9)
    matrix = num.reshape(3,3)
    print(matrix)
    print(matrix[:,:3])

    print("------------")
    # 测试代码30行的维度
    # t = torch.tensor([0, 1, 2,4,5,6,7,8])          # 3个位置
    # inv_freq = torch.tensor([1.0, 0.5,0.03,0.4])  # 2组频率
    # a = t[:4]
    # print(a)
    # freqs = torch.outer(a, inv_freq)
    # print(freqs)
    # new_t = torch.cat((freqs.cos(), freqs.sin()), dim=-1),
    # print(new_t)
    