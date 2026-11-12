# 配置模型使用时的相关数据
from dataclasses import dataclass

# 不需要写构造函数 dataclass的性质
@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8