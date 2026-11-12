# 配置模型使用时的相关数据
from dataclasses import dataclass

# 不需要写构造函数 dataclass的性质
@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    # swiglu的隐藏层维度
    d_ff: int = 704
    max_seq_len: int = 2048
    theta: float = 10000.0
    # "cpu" 使用 CPU；"cuda" 使用 GPU，也可指定 "cuda:0"
    device: str = "cuda:0"


@dataclass
class TransformerConfigTrain:
    train_path: str = "data/TinyStories-train.txt"
    val_path: str = "data/TinyStories-valid.txt"
    tokenizer_path: str = "models/tinystories.model"
    output_dir: str = "checkpoints/tinystories"
    batch_size: int = 4
    seq_len: int = 128
    grad_accum_steps: int = 4
    max_steps: int = 1000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 10
    seed: int = 42
    max_chars: int = 1_000_000  # 教学版限制内存占用
    device: str = "auto"
