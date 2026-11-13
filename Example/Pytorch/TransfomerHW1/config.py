# 配置模型使用时的相关数据
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# 不需要写构造函数 dataclass的性质
@dataclass
class TransformerConfig:
    vocab_size: int
    # TinyStories 基线：8 层、512 宽度，4096 词表时约 29.9M 参数。
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    # swiglu的隐藏层维度
    d_ff: int = 1408
    max_seq_len: int = 512
    theta: float = 10000.0
    # "cpu" 使用 CPU；"cuda" 使用 GPU，也可指定 "cuda:0"
    device: str = "cuda:0"

    def __post_init__(self):
        for name in ('vocab_size', 'd_model', 'n_heads', 'n_layers', 'd_ff', 'max_seq_len'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} 必须大于零')
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError('d_model 必须整除 n_heads，且 RoPE 每头维度必须为偶数')


@dataclass
class TransformerConfigTrain:
    train_path: str = str(DATA_DIR / "TinyStoriesV2-GPT4-train.txt")
    val_path: str = str(DATA_DIR / "TinyStoriesV2-GPT4-valid.txt")
    tokenizer_path: str = "models/tinystories-4096.model"
    output_dir: str = "checkpoints/tinystories-4096-8x512"
    batch_size: int = 4
    seq_len: int = 512
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
