import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from minbpe import RegexTokenizer
from config import TransformerConfig
from model import TransformerLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1. 加载训练时使用的 tokenizer
tokenizer = RegexTokenizer()
tokenizer.load(str(ROOT / "models/tinystories-4096.model"))

# 2. 读取 checkpoint，在 CPU 上加载，避免直接占用 GPU 显存
checkpoint = torch.load(
    ROOT / "checkpoints/AI-Infra-guide-based-on-CS336/checkpoints/best.pt",
    map_location="cpu",
    weights_only=True,
)

# 3. 按 checkpoint 的配置重建模型，不能随意改层数或宽度
config_data = dict(checkpoint["model_config"])
config_data["device"] = device
config = TransformerConfig(**config_data)

vocab_ids = set(tokenizer.vocab) | set(tokenizer.special_tokens.values())
assert vocab_ids == set(range(config.vocab_size)), "tokenizer 词表不匹配"

model = TransformerLM(config)
model.load_state_dict(checkpoint["model"], strict=True)
model.to(device)
model.eval()

print("device:", device)
print("step:", checkpoint["step"], "val_loss:", checkpoint["val_loss"])

# 4. 编码提示词
prompt = "Once upon a time, there was a little cat"
ids = tokenizer.encode(prompt)
x = torch.tensor([ids], dtype=torch.long, device=device)
eos_id = tokenizer.special_tokens.get("<|endoftext|>")

# 5. 自回归生成：每轮只新增一个 token
with torch.inference_mode():
    for _ in range(100):
        context = x[:, -config.max_seq_len:]
        logits = model(context)[:, -1, :]  # [1, vocab_size]

        # 入门先用贪心解码，方便排查正确性
        next_id = logits.argmax(dim=-1, keepdim=True)

        if next_id.item() == eos_id:
            break

        x = torch.cat([x, next_id], dim=1)

print(tokenizer.decode(x[0].tolist()))