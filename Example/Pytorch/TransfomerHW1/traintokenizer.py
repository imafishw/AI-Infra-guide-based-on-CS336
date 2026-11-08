# 已经在src文件夹实现好了工业级的tokenizer，直接导入使用即可
from src.minbpe import RegexTokenizer
# 请注意tokenizer的运行需要 export PYTHONPATH="$PWD/src"

from pathlib import Path
from minbpe import RegexTokenizer

TOTAL_VOCAB_SIZE = 1024
EOS = "<|endoftext|>"
EOS_ID = TOTAL_VOCAB_SIZE - 1

# 只读取训练集的一小部分，先跑通流程。
with Path("data/TinyStories-train.txt").open(
    encoding="utf-8"
) as f:
    sample = f.read(100_000)

# 不让 BPE 把 EOS 标记本身学成普通文本片段。
train_text = sample.replace(EOS, "\n\n")

tokenizer = RegexTokenizer()
tokenizer.train(
    train_text,
    vocab_size=TOTAL_VOCAB_SIZE - 1,
    verbose=False,
)

# 特殊 token 单独分配 ID。
tokenizer.register_special_tokens({EOS: EOS_ID})

Path("models").mkdir(exist_ok=True)
tokenizer.save("models/tinystories")

# 验证普通文本和 EOS 都能正确往返。
text = "Once upon a time." + EOS
ids = tokenizer.encode(text, allowed_special={EOS})

assert tokenizer.decode(ids) == text
assert ids[-1] == EOS_ID

# 这个实现的 vocab 不包含 register_special_tokens 注册的 ID，
# 因此必须把两部分一起计算。
all_ids = set(tokenizer.vocab) | set(tokenizer.special_tokens.values())
model_vocab_size = max(all_ids) + 1

print("模型词表大小:", model_vocab_size)
print("编码结果:", ids)