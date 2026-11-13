"""训练 TinyStories 使用的 RegexTokenizer。

可以在任意目录运行；默认读取云服务器上的 TinyStories 数据，
并把 tokenizer 保存到项目根目录的 models/ 下。

示例：
    uv run python Example/Pytorch/TransfomerHW1/traintokenizer.py

    # 或者显式指定参数
    uv run python Example/Pytorch/TransfomerHW1/traintokenizer.py \
        --train-file Example/data/TinyStoriesV2-GPT4-train.txt \
        --sample-chars 2000000 \
        --output-dir models
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from minbpe import RegexTokenizer

TOTAL_VOCAB_SIZE = 1024
EOS = "<|endoftext|>"
DEFAULT_DATA_DIR = ROOT / "Example" / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TinyStories tokenizer")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
        help="TinyStories 训练集路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models",
        help="tokenizer 输出目录",
    )
    parser.add_argument(
        "--output-name",
        default="tinystories",
        help="输出文件名前缀，会生成 <name>.model 与 <name>.vocab",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=TOTAL_VOCAB_SIZE,
        help="模型总词表大小（包含 EOS）",
    )
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=1_000_000,
        help="从训练集读取多少个字符用于 BPE 训练",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.vocab_size <= 257:
        raise ValueError("vocab-size 必须大于 257")
    if args.sample_chars <= 0:
        raise ValueError("sample-chars 必须大于零")
    if not args.train_file.is_file():
        raise FileNotFoundError(f"找不到训练集: {args.train_file}")

    eos_id = args.vocab_size - 1

    # 只读取训练集的一部分，先跑通流程；云端可通过 --sample-chars 调大。
    with args.train_file.open(encoding="utf-8") as f:
        sample = f.read(args.sample_chars)
    if not sample:
        raise ValueError(f"训练集为空: {args.train_file}")

    # 不让 BPE 把 EOS 标记本身学成普通文本片段。
    train_text = sample.replace(EOS, "\n\n")

    tokenizer = RegexTokenizer()
    tokenizer.train(
        train_text,
        vocab_size=args.vocab_size - 1,
        verbose=False,
    )

    # 特殊 token 单独分配 ID。
    tokenizer.register_special_tokens({EOS: eos_id})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_dir / args.output_name
    tokenizer.save(str(output_prefix))

    # 验证普通文本和 EOS 都能正确往返。
    text = "Once upon a time." + EOS
    ids = tokenizer.encode(text, allowed_special={EOS})

    assert tokenizer.decode(ids) == text
    assert ids[-1] == eos_id

    # 这个实现的 vocab 不包含 register_special_tokens 注册的 ID，
    # 因此必须把两部分一起计算。
    all_ids = set(tokenizer.vocab) | set(tokenizer.special_tokens.values())
    model_vocab_size = max(all_ids) + 1

    print("训练集:", args.train_file)
    print("训练字符数:", len(train_text))
    print("普通词表大小:", len(tokenizer.vocab))
    print("模型词表大小:", model_vocab_size)
    print("EOS ID:", eos_id)
    print("编码结果:", ids)
    print("保存到:", output_prefix.with_suffix(".model"))


if __name__ == "__main__":
    main()
