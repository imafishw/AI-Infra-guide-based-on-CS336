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
import json
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from minbpe import RegexTokenizer

TOTAL_VOCAB_SIZE = 4096
EOS = "<|endoftext|>"
DEFAULT_DATA_DIR = ROOT / "Example" / "data"


def format_duration(seconds):
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class BPEProgress:
    """在终端显示 BPE 合并进度、耗时和预计剩余时间。"""

    def __init__(self, total, width=30, min_interval=0.1):
        self.total = total
        self.width = width
        self.min_interval = min_interval
        self.start = time.monotonic()
        self.last_update = 0.0
        self.is_tty = sys.stderr.isatty()
        self.print_every = max(1, total // 20)

    def __call__(self, step, total, idx, pair, count):
        now = time.monotonic()
        finished = step >= total
        if self.is_tty:
            due = finished or now - self.last_update >= self.min_interval
        else:
            due = finished or step % self.print_every == 0
        if not due:
            return

        self.last_update = now
        ratio = step / total if total else 1.0
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        eta = elapsed / ratio - elapsed if ratio > 0 else 0.0
        message = (
            f"BPE [{bar}] {ratio:6.2%} {step}/{total} "
            f"已用 {format_duration(elapsed)} 剩余 {format_duration(eta)} "
            f"新 token={idx} 合并次数={count}"
        )

        if self.is_tty:
            sys.stderr.write("\r" + message)
            sys.stderr.flush()
            if finished:
                sys.stderr.write("\n")
        else:
            sys.stderr.write(message + "\n")


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
        default="tinystories-4096",
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭训练进度显示",
    )
    parser.add_argument('--valid-file', type=Path, default=DEFAULT_DATA_DIR / 'TinyStoriesV2-GPT4-valid.txt')
    parser.add_argument('--eval-chars', type=int, default=200_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.sample_chars <= 100_000_000:
        raise ValueError('sample-chars 必须在 1 到 1 亿之间：纯 Python 内存/时间护栏，全量请换快速实现')
    if not 0 < args.eval_chars <= 100_000_000:
        raise ValueError('eval-chars 必须在 1 到 1 亿之间')
    if args.valid_file.resolve() == args.train_file.resolve():
        raise ValueError('验证集不能是训练集')
    with args.valid_file.open(encoding='utf-8') as f:
        valid = f.read(args.eval_chars)
    if not valid:
        raise ValueError('验证集为空')
    output_prefix = args.output_dir / args.output_name
    if any(Path(str(output_prefix) + ext).exists() for ext in ('.model', '.vocab', '.report.json')):
        raise FileExistsError('输出已存在，请换 --output-name，避免覆盖旧 tokenizer')

    if args.vocab_size <= 257:
        raise ValueError("vocab-size 必须大于 257")
    if args.sample_chars <= 0:
        raise ValueError("sample-chars 必须大于零")
    if not args.train_file.is_file():
        raise FileNotFoundError(f"找不到训练集: {args.train_file}")

    eos_id = args.vocab_size - 1
    num_merges = args.vocab_size - 1 - 256

    # 只读取训练集的一部分，先跑通流程；云端可通过 --sample-chars 调大。
    print(f"读取训练集: {args.train_file} (最多 {args.sample_chars} 字符)", flush=True)
    with args.train_file.open(encoding="utf-8") as f:
        sample = f.read(args.sample_chars)
    if not sample:
        raise ValueError(f"训练集为空: {args.train_file}")

    # 不让 BPE 把 EOS 标记本身学成普通文本片段。
    train_text = sample.replace(EOS, "\n\n")

    progress = None if args.no_progress else BPEProgress(num_merges)
    print(f"开始训练 BPE: 读取 {len(train_text)} 字符，目标 {num_merges} 次合并", flush=True)
    tokenizer = RegexTokenizer()
    estimate = num_merges * len(train_text) * 7 / (20 * 1_000_000)
    print(f'粗略训练 ETA: {format_duration(estimate)}（线性外推，依赖硬件和语料）', flush=True)
    started = time.monotonic()
    tokenizer.train(
        train_text,
        vocab_size=args.vocab_size - 1,
        verbose=False,
        progress=progress,
    )

    # 特殊 token 单独分配 ID。
    tokenizer.register_special_tokens({EOS: eos_id})
    elapsed = time.monotonic() - started
    all_ids = set(tokenizer.vocab) | set(tokenizer.special_tokens.values())
    if all_ids != set(range(args.vocab_size)):
        raise RuntimeError('最终词表不连续或大小不符合请求')
    valid_ids = tokenizer.encode(valid, allowed_special={EOS})
    words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", valid.replace(EOS, ' ')))
    metrics = dict(bytes_per_token=len(valid.encode('utf-8')) / len(valid_ids),
                   tokens_per_word=len(valid_ids) / words if words else None,
                   single_byte_token_ratio=sum(i < 256 for i in valid_ids) / len(valid_ids),
                   round_trip=tokenizer.decode(valid_ids) == valid)
    if not metrics['round_trip']:
        raise RuntimeError('验证集 round-trip 失败')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_dir / args.output_name
    tokenizer.save(str(output_prefix))
    report = dict(train_file=str(args.train_file.resolve()), valid_file=str(args.valid_file.resolve()),
                  vocab_size=args.vocab_size, requested_sample_chars=args.sample_chars,
                  sample_chars=len(sample), training_chars=len(train_text), eval_chars=len(valid),
                  actual_merges=len(tokenizer.merges), training_seconds=elapsed,
                  estimated_seconds=estimate, estimate_error_seconds=elapsed-estimate,
                  train_sample_sha256=hashlib.sha256(sample.encode('utf-8')).hexdigest(),
                  valid_sample_sha256=hashlib.sha256(valid.encode('utf-8')).hexdigest(),
                  metrics=metrics, metric_definition='UTF-8 bytes and all tokens include EOS; English words exclude EOS',
                  sampling='file prefix', python_version=sys.version)
    Path(str(output_prefix) + '.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f'实际训练 {elapsed:.2f}s，预估偏差 {elapsed-estimate:+.2f}s')

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
