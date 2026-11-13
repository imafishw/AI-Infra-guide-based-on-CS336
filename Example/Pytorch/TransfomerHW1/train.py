"""先训练 tokenizer，再运行本文件。路径均相对于项目根目录。

教学版使用 float32、内存 token 张量；大语料应离线编码并用磁盘映射。
"""
import math
import os
import sys
from pathlib import Path
from dataclasses import asdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from minbpe import RegexTokenizer
from config import TransformerConfig, TransformerConfigTrain
from model import TransformerLM


def get_batch(tokens, cfg, device, generator):
    starts = torch.randint(len(tokens) - cfg.seq_len, (cfg.batch_size,), generator=generator)
    windows = tokens[starts[:, None] + torch.arange(cfg.seq_len + 1)]
    # 输入 [a,b,c]，目标 [b,c,d]：每个位置预测下一个 token。
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


def batch_loss(model, batch):
    x, y = batch
    logits = model(x)
    # 不要先 softmax，cross_entropy 接收原始 logits。
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def build_optimizer(model, cfg):
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=cfg.learning_rate, betas=(0.9, 0.95))


def learning_rate(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps - 1)
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def evaluate(model, tokens, cfg, device):
    model.eval()
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    try:
        return sum(batch_loss(model, get_batch(tokens, cfg, device, generator)).item()
                   for _ in range(cfg.eval_batches)) / cfg.eval_batches
    finally:
        model.train()


def train(cfg=None, model_cfg=None):
    cfg = cfg or TransformerConfigTrain()
    for name in ("batch_size", "seq_len", "grad_accum_steps", "max_steps", "eval_interval", "eval_batches", "max_chars"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} 必须大于零")
    if not 0 <= cfg.warmup_steps < cfg.max_steps:
        raise ValueError("需要 0 <= warmup_steps < max_steps")
    torch.manual_seed(cfg.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world_size > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl')
    tokenizer_path = ROOT / cfg.tokenizer_path
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"请先运行 traintokenizer.py，缺少 {tokenizer_path}")
    tokenizer = RegexTokenizer()
    tokenizer.load(str(tokenizer_path))
    vocab_size = max(set(tokenizer.vocab) | set(tokenizer.special_tokens.values())) + 1
    if set(tokenizer.vocab) | set(tokenizer.special_tokens.values()) != set(range(vocab_size)):
        raise ValueError('tokenizer ID 必须连续')
    model_cfg = model_cfg or TransformerConfig(vocab_size=vocab_size, max_seq_len=cfg.seq_len, device=str(device))
    if model_cfg.vocab_size != vocab_size or model_cfg.max_seq_len < cfg.seq_len:
        raise ValueError("模型词表须匹配 tokenizer，max_seq_len 须 >= seq_len")

    def read_tokens(path):
        with (ROOT / path).open(encoding="utf-8") as f:
            text = f.read(cfg.max_chars)
        ids = tokenizer.encode(text, allowed_special=set(tokenizer.special_tokens))
        if len(ids) <= cfg.seq_len:
            raise ValueError(f"{path} 至少需要 seq_len + 1 个 token")
        return torch.tensor(ids, dtype=torch.long)

    train_tokens, val_tokens = read_tokens(cfg.train_path), read_tokens(cfg.val_path)
    model = TransformerLM(model_cfg).to(device)
    expected_shape = (vocab_size, model_cfg.d_model)
    if tuple(model.embedding.weights.shape) != expected_shape or tuple(model.lm_head.weight.shape) != expected_shape:
        raise ValueError('embedding 和 lm_head 必须与 tokenizer 词表和 d_model 一致')
    if rank == 0:
        print(f'模型: layers={model_cfg.n_layers}, d_model={model_cfg.d_model}, '
              f'vocab={vocab_size}, parameters={sum(p.numel() for p in model.parameters()):,}; '
              f'embedding/lm_head weights={expected_shape}')
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])
    optimizer = build_optimizer(model, cfg)
    generator = torch.Generator().manual_seed(cfg.seed + rank)
    output = ROOT / cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    model.train()
    for step in range(cfg.max_steps):
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            loss = batch_loss(model, get_batch(train_tokens, cfg, device, generator))
            if not torch.isfinite(loss):
                raise RuntimeError("loss 出现 NaN/Inf")
            train_loss += loss.item() / cfg.grad_accum_steps
            (loss / cfg.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % cfg.eval_interval == 0 or step + 1 == cfg.max_steps:
            val_loss = evaluate(model, val_tokens, cfg, device)
            if world_size > 1:
                losses = torch.tensor([train_loss, val_loss], device=device)
                dist.all_reduce(losses)
                train_loss, val_loss = (losses / world_size).tolist()
            if rank != 0:
                continue
            print(f"step={step + 1} train={train_loss:.4f} val={val_loss:.4f} lr={lr:.2e}")
            checkpoint = {"model": (model.module if world_size > 1 else model).state_dict(), "optimizer": optimizer.state_dict(),
                          "step": step + 1, "val_loss": val_loss,
                          "model_config": asdict(model_cfg), "train_config": asdict(cfg)}
            torch.save(checkpoint, output / "last.pt")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(checkpoint, output / "best.pt")
    if world_size > 1:
        dist.destroy_process_group()
    return model


if __name__ == "__main__":
    train()
