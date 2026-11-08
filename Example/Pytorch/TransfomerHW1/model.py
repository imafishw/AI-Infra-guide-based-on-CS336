import torch
import torch.nn as nn
from torch import Tensor

from config import TransformerConfig


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # TODO: Initialize embedding weights

    def forward(self, token_ids: Tensor) -> Tensor:
        # TODO: [B, T] -> [B, T, d_model]
        raise NotImplementedError


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        # TODO: Initialize learnable scale

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        raise NotImplementedError


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # TODO: Q, K, V and output projections
        # TODO: RoPE

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

    def forward(self, token_ids: Tensor) -> Tensor:
        # [B, T] -> [B, T, vocab_size]
        raise NotImplementedError
