# 完成整套 based on BPE的 tokenizer后
# 要使我们的代码能复用 需要保存bpe训练需要合并的


"""A small educational byte-level BPE tokenizer.

Author: william
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


TokenPair = Tuple[int, int]
PathLike = Union[str, Path]


class ByteBPETokenizer:
    """一个轻量、易读的字节级 BPE Tokenizer。

    训练时从 UTF-8 字节（0~255）开始，反复合并出现次数最多的相邻
    token pair。训练得到的合并规则可用于编码新文本，也可以被保存和加载。

    Parameters
    ----------
    vocab_size:
        目标词表大小。前 256 个 ID 留给原始字节，因此必须不小于 256。
    """

    BASE_VOCAB_SIZE = 256

    def __init__(self, vocab_size: int = 276) -> None:
        if vocab_size < self.BASE_VOCAB_SIZE:
            raise ValueError("vocab_size 必须不小于 256")

        self.vocab_size = vocab_size
        self.merges: Dict[TokenPair, int] = {}
        self.vocab: Dict[int, bytes] = {
            token_id: bytes([token_id])
            for token_id in range(self.BASE_VOCAB_SIZE)
        }
        self.is_fitted = False

    @staticmethod
    def _get_stats(ids: Iterable[int]) -> Dict[TokenPair, int]:
        """统计相邻 token pair 的出现次数。"""
        ids = list(ids)
        counts: Dict[TokenPair, int] = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids: Iterable[int], pair: TokenPair, new_id: int) -> List[int]:
        """把序列中所有不重叠的 ``pair`` 合并为 ``new_id``。"""
        ids = list(ids)
        result: List[int] = []
        i = 0

        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1

        return result

    def fit(self, text: str, verbose: bool = False) -> "ByteBPETokenizer":
        """在 ``text`` 上学习 BPE 合并规则，并返回自身。"""
        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {
            token_id: bytes([token_id])
            for token_id in range(self.BASE_VOCAB_SIZE)
        }

        num_merges = self.vocab_size - self.BASE_VOCAB_SIZE
        for step in range(num_merges):
            counts = self._get_stats(ids)
            if not counts:
                break

            # 次数相同时选择数值较小的 pair，让训练结果可复现。
            pair = max(
                counts,
                key=lambda item: (counts[item], -item[0], -item[1]),
            )
            new_id = self.BASE_VOCAB_SIZE + step
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                piece = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"merge {step + 1:>2}: {pair} -> {new_id} "
                    f"({counts[pair]} 次, {piece!r})"
                )

        self.is_fitted = True
        return self

    def encode(self, text: str) -> List[int]:
        """使用已经学习的规则把字符串编码成 token ID 列表。"""
        self._check_is_fitted()
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            available_pairs = [pair for pair in stats if pair in self.merges]
            if not available_pairs:
                break

            # ID 越小，代表该规则在训练中越早学到，必须优先应用。
            pair = min(available_pairs, key=self.merges.__getitem__)
            ids = self._merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """把 token ID 还原成字符串。"""
        self._check_is_fitted()
        try:
            raw_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"未知 token ID: {error.args[0]}") from error
        return raw_bytes.decode("utf-8", errors=errors)

    def compression_ratio(self, text: str) -> float:
        """返回 UTF-8 字节数与编码后 token 数之比。"""
        token_ids = self.encode(text)
        return len(text.encode("utf-8")) / max(len(token_ids), 1)

    def save(self, path: PathLike) -> None:
        """把词表大小与合并规则保存为 JSON。"""
        self._check_is_fitted()
        data = {
            "author": "william",
            "vocab_size": self.vocab_size,
            "merges": [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.merges.items()
            ],
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: PathLike) -> "ByteBPETokenizer":
        """从 JSON 文件恢复 Tokenizer。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(vocab_size=int(data["vocab_size"]))

        for item in data["merges"]:
            pair = tuple(item["pair"])
            if len(pair) != 2:
                raise ValueError(f"非法 pair: {pair}")
            typed_pair = (int(pair[0]), int(pair[1]))
            new_id = int(item["new_id"])
            tokenizer.merges[typed_pair] = new_id
            tokenizer.vocab[new_id] = (
                tokenizer.vocab[typed_pair[0]] + tokenizer.vocab[typed_pair[1]]
            )

        tokenizer.is_fitted = True
        return tokenizer

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit(text) 训练 Tokenizer")


if __name__ == "__main__":
    corpus = """
    banana bandana banana banana
    machine learning learning machine
    token token tokenizer token token
    hello hello hello world world
    """

    tokenizer = ByteBPETokenizer(vocab_size=276).fit(corpus)
    sample = "hello world 你好"
    token_ids = tokenizer.encode(sample)

    print("原文:", sample)
    print("Token IDs:", token_ids)
    print("还原:", tokenizer.decode(token_ids))
    print(f"压缩率: {tokenizer.compression_ratio(sample):.2f}")
