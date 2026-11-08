# ByteBPETokenizer 使用说明

作者：william

这是一个从原 Notebook 整理出来的教学版字节级 BPE Tokenizer。它把训练、编码、解码、压缩率计算以及模型保存/加载集中到了 `ByteBPETokenizer` 类中，不依赖第三方库。

## 1. 最小示例

把 `william_bpe_tokenizer.py` 放在当前工作目录，然后运行：

```python
from william_bpe_tokenizer import ByteBPETokenizer

# 1) 用语料训练。vocab_size=276 表示学习最多 20 条合并规则。
corpus = """
banana bandana banana banana
machine learning learning machine
token token tokenizer token token
hello hello hello world world
"""

tokenizer = ByteBPETokenizer(vocab_size=276)
tokenizer.fit(corpus, verbose=True)

# 2) 编码新文本。
ids = tokenizer.encode("hello world 你好")
print(ids)

# 3) 解码；结果应与原文一致。
text = tokenizer.decode(ids)
print(text)
assert text == "hello world 你好"
```

## 2. 查看训练结果

```python
print(tokenizer.merges)  # {(token_a, token_b): 新 token_id}
print(tokenizer.vocab)   # {token_id: 对应的 bytes}
```

ID `0~255` 对应原始 UTF-8 字节；从 `256` 开始的 ID 是训练过程中学到的新 token。

## 3. 检查压缩率

```python
ratio = tokenizer.compression_ratio("hello hello hello")
print(f"压缩率: {ratio:.2f}")
```

压缩率的定义是“原始 UTF-8 字节数 / 编码后的 token 数”。数值越大，表示这份训练语料上的压缩效果越明显。

## 4. 保存与加载

```python
tokenizer.save("tokenizer.json")

restored = ByteBPETokenizer.load("tokenizer.json")
ids = restored.encode("hello world")
print(restored.decode(ids))
```

## 5. 类中各方法的作用

| 方法 | 用途 |
|---|---|
| `fit(text, verbose=False)` | 在训练文本上学习 BPE 合并规则 |
| `encode(text)` | 把新文本转换成 token ID |
| `decode(ids)` | 把 token ID 还原成文本 |
| `compression_ratio(text)` | 计算指定文本的压缩率 |
| `save(path)` | 保存训练结果 |
| `load(path)` | 加载训练结果 |

## 6. 使用时要注意

- 必须先 `fit()`，再调用 `encode()`、`decode()` 或 `save()`。
- `vocab_size` 至少是 `256`；每增加 1，最多多学习一条合并规则。
- 这是便于理解 BPE 原理的教学实现，不包含 GPT 系列常用的正则预分词、特殊 token 管理和大规模训练优化。
- 字节级实现天然支持中文、英文、emoji 等 UTF-8 文本，但训练语料越有代表性，对新文本的压缩效果通常越好。
- 需要更加modern的方法看目录底下的regex实现 

不同点
BasicTokenizer
text
 ↓
UTF-8 bytes
 ↓
BPE



RegexTokenizer
text
 ↓
regex split
 ↓
["Hello", " world", "!"]
 ↓
每个 chunk → UTF-8 bytes
 ↓
每个 chunk 内 BPE

