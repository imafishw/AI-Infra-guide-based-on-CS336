# AI Infra Guide

基于 CS336 学习路线的 AI Infrastructure 实践记录，包含学习笔记、教学示例与工程代码。目前已完成的内容集中在 Tokenizer。

## 已完成功能

| 模块 | 已有功能 |
| --- | --- |
| [教学版 ByteBPETokenizer](Example/Tokenizer/myTokenizer.py) | 训练、编码与解码、压缩率统计、JSON 模型保存与加载 |
| [BasicTokenizer](src/minbpe/basic.py) | 基础字节级 BPE 训练、编码与解码、模型与词表导出 |
| [RegexTokenizer](src/minbpe/regex.py) | 正则预分词、BPE 训练、编码与解码、特殊 token 管理、模型与词表导出 |
| [GPT4Tokenizer](src/minbpe/gpt4.py) | 基于 tiktoken 的 `cl100k_base` 恢复合并规则、普通文本编码与解码、词表导出 |

## 学习记录

- [Tokenizer Notebook](Example/Tokenizer/Tokenizer.ipynb)：Tokenizer 学习与实验记录。
- [教学版 Tokenizer 说明](Example/Tokenizer/mytokenizerguide.md)：教学实现的配套说明。
- [Tokenizer 笔记](notes/Tokenizer/Tokenizer.md)：BasicTokenizer 与 RegexTokenizer 的流程记录。

## 目录索引

| 目录 | 内容 |
| --- | --- |
| `Example/Tokenizer/` | 教学代码、Notebook 与配套说明 |
| `src/minbpe/` | Tokenizer 包实现 |
| `src/testexamples/` | 工程实验脚本 |
| `notes/` | 学习笔记 |

## 参考

学习路线参考 Stanford CS336，Tokenizer 工程实现参考 Andrej Karpathy 的 minbpe。
