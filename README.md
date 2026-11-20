# AI Infra Guide

基于 Stanford CS336 学习路线的 AI Infrastructure 实践记录，包含学习笔记、教学示例与工程代码。目前已实现字节级 BPE Tokenizer、PyTorch Transformer 语言模型、TinyStories 训练流程与基础自回归生成；下一阶段聚焦推理优化。

## 大纲目录

- [目前实现的功能](#目前实现的功能)
- [学习内容与文档](#学习内容与文档)
- [项目目录](#项目目录)
- [使用说明](#使用说明)
- [后续计划](#后续计划)
- [参考](#参考)

## 目前实现的功能

| 模块 | 已有功能 | 代码入口 |
| --- | --- | --- |
| 教学版 ByteBPETokenizer | BPE 训练、编码与解码、压缩率统计、JSON 模型保存与加载 | [myTokenizer.py](Example/Tokenizer/myTokenizer.py) |
| BasicTokenizer | 基础字节级 BPE 训练、编码与解码、模型与词表导出 | [basic.py](src/minbpe/basic.py) |
| RegexTokenizer | 正则预分词、BPE 训练、编码与解码、特殊 token 管理、模型与词表导出 | [regex.py](src/minbpe/regex.py) |
| GPT4Tokenizer | 基于 tiktoken 的 cl100k_base 恢复合并规则、普通文本编码与解码、词表导出 | [gpt4.py](src/minbpe/gpt4.py) |
| TinyStories 分词器训练 | 命令行参数、训练进度、模型与词表保存、分词评估报告及往返一致性检查 | [traintokenizer.py](Example/Pytorch/TransfomerHW1/traintokenizer.py) |
| Transformer 语言模型 | Embedding、RMSNorm、RoPE、多头自注意力、SwiGLU、Transformer Block 与语言模型输出层 | [model.py](Example/Pytorch/TransfomerHW1/model.py) |
| 模型训练 | AdamW、学习率预热与衰减、梯度累积与裁剪、验证损失评估、best/last checkpoint 保存、DDP 多卡训练入口 | [train.py](Example/Pytorch/TransfomerHW1/train.py) |
| 文本生成 | 加载 tokenizer 与 checkpoint、按保存配置重建模型、贪心自回归解码、EOS 停止与上下文截断 | [generate.py](Example/Pytorch/TransfomerHW1/generate.py) |

默认模型配置为 8 层、512 隐藏维度、8 个注意力头、SwiGLU 中间维度 1408、上下文长度 512；4096 词表时约 29.9M 参数。详见 [config.py](Example/Pytorch/TransfomerHW1/config.py)。

以上为已有代码能力，不代表已完成全量训练或性能验证。当前训练仍为 FP32 教学配置，默认读取有限字符样本；双卡显存峰值和完整训练效果需在实际环境验证。

## 学习内容与文档

| 大纲 | 内容与入口 |
| --- | --- |
| 1. Tokenizer 原理与实现 | [实验 Notebook](Example/Tokenizer/Tokenizer.ipynb)、[教学实现说明](Example/Tokenizer/mytokenizerguide.md)、[Tokenizer 笔记](notes/Tokenizer/Tokenizer.md) |
| 2. Transformer / CS336 HW1 | [HW1 学习记录](notes/TramsfomerHW1/Hw1.md)、[环境配置](Example/Pytorch/TransfomerHW1/Instruction.md) |
| 3. TinyStories 训练实践 | [训练说明](Example/Pytorch/TransfomerHW1/TRAINING.md)、[训练笔记](notes/TramsfomerHW1/Training.md) |
| 4. AI Infrastructure 基础 | [基础记录](Example/infrastructure/infrabase.md) |
| 5. 推理优化学习路线 | [阶段计划与验收标准](plan.md) |

## 项目目录

```text
.
├── README.md                       # 项目功能与文档索引
├── pyproject.toml                  # Python 项目与依赖配置
├── plan.md                         # 推理优化阶段计划
├── Example/
│   ├── Tokenizer/                  # 教学版 BPE、Notebook 与说明
│   ├── Pytorch/TransfomerHW1/
│   │   ├── config.py               # 模型与训练配置
│   │   ├── model.py                # Transformer 语言模型
│   │   ├── traintokenizer.py       # TinyStories 分词器训练
│   │   ├── train.py                # 单进程与 DDP 训练
│   │   ├── generate.py             # 基础文本生成
│   │   ├── Instruction.md          # 环境配置
│   │   └── TRAINING.md             # 训练流程与限制
│   └── infrastructure/             # AI Infra 基础记录
├── src/
│   ├── minbpe/                     # Basic、Regex、GPT4 Tokenizer 实现
│   ├── testexamples/               # Tokenizer 实验脚本
│   └── fixtures/                   # 数据相关说明
└── notes/
    ├── Tokenizer/                  # 分词器学习笔记
    └── TramsfomerHW1/              # HW1 与训练笔记
```

## 使用说明

项目要求 Python 3.12 或更高版本，依赖见 [pyproject.toml](pyproject.toml)。使用 uv 时，在仓库根目录运行 `uv sync` 安装依赖。

1. 按 [环境配置](Example/Pytorch/TransfomerHW1/Instruction.md) 设置模块搜索路径。
2. 准备本地 TinyStories 训练与验证文本，检查 [配置文件](Example/Pytorch/TransfomerHW1/config.py) 中的数据路径。
3. 按 [训练说明](Example/Pytorch/TransfomerHW1/TRAINING.md) 训练分词器，再运行模型训练。
4. 生成前修改 [generate.py](Example/Pytorch/TransfomerHW1/generate.py) 中的 checkpoint 路径，使其指向实际训练产物，并使用训练时对应的 tokenizer。当前生成脚本路径与默认训练输出目录不同，需要手动对齐。

## 后续计划

以下为待实现或待验证内容，详细安排见 [plan.md](plan.md)：

- [ ] KV cache：实现 prefill / decode，并与无缓存前向结果对照。
- [ ] Benchmark 与 profiling：测量延迟、吞吐、峰值显存并保存实验数据。
- [ ] 简单 continuous batching：请求调度、完成退出与槽位复用。
- [ ] 完善训练工程：全量语料离线编码、恢复训练与 tokenizer 哈希绑定。
- [ ] 根据实验瓶颈扩展算子优化、推理框架阅读与多卡实验。

## 参考

学习路线参考 Stanford CS336，Tokenizer 工程实现参考 Andrej Karpathy 的 minbpe。
