# Transformer HW1 初始准备

## 1. 安装项目依赖

准备 Python 3.12 或更高版本以及 uv，在terminal中执行：

```
cd AI-Infra-guide-based-on-CS336
uv sync
```

后续使用 `uv run python` 运行脚本，以使用项目环境。

## 2. 配置 minbpe 导入路径

本项目的包位于 `src/minbpe`，将其父目录 `src` 加入 Python 模块搜索路径：

macOS / Linux（bash、zsh）：

```bash
cd AI-Infra-guide-based-on-CS336
export PYTHONPATH="$PWD/src"
```

Windows PowerShell：

```powershell
cd AI-Infra-guide-based-on-CS336
$env:PYTHONPATH = "$PWD\src"
```

Windows CMD：

```bat
cd AI-Infra-guide-based-on-CS336
set PYTHONPATH=%CD%\src
```

`$PWD`（或 `%CD%`）是当前目录，所以应先进入项目根目录。设置仅对当前终端及其启动的进程有效，新开终端需要重新执行。无需复制 minbpe 或另装同名包。

验证导入（各平台通用）：

```bash
uv run python -c "from minbpe import RegexTokenizer; print(RegexTokenizer)"
```

脚本中直接使用：

```python
from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
```

Notebook 内核也需要能找到 `src`；在另一个终端设置环境变量不会改变已经运行的内核环境。

## 3. 准备 TinyStories

约定将训练集放在 `data/TinyStories-train.txt`，验证集放在 `data/TinyStories-valid.txt`。这些数据需要另行准备。生成的 tokenizer 文件保存在 `models/`，保存前先创建目录。

只在训练集上学习 BPE，验证集使用同一份训练好的 tokenizer 编码。先用少量文本和较小词表跑通；当前 minbpe 每次合并都会遍历文本块，全量训练较慢且占内存。

现有 `src/testexamples/trainTokenizer.py` 使用另一份示例语料，需先修改数据路径和训练配置，再从项目根目录运行：

```bash
uv run python src/testexamples/trainTokenizer.py
```

## 4. 训练与保存 BPE

- 调用 `RegexTokenizer.train(text, vocab_size=...)` 学习合并规则。
- 若语料以 `<|endoftext|>` 分隔故事，训练前将该标记替换为换行，避免学习标记本身；训练后用 `register_special_tokens()` 分配独立 ID。
- 例如总词表为 1024 时，训练 1023 个普通 token，再将 EOS 注册为 1023。
- 调用 `tokenizer.save("models/tinystories")`，生成 `.model` 和 `.vocab`；前者用于加载，后者用于人工查看。
- 文本含注册过的 EOS 时，用 `encode(text, allowed_special={"<|endoftext|>"})` 编码。
- 语料太少、词表目标太大时，当前训练实现可能因无可合并 pair 而报错，需要减小目标、增加语料或添加提前终止判断。

## 5. 加载并接入 LM

当前父类 `load()` 不会重新编译正则，加载后同步派生状态：

```python
import regex
from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.load("models/tinystories.model")
tokenizer.compiled_pattern = regex.compile(tokenizer.pattern)
tokenizer.register_special_tokens(tokenizer.special_tokens)

all_ids = set(tokenizer.vocab) | set(tokenizer.special_tokens.values())
vocab_size = max(all_ids) + 1
```

完整流程：

```text
文本 → tokenizer.encode() → token ID → torch.long 张量
     → nn.Embedding → Transformer → 输出词表 logits
生成的 ID → 同一个 tokenizer.decode() → 文本
```

Embedding 和 LM 输出层均使用包含特殊 token 的 `vocab_size`。训练时输入与目标错开一位，让模型预测下一个 token。BPE 规则固定，不参与反向传播。

tokenizer 的 `.model` 与 LM checkpoint 是不同文件，必须配套保存和加载；即使词表大小相同，不同 tokenizer 的 ID 含义也可能不同。
