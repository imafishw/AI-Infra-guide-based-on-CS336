# HW1 训练流程笔记：单卡、多卡与易错点

核对日期：2026-11-16。依据仓库实际实现整理，目标是用小数据完成训练、验证和保存；不要求扩大数据或步数。用户已反馈云端双卡启动跑通，本次笔记整理没有重新执行 GPU 训练。

## 1. 入口与当前配置

代码位于 `Example/Pytorch/TransfomerHW1/`：

| 文件 | 实际职责 |
|---|---|
| `traintokenizer.py` | 在训练集前缀学习 BPE，独立验证集评估，保存分词器和报告 |
| `config.py` | 模型结构和训练参数，两套 dataclass |
| `model.py` | Embedding、RoPE、因果注意力、SwiGLU、TransformerLM |
| `train.py` | 数据编码、随机窗口、优化、验证、单卡/DDP、checkpoint |
| `generate.py` / `inference.py` | 目前仅注释/空文件，尚无可用推理入口 |

当前是 8 层、宽度 512、8 头（每头 64）、SwiGLU 中间维度 1408、序列长度 512。4096 词表、输入输出权重不共享时为 29,893,120 参数。

训练参数：每卡 batch=4，梯度累积=4，max_steps=1000，train/valid 各最多读取 100 万字符；学习率 3e-4，前 100 步 warmup，随后 cosine 降至 3e-5；每 100 步评估 10 批。不是全量 TinyStories 训练。

## 2. 环境与启动命令

以下命令在 Linux 云端项目根目录执行。使用已跑通的 uv 项目环境；检查 GPU 是否可见：

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

数据默认放在 `Example/data/TinyStoriesV2-GPT4-train.txt` 和 `Example/data/TinyStoriesV2-GPT4-valid.txt`。不要照早期笔记的 `data/TinyStories-train.txt` 路径直接运行。当前训练和分词入口自己将 src 加入搜索路径，无须额外设置 PYTHONPATH。

已存在正确的 tokenizer 时直接训练模型；重新训练分词器示例：

```bash
uv run python Example/Pytorch/TransfomerHW1/traintokenizer.py --vocab-size 4096 --sample-chars 5000000 --eval-chars 200000 --output-name tinystories-4096
```

同名产物存在会拒绝覆盖。改 output-name 后必须同步 config 中 tokenizer_path；小词表冒烟产物不要冒充正式 4096 词表。

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python Example/Pytorch/TransfomerHW1/train.py
```

双卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 Example/Pytorch/TransfomerHW1/train.py
```

等价双卡入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run --standalone --nproc_per_node=2 Example/Pytorch/TransfomerHW1/train.py
```

**易错：是 `uv run torchrun`，不是 `uv torchrun`。** 当前 train.py 没有 argparse，不能传 `--max-steps` 等参数；应修改 config.py。上述环境变量赋值是 Bash 写法，不是 PowerShell 写法。当前多卡使用 NCCL，按 Linux CUDA 环境运行。

## 3. 分词器到模型的数据契约

4096 = 256 个基础字节 token + 3839 个合并 token + 1 个 EOS；EOS ID=4095。学习 BPE 时将 EOS 文本替换成换行，避免把标记学成普通片段；验证和模型训练编码时保留标记，通过 allowed_special 将其编码成一个特殊 ID。

模型从普通词表 ID 与特殊 token ID 的并集推导大小，并检查连续性；不能只用 len(tokenizer.vocab)，它不包含单独注册的 EOS。

```text
token IDs [B,T] → Embedding [B,T,512]
→ 8 个 TransformerBlock [B,T,512]
→ final RMSNorm → LM Head → logits [B,T,4096]
```

embedding.weights 与 lm_head.weight 均为 [4096,512]，但不共享参数。Linear 的权重按 [输出维度,输入维度] 保存，所以前向是隐藏向量乘权重转置。4096 是候选 token 数，512 是每个 token 的特征维度，两者不必相等。每头 512/8=64，满足 RoPE 的偶数要求。

**同样大小的两个 tokenizer 也不一定兼容**：ID 对应的文本可能不同。已有 checkpoint 必须配原 tokenizer，不能只检查维度。当前 checkpoint 未保存 tokenizer 哈希，需自行成套保管。

分词报告：bytes/token 包含原始 EOS 字节和全部 token；tokens/word 的分母是剔除 EOS 后的英文单词数，分子仍包含标点和 EOS；单字节占比按出现次数计算；round-trip 必须为真。报告含采样哈希，但固定种子或哈希并不意味着跨软件、硬件环境逐位可复现。ETA 仅估计 BPE 训练，未覆盖读文件、评估和保存。

## 4. 一步训练内部发生什么

read_tokens 把文件前缀一次编码成 CPU 上的 long 张量；字符数不等于字节数，也不等于 token 数。train 与 valid 是独立文件，但使用同一 tokenizer。

get_batch 随机取起点，每段读取 T+1=513 个 token：

```python
x = windows[:, :-1]  # [a,b,c]，模型输入
y = windows[:, 1:]   # [b,c,d]，每个位置的下一个 token
```

这里不按完整故事切分，窗口可以跨 EOS；EOS 不会重置注意力，故事之间也没有额外 attention mask。因果 mask 只屏蔽未来位置，防止答案泄露。

每一步依次执行：设置学习率 → 清空梯度 → 累积 4 批前向/反向 → 梯度裁剪 → optimizer.step。代码关键点：

```python
(loss / cfg.grad_accum_steps).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
optimizer.step()
```

除以累积次数使梯度对应批次平均；zero_grad 只能在累积开始前做，否则清掉前面批次的梯度。backward 计算梯度，step 才更新参数；裁剪在全部累积完成后做。AdamW 对二维及以上参数施加 weight_decay=0.1，一维 RMSNorm 权重不衰减。

cross_entropy 接收原始 logits，将 [B,T,V] 展成 [B*T,V]，标签展成 [B*T]；不要提前 softmax，标签必须是合法整数 token ID。模型用真实前文同时预测所有位置，训练时不需要逐 token 生成。

## 5. 单卡、多卡、累积和 epoch

| 当前默认配置 | 单卡 | 双卡 |
|---|---:|---:|
| 每卡 micro-batch | 4 | 4 |
| 累积次数 | 4 | 4 |
| 每次更新全局序列数 | 16 | 32 |
| 每次更新 token 数 | 8192 | 16384 |
| 1000 步处理 token 数 | 819.2 万 | 1638.4 万 |

global batch = world_size × batch_size × grad_accum_steps。相同步数下，双卡处理的数据量翻倍，不能直接视作完全相同的单卡实验。若要保持双卡 32 条的全局 batch，单卡可累积 8 次；这里只是比较方法，不要求修改当前试运行配置。

step 是一次参数更新；epoch 是完整遍历数据一次。本项目随机有放回地抽窗口，没有严格 epoch。等效遍数约为 `steps × 每步 token / 训练集 token 数`。假设编码后有 30 万 token，双卡 1000 步约 54.6 遍，但不表示每个位置都恰好出现这么多次。保持 1000 步即可完成当前流程验证。

DDP 工程细节：

- torchrun 启动每卡一个进程；WORLD_SIZE 是总进程数，RANK 是全局编号，LOCAL_RANK 用来绑定本机 GPU。
- CUDA_VISIBLE_DEVICES 会重编号，例如物理 2,3 在进程中变成 cuda:0, cuda:1。
- 每卡各有完整模型、梯度和优化器状态，两张 24GB 不会变成一张 48GB。
- 当前每个进程都重复读文件和编码，CPU 内存和预处理成本随进程增加；没有 DistributedSampler，训练通过 seed+rank 产生不同窗口，但不保证窗口互不重叠。
- DDP 自动平均梯度，无须手工再除 world_size。当前每次 backward 都会同步，不是累积四次才同步；数值逻辑正确，但通信偏多。未来可用 no_sync 优化前几批，必须将对应前向和反向一起包住。
- 验证必须保持各 rank 同步参与；不能直接只让 rank 0 调用当前 DDP evaluate，否则其他进程可能进入不匹配的 collective。

## 6. 验证和 checkpoint 的真实含义

evaluate 使用 eval() 和 no_grad()，结束后恢复 train()。每次重建 seed+1 的生成器，因此每次验证都取相同窗口，方便比较；多卡也取相同窗口，所以 all_reduce 后不会扩大验证覆盖范围。默认评估 10 批，不是遍历整个 valid 文件。

日志 train loss 是当前更新步内的批次均值；val loss 是固定验证窗口的均值，不是整段训练历史的平均。训练 loss 下降但验证 loss 上升可能是过拟合，不能只看训练 loss。

rank 0 在评估点保存：

- last.pt：最近一次评估点，正常跑完是第 1000 步；不是每一步保存。
- best.pt：本次运行中验证 loss 最低的一次。最后一步最好时，两份状态相同。

内容包括 model、optimizer、step、val_loss、model_config、train_config。DDP 保存 model.module.state_dict()，去除包装层，方便普通模型加载。

**易错：train.py 重跑会从随机初始化开始，并覆盖同目录文件。** tokenizer 有覆盖保护，checkpoint 没有；新实验先备份或改 output_dir。当前没有续训入口，也没保存各 rank 的采样 RNG、完整随机状态或历史 best_loss，不能宣称可精确恢复中断现场。文件也不是临时文件加原子重命名保存，中断写盘可能损坏产物。

## 7. 显存、速度与排错

当前采用 FP32，无 AMP、GradScaler、激活检查点或 FlashAttention。注意力显式创建 [B,H,T,T]，序列翻倍时其矩阵元素约变四倍。参数量不能直接代表训练显存，还要计入梯度、Adam 状态和激活。

| 现象 | 优先核对 |
|---|---|
| 启动后暂时无 step 日志 | CPU 编码尚未完成，或尚未到第 100 步；不是逐步打印 |
| 只有一张卡工作 | 是否用 torchrun、进程数和可见 GPU 数是否为 2 |
| CUDA OOM | 先减每卡 batch，再考虑 seq_len；增加累积只能保持有效 batch，不能减少模型本身占用 |
| NCCL 错误或挂起 | 找最早的某 rank 异常、GPU 可见性、驱动环境；末尾报错可能只是连带失败 |
| NaN/Inf | loss 和梯度有检查；排查学习率、输入与数值，再考虑调整配置 |
| 索引越界/维度不符 | 检查 tokenizer 和 checkpoint 是否配套，EOS 是否计入词表 |
| 双卡没有两倍速度 | 当前有重复预处理、每次反向通信、小模型通信开销、验证和磁盘写入 |

可在另一终端 `watch -n 1 nvidia-smi` 观察 GPU，但利用率不能替代训练日志。保留日志可在 Bash 使用 `set -o pipefail` 后把启动命令接 `2>&1 | tee train.log`，避免管道掩盖训练退出码。远程长任务可在已有 tmux 会话运行，防止终端断开影响进程。

## 8. 产物交接与推理边界

代码用 git 同步，models、checkpoints、数据均被忽略，需要 scp 单独传输。至少成套保留 best.pt、原 tokenizer.model 和报告；vocab 文件便于人工检查。不要把同名但不同训练得到的 tokenizer 混在一起。

推理尚未实现。后续流程应为：读取 checkpoint 的 model_config 并覆盖运行 device → 构建模型并严格加载 model 权重 → 加载原 tokenizer → eval/no_grad → 编码非空 prompt → 每次取最后位置 logits 选下一个 token → 拼回输入 → EOS 或长度上限停止 → decode。单卡即可推理，不需要 DDP 或 optimizer。

当前 forward 最长 512 token；生成时须明确上下文截断策略，不能无限追加直接传入。没有 KV cache，每次需要重算输入上下文。此处是待实现流程，不是现有可执行推理命令。

## 9. 本次跑通验收

确认分词器 report 的 round_trip 为 true；模型启动打印正确词表和矩阵形状；两卡进程正常；loss 有限；完成 1000 步；输出目录有 last.pt 与 best.pt。这些证明工程流程可运行，不等同于故事生成质量达标。当前目标下无需增加层数、数据量或训练步数。
