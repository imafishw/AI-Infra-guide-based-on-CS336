# TinyStories 训练说明

模型基线现为 8 层、d_model=512、8 个注意力头（每头 64）、SwiGLU d_ff=1408、上下文 512。4096 词表且输入输出权重不共享时，共 29,893,120 参数。embedding 与 lm_head 权重均为 [4096, 512]，前者查表得到 [B,T,512]，后者投影得到 [B,T,4096]。词表大小不需要等于隐藏维度。输出目录改为 checkpoints/tinystories-4096-8x512，避免与旧宽度混淆。此规模是实验起点，不是质量保证；数据覆盖与训练预算仍需扩大后评估。

可在安装 PyTorch 的环境执行 `python Example/Pytorch/TransfomerHW1/model.py`，检查默认 8 层模型的前向形状与反向传播。本机 Python 缺少 torch，目前仅通过语法检查，未验证双卡显存峰值。

在仓库根目录执行。默认词表由 1024 改为 4096，默认 100 万字符训练预计约 22 分钟，实际随机器和文本变化。默认名称为 tinystories-4096，已有产物会拒绝覆盖。

```bash
# 本机冒烟：需要本地训练/验证文本，路径可自行指定
python Example/Pytorch/TransfomerHW1/traintokenizer.py --vocab-size 320 --sample-chars 50000 --eval-chars 20000 --output-name smoke-320
# 回归
python Example/Pytorch/TransfomerHW1/traintokenizer.py --vocab-size 1024 --sample-chars 200000 --output-name regression-1024
# 云端正式分词器：预计约 1.8 小时，不保证压缩率
python Example/Pytorch/TransfomerHW1/traintokenizer.py --sample-chars 5000000 --output-name tinystories-4096
# CS336 词表规模对照（默认采样下约 57 分钟）
python Example/Pytorch/TransfomerHW1/traintokenizer.py --vocab-size 10000 --output-name tinystories-10000
# Linux 云端两张 3090
torchrun --standalone --nproc_per_node=2 Example/Pytorch/TransfomerHW1/train.py
```

报告中 bytes_per_token 使用原始 UTF-8 字节数除以全部 token 数，包含 EOS；tokens_per_word 使用全部 token 数除以英文单词数（不把 EOS 当单词），因此包含标点和边界开销。single_byte_token_ratio 是 ID 小于 256 的 token 出现次数占比，不是词表占比。round_trip 必须为真。采样为文件前缀，报告记录样本 SHA256，便于确认比较使用相同数据。不同口径的历史数字不可直接比较。

词表校验使用完整 ID 集合与 range(vocab_size) 比较：仅检查最大 ID 无法识别中间空洞，因为 EOS 本来就放在最后一位。

DDP 每卡一份模型，local_rank 绑定对应 GPU；反向传播自动平均梯度。各 rank 使用不同采样种子，避免两卡处理同一批数据。只有 rank 0 保存 checkpoint，保存去除 DDP 包装的权重，保持原有加载方式。

当前配置是保守的双卡试运行：每卡 batch=4、序列=512、累积=4，两卡每次更新共 16384 token；仍使用 FP32，1000 步约 1638 万 token。max_chars=100 万仍是教学限制，会重复采样同一小段数据，不能作为完整 TinyStories 正式训练。先确认 loss、显存和吞吐，再考虑混合精度以及批量增大；双卡显存不会自动合并成单卡 48GB。

正式模型训练下一阶段应先将全量语料分片离线编码并存为磁盘映射文件，再增加训练 token 预算；同时补充恢复训练、tokenizer 哈希绑定和独立最终测试集。不要直接让当前 read_tokens 读取 2.2GB 全量文本。旧 1024 tokenizer 对应的 checkpoint 继续配旧 tokenizer 使用，不能混配新词表。

流程：本机冒烟和回归通过后再 push；云端 pull 后训练；.model、.vocab、.report.json 均用 scp 回传。没有执行自动 push 或云端训练。
