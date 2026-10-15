class BasicBPE:
    def __init__(self, vocab_size=276):
        """
        初始化 BPE tokenizer

        vocab_size:
            最终词表大小。

        byte-level BPE 的基础词表固定有 256 个 token：
            0 ~ 255

        如果 vocab_size = 276，
        那么需要额外学习：
            276 - 256 = 20
        个 merge token。
        """

        if vocab_size < 256:
            raise ValueError("vocab_size 必须 >= 256")

        self.vocab_size = vocab_size

        # 保存训练出来的 merge 规则
        #
        # 格式：
        # {
        #     (token1, token2): new_token_id
        # }
        #
        # 例如：
        # {
        #     (97, 110): 256
        # }
        #
        # 表示：
        # 97 + 110 -> 256
        self.merges = {}

        # 保存 token_id -> bytes
        #
        # 例如：
        # 97  -> b'a'
        # 256 -> b'an'
        #
        # 一开始只初始化 0~255 的基础 byte token
        self.vocab = {
            idx: bytes([idx])
            for idx in range(256)
        }


    def get_stats(self, ids):
        """
        统计 ids 中所有相邻 token pair 的出现次数。

        参数：
            ids: token id 列表

        例如：
            ids = [97, 97, 98, 97]

        相邻 pair：
            (97, 97)
            (97, 98)
            (98, 97)

        返回：
            {
                (97, 97): 1,
                (97, 98): 1,
                (98, 97): 1
            }
        """

        counts = {}

        # zip(ids, ids[1:])
        #
        # 假设：
        # ids = [1, 2, 3, 4]
        #
        # ids[1:] = [2, 3, 4]
        #
        # zip 后：
        # (1, 2)
        # (2, 3)
        # (3, 4)
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts


    def merge(self, ids, pair, idx):
        """
        把 ids 中所有连续出现的 pair 合并成新的 token idx。

        例如：

            ids  = [97, 97, 97, 98]
            pair = (97, 97)
            idx  = 256

        合并后：

            [256, 97, 98]

        注意：
            匹配成功后 i += 2，
            因为两个 token 已经一起被消费掉。
        """

        newids = []

        i = 0

        while i < len(ids):

            # 必须先检查 i 不是最后一个位置
            #
            # 因为下面还需要访问：
            # ids[i + 1]
            if (
                i < len(ids) - 1
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
            ):
                # 找到目标 pair
                #
                # 用新的 token idx 替换它
                newids.append(idx)

                # 两个 token 都已经处理
                i += 2

            else:
                # 当前 pair 不匹配
                #
                # 当前 token 原样保留
                newids.append(ids[i])

                i += 1

        return newids


    def train(self, text):
        """
        在给定文本上训练 BPE merge 规则。

        训练过程：

        text
          ↓
        UTF-8 bytes
          ↓
        0~255 的基础 token
          ↓
        找最高频 pair
          ↓
        merge
          ↓
        创建新 token 256、257、258...
          ↓
        不断重复
        """

        # 先把文本编码成 UTF-8 bytes
        #
        # 例如：
        # "hello"
        #
        # ->
        #
        # b'hello'
        token_bytes = text.encode("utf-8")

        # bytes 转成整数列表
        #
        # b'hello'
        #
        # ->
        #
        # [104, 101, 108, 108, 111]
        ids = list(token_bytes)

        # 保存训练前长度
        original_length = len(ids)

        # 需要训练多少次 merge
        #
        # 例如：
        # vocab_size = 276
        #
        # 256 个基础 token
        #
        # 所以：
        # num_merges = 20
        num_merges = self.vocab_size - 256

        # 每次重新训练前清空旧规则
        self.merges = {}

        print(f"原始 token 数量: {original_length}")
        print(f"计划训练 merge 次数: {num_merges}")
        print()

        for i in range(num_merges):

            # 统计当前所有相邻 pair
            stats = self.get_stats(ids)

            # 如果已经没有 pair 了
            # 就不能再 merge
            if not stats:
                break

            # 找出现次数最多的 pair
            #
            # stats 例如：
            #
            # {
            #     (97, 110): 10,
            #     (110, 97): 5
            # }
            #
            # max(..., key=stats.get)
            #
            # 会返回：
            # (97, 110)
            pair = max(
                stats,
                key=stats.get
            )

            # 新 token id
            #
            # 第一次：
            # 256
            #
            # 第二次：
            # 257
            #
            # ...
            idx = 256 + i

            print(
                f"merge {i + 1}: "
                f"{pair} -> {idx}, "
                f"count = {stats[pair]}"
            )

            # 真正执行 merge
            ids = self.merge(
                ids,
                pair,
                idx
            )

            # 保存 merge 规则
            self.merges[pair] = idx

        # 训练结束后重建 vocab
        self.build_vocab()

        final_length = len(ids)

        print()
        print(f"训练后 token 数量: {final_length}")

        if final_length > 0:
            compression_ratio = (
                original_length / final_length
            )

            reduction_rate = (
                1 - final_length / original_length
            )

            print(
                f"Compression Ratio: "
                f"{compression_ratio:.2f}x"
            )

            print(
                f"Token Reduction: "
                f"{reduction_rate:.2%}"
            )

        return ids


    def build_vocab(self):
        """
        根据 merges 构建：

            token_id -> bytes

        例如：

            (104, 101) -> 256
            (108, 108) -> 257
            (256, 257) -> 258

        最终：

            vocab[104] = b'h'
            vocab[101] = b'e'

            vocab[256] = b'he'
            vocab[257] = b'll'

            vocab[258] = b'hell'
        """

        # 先重新创建基础 byte vocab
        self.vocab = {
            idx: bytes([idx])
            for idx in range(256)
        }

        # merges.items()
        #
        # 每一项格式：
        #
        # (pair, idx)
        #
        # 比如：
        #
        # ((104, 101), 256)
        for (p0, p1), idx in self.merges.items():

            # 关键：
            #
            # vocab[p0] 和 vocab[p1]
            # 本身可能已经代表多个 byte
            #
            # 所以可以不断组合出更长的 token
            self.vocab[idx] = (
                self.vocab[p0]
                + self.vocab[p1]
            )


    def encode(self, text):
        """
        把普通字符串转换成 BPE token ids。

        注意：
        encode 阶段不会重新训练 BPE。

        它只使用 train() 已经学习好的：
            self.merges

        例如：

            "hello"
              ↓
            UTF-8 bytes
              ↓
            [104, 101, 108, 108, 111]
              ↓
            按照 merges 规则不断 merge
              ↓
            [258, 111]
        """

        # 普通字符串 -> UTF-8 byte ids
        tokens = list(
            text.encode("utf-8")
        )

        # 至少需要两个 token
        # 才可能发生 merge
        while len(tokens) >= 2:

            # 获取当前 token 序列中的 pair
            stats = self.get_stats(tokens)

            if not stats:
                break

            # 找当前存在的 pair 中，
            # 在训练 merge 顺序里优先级最高的。
            #
            # self.merges 的 value：
            #
            # 256
            # 257
            # 258
            #
            # 数值越小，代表训练时越早 merge。
            #
            # 如果 p 不存在于 merges 中：
            #
            # 返回 inf
            #
            # 这样它就不会被优先选择。
            pair = min(
                stats,
                key=lambda p: self.merges.get(
                    p,
                    float("inf")
                )
            )

            # 当前已经不存在任何可用 merge
            if pair not in self.merges:
                break

            # 查找这个 pair 应该变成哪个新 token
            idx = self.merges[pair]

            # 真正执行 merge
            tokens = self.merge(
                tokens,
                pair,
                idx
            )

        return tokens


    def decode(self, ids):
        """
        把 BPE token ids 还原成普通字符串。

        流程：

            token ids
               ↓
            vocab 查表
               ↓
            bytes
               ↓
            拼接
               ↓
            UTF-8 decode
               ↓
            Python str
        """

        # 将每个 token 转回它对应的 bytes
        #
        # 比如：
        #
        # 258 -> b'hell'
        # 111 -> b'o'
        #
        # 最终：
        #
        # b'hell' + b'o'
        #
        # ->
        #
        # b'hello'
        token_bytes = b"".join(
            self.vocab[idx]
            for idx in ids
        )

        # bytes -> Python 字符串
        return token_bytes.decode(
            "utf-8",
            errors="replace"
        )

text = """
banana banana banana
bandana banana bandana
hello hello hello world world
token token tokenizer token
"""

# 原始vocab是0-255
tokenizer = BasicBPE(
    vocab_size=276
)

tokenizer.train(text)

print(tokenizer.merges)

test_text = "hello banana world"

ids = tokenizer.encode(test_text)

print("encoded:")
print(ids)

decoded = tokenizer.decode(ids)

print("decoded:")
print(decoded)

# 验证 encode -> decode 是否正确    
print(
    tokenizer.decode(
        tokenizer.encode(test_text)
    ) == test_text
)