# gdtensor 入门导引：写给做分布式存储的人

这份导引写给有分布式存储经验、没接触过模型训练和数值计算的人，目标是把上手这个项目需要的前置知识一次讲全。我的判断是，gdtensor 的大部分概念你已经会了，只是名字不同：它本质上是一个"张量版的条带布局 + 拷贝计划生成器"，给定数据在计算侧和存储侧各自怎么切，算出每个进程该把哪段字节搬到文件的哪个偏移，再按格式原子发布。你需要补的是三层东西：张量和训练的基本词汇，张量在几千个进程之间怎么切，以及这个项目自己的一套建模语言。补完这三层，剩下的问题几乎都是存储问题，而且项目当前最薄弱的也恰好在存储侧。

本文的数字都来自仓库内的文档、示例报告、测试和我在本机实际跑出的结果，出处随文注明。设计的唯一依据是 [DTensorsView-design.md](DTensorsView-design.md)，本文与它冲突时以设计文档为准。

## 0 怎么用这份导引

前置知识按"不懂就读不下去代码"的程度分三档：

| 档位 | 含义 | 对应章节 |
|-|-|-|
| 必需 | 不懂就看不懂设计文档和任何一个测试 | 第 2、3、4、6 节，第 10 节的跑通部分 |
| 需要 | 第一个月内会碰到，改代码前要懂 | 第 5、7、8、9、13 节 |
| 选修 | 只在特定方向上需要 | 第 11 节形式化方法；第 3.6 节的 torch 进程组细节 |

每节开头写了"掌握到什么程度"，结尾的自测题（第 14 节）能用来检查自己是否真的懂了。存储方面的知识（pread/pwrite、fsync、原子 rename、对象存储语义）你已经有了，本文只指出这个项目对它们的具体用法，不从头讲。

## 1 这个项目在解决什么问题

一个大模型，在存储眼里就是几百个有名字的多维数组，训练的过程就是反复更新这些数组。仓库里用来跑真实文件测试的 Qwen3-0.6B 有 311 个数组，payload 共 1,503,264,768 字节（见 [artifacts/cache-fix-notes.md](../artifacts/cache-fix-notes.md)）；更大的模型是同样的结构，只是数组更多、更大。这些数组在 PyTorch 里叫张量（tensor），定期把它们写到盘上，就是 checkpoint。

麻烦在于，训练时这些张量并不完整地待在某一张卡上。一个几千张卡的训练任务会把每个张量切开分散到各个进程，每个进程只持有自己那一片；写 checkpoint 时，几千个进程要并发地把各自的碎片写进同一组文件。读的时候情况更复杂：读的一方往往是另一个集群，卡数和切法都与写的时候不同。设计文档举的例子是 LAWA（对最近 k 个 checkpoint 的权重做逐元素平均）：训练集群按 32 卡的切法写了 k 份 checkpoint，处理任务却在一个 12 卡的临时集群上跑，它要把每份 checkpoint 按自己的切法读进来，平均之后再写出去。

所以这个项目的核心问题，翻译成存储语言就是：同一份逻辑数据，写入端按一种条带方式分布在 N 个客户端上，读取端按另一种条带方式分布在 M 个客户端上，文件本身又是第三种布局，系统要算出每个客户端读写哪些字节区间，并且保证不丢、不重、不错位，发布是原子的。gdtensor 的核心只做这件事：根据两侧的描述生成 `CopyPlan`（源区间到目标区间的列表）并执行。它不做序列化，不决定文件布局，也不做平均这类数值计算，这些都留给业务层。

此外还有一个读侧的问题：同一份权重可能有上万个 rank 同时要读，存储扛不住这种热点。设计用节点本地的多级缓存逐级扇出来分摊，这部分叫读链，第 7 节单独讲。

## 2 张量和数值基础（必需）

掌握到什么程度：看到 `GlobalTensorMeta([1024, 2816], "BF16", 2)` 能立刻说出它多大、第 `(i, j)` 个元素在第几个字节；知道 reshape 不搬数据；知道为什么平均要用更宽的类型累加。

### 2.1 形状、dtype 和行优先展平

张量就是带形状的数组。描述它只需要三样东西：形状（`shape`，比如 `[1024, 2816]`）、元素类型（`dtype`）和元素大小（`itemsize`），仓库里合起来叫 `GlobalTensorMeta`。字节数就是各维相乘再乘 itemsize：`1024 × 2816 × 2 = 5,767,168` 字节。

张量落盘时按行优先（row-major，也叫 C order）展平成一段连续字节：最后一维变化最快。二维时第 `(i, j)` 个元素的字节偏移是 `(i × 列数 + j) × itemsize`，三维 `[A, B, C]` 时第 `(a, b, c)` 个元素是 `((a × B + b) × C + c) × itemsize`。之后所有的"切分"都会变成这条线性地址上的区间集合，这正是你熟悉的领域。

形状可以有 0 维（标量，1 个元素）和长度为 0 的维（空张量，0 字节）。测试特意覆盖了这两种边界，因为它们最容易让偏移计算出错。

### 2.2 常见 dtype

| dtype | 字节 | 说明 |
|-|-|-|
| `F32` | 4 | 单精度浮点，优化器状态和累加器常用 |
| `BF16` | 2 | bfloat16：1 位符号、8 位指数、7 位尾数。数值范围和 F32 一样大，精度只有约 3 位有效十进制数字。大模型权重最常用的类型 |
| `F16` | 2 | 半精度：5 位指数、10 位尾数。精度比 BF16 高，范围小得多，容易溢出 |
| `F64` | 8 | 双精度，测试里做 oracle 计算用 |
| `I64` | 8 | 64 位整数，测试最爱用，因为值可以精确比对 |

一个和代码直接相关的细节：NumPy 没有 bfloat16 类型。仓库处理 BF16 时始终把它当成原始的 2 字节搬运，需要计算时再交给 torch 解释，不经过 F32 中转（`gdtensor_conn/dcp_backend.py` 开头注释）。对存储人来说这反而简单：核心层只搬字节，从不解释字节。

### 2.3 reshape 不搬数据

reshape 只换一种方式解读同一段行优先字节，元素顺序不变。`[3, 4]` reshape 成 `[2, 6]` 或 `[12]`，内存里一个字节都不动。这个性质是 `ReshapedDTensor` 的基础：优化器状态在计算侧被展平成一维再切分（第 3.4 节的 ZeRO），但落盘时仍然要用原来的二维形状命名，gdtensor 用"底层按一维切、逻辑上是二维"表达这件事，两者的字节完全一致。

### 2.4 为什么平均要用宽类型累加

LAWA 要对 k 份 BF16 权重求平均。直接在 BF16 里累加，每加一次都要舍入到 7 位尾数，误差会累积；正确做法是先转成 F32 或 F64 累加，最后再转回 BF16。仓库的 `tests/torch/test_lawa_dtype.py` 锁定了这一点：直接用 BF16 累加的误差被要求严格大于 F32 累加的 2 倍以上，以防有人"优化"掉宽类型累加器。

平均还有一种增量算法叫 Welford：`mean_k = mean_{k-1} + (x_k - mean_{k-1}) / k`，好处是不用一次把 k 份都读进内存，每读一份更新一次。设计文档第 10.2 节两种写法都给了。整数类型转回去时按截断处理，所以 `[5.5, 6.5, 7.5]` 存成 I64 是 `[5, 6, 7]`（README 的 checkpoint 合并测试一节有这个例子）。

## 3 训练和分布式训练（必需）

掌握到什么程度：能看懂 `gdtensor_conn/models.py` 里每个张量为什么是那种切法；能说出 TP=4、DP=4、PP=2 时 rank 13 在哪个 stage、哪个副本、第几个分片；知道 DTensor 的 `Shard`/`Replicate` 是什么。反向传播、梯度、loss 和注意力的数学可以完全不学，这个项目的代码不涉及它们。

### 3.1 一次训练在做什么

训练是一个循环：取一批数据，前向计算得到预测，和正确答案比较得到误差（loss），反向计算出每个权重该往哪个方向调（梯度），再由优化器按梯度更新权重。对 checkpoint 来说，只有循环结束时留下的东西重要：权重本身，加上优化器为每个权重维护的状态。

### 3.2 一个模型里有哪些张量

张量名会出现在每个测试和日志里，认识它们能省很多时间。仓库模拟的是一个 Llama/Qwen 风格的 transformer（`gdtensor_conn/models.py` 的 `model_tensors`），张量名沿用 HuggingFace 的命名：

| 张量名 | 形状 | 是什么 |
|-|-|-|
| `model.embed_tokens.weight` | `[V, H]` | 词表嵌入，每个词一行向量。V 是词表大小，H 是隐藏维度 |
| `model.layers.{i}.input_layernorm.weight` | `[H]` | 归一化层的缩放系数，很小 |
| `model.layers.{i}.self_attn.{q,k,v}_proj.weight` | `[H, H]` | 注意力的三个输入投影矩阵 |
| `model.layers.{i}.self_attn.o_proj.weight` | `[H, H]` | 注意力的输出投影矩阵 |
| `model.layers.{i}.mlp.{gate,up}_proj.weight` | `[F, H]` | 前馈网络的两个输入矩阵，F 是前馈维度 |
| `model.layers.{i}.mlp.down_proj.weight` | `[H, F]` | 前馈网络的输出矩阵 |
| `model.norm.weight`、`lm_head.weight` | `[H]`、`[V, H]` | 最后一层归一化和输出层 |
| `...mlp.experts.{e}.{gate,up,down}_proj.weight` | 同上 | MoE 模型里第 e 个 expert 的前馈网络 |
| `optimizer.state.<名字>.exp_avg` / `.exp_avg_sq` | 同对应权重 | Adam 优化器的两份状态，F32 |

规律是：模型由 L 个结构相同的层叠起来，每层一组注意力矩阵加一组前馈矩阵，参数量几乎全在这些大矩阵里，归一化系数可以忽略。PyTorch 的线性层权重形状是 `[输出维, 输入维]`，这一点决定了下面 TP 的切法。

### 3.3 优化器状态

最常用的 Adam 优化器为每个权重额外保存两份同形状的 F32 统计量：一阶矩 `exp_avg` 和二阶矩 `exp_avg_sq`。在仓库的模拟模型里，权重是 BF16，每个元素 2 字节，两份 F32 状态每个元素共 8 字节，所以打开 `--optimizer` 后 checkpoint 体积约为纯权重的 5 倍。真实训练里往往还有一份 F32 的主权重副本，比例更大。对存储来说，这意味着 checkpoint 的大头经常是优化器状态，而不是模型本身。

### 3.4 四种并行方式

模型太大一张卡放不下，计算也太慢，所以训练会同时用几种方式把张量和工作分出去。对 checkpoint 来说，你只需要知道每种方式让张量在 rank 之间怎么分布：

| 缩写 | 全称 | 对张量做了什么 | 存储类比 |
|-|-|-|-|
| DP | Data Parallel，数据并行 | 每组 rank 持有完整的同一份权重，各自处理不同的训练数据 | 多副本 |
| TP | Tensor Parallel，张量并行 | 把单个大矩阵按行或按列切给几个 rank | 把一个大对象条带化到几块盘 |
| PP | Pipeline Parallel，流水线并行 | 按层切：前几层在 stage 0，后几层在 stage 1 | 按命名空间把数据分到不同节点组 |
| EP | Expert Parallel，专家并行 | 把 MoE 的不同 expert 分给不同 rank | 按 key 分桶 |

**TP 的列切和行切。** 这是最需要理解的一种。Megatron-LM 的做法是把线性层成对切：`q/k/v_proj`、`gate/up_proj` 按输出维切（权重 `[out, in]` 的第 0 维，叫 column parallel），`o_proj`、`down_proj` 按输入维切（第 1 维，叫 row parallel）。这样配对的原因是，列切层的输出本来就按隐藏维分散在各个 rank 上，正好可以直接喂给行切层，一个前馈块只需要在末尾做一次跨卡求和。这和存储无关，但它解释了为什么 `models.py` 里 `kind == "col"` 对应 `Shard(0)`、`kind == "row"` 对应 `Shard(1)`，也解释了第 6 节和第 12 节的写入碎片：行优先文件里，第 1 维被切开的矩阵必然产生跨步小写入。词表嵌入和 `lm_head` 按词表维切（`Shard(0)`），归一化系数太小，每个 rank 都存一份完整的（`Replicate`）。

**ZeRO 和 FSDP。** 普通 DP 每个副本都存一份完整的优化器状态，很浪费。ZeRO 的做法是把优化器状态（stage 1）、梯度（stage 2）乃至权重本身（stage 3）也在 DP 组内打散，每人只存一片；PyTorch 的 FSDP 大致相当于 ZeRO-3。仓库模拟的是"分布式优化器"：优化器状态先展平成一维，再在 DP×TP 上切开（`models.py` 中 `kind == "zero"` 的分支），这就是第 2.3 节 `ReshapedDTensor` 的用途。

**EP 和 MoE。** MoE（Mixture of Experts）模型的每一层有很多个结构相同的前馈网络（expert），每个 token 只经过其中少数几个。EP 把不同 expert 放到不同 rank 上。仓库的模拟模型把 EP 折叠进 DP×TP（expert e 放在其 stage 内第 `e % (DP×TP)` 个 rank 上，见 `_expert_owner`）；`tests/checkpoint/test_parallel_modes.py` 则把 EP 当作独立的第四维。

### 3.5 rank 编号和多维网格

训练任务里每张 GPU 通常对应一个进程，进程的全局编号叫 rank，总数叫 world size；同一台机器上的进程还有一个机内编号 local rank。这几种并行会叠加，所有 rank 排成一个多维网格。仓库沿用 Megatron 的排列：`rank = (pp × DP + dp) × TP + tp`，TP 在最内层，PP 在最外层（见 `models.py` 开头注释）。

以 TP=4、DP=4、PP=2 为例，world size 是 32。反过来算 rank 13：`tp = 13 % 4 = 1`，`dp = 13 // 4 % 4 = 3`，`pp = 13 // 16 = 0`，即 stage 0、第 3 个 DP 副本、第 1 个 TP 分片。TP 放在最内层，是因为 TP 的通信最频繁，相邻编号的 rank 通常在同一台机器上，可以走机内高速互联。

### 3.6 进程组和集合通信（选修，读 torch 测试前需要）

多进程之间靠集合通信同步数据。你只需要认识四个操作：`all_reduce`（每人贡献一个值，所有人拿到求和或求最小值的结果）、`all_gather`（每人贡献一片，所有人拿到拼起来的全部）、`broadcast`（一人发、所有人收）、`barrier`（所有人到齐再继续）。GPU 上用 NCCL 后端，CPU 上用 Gloo 后端，仓库的测试全在 CPU 上跑，所以只用 Gloo。

启动多进程任务用 `torchrun`。它为每个进程设置 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 等环境变量，多台机器之间通过一个 rendezvous 端点会合（README 的"torchrun + Gloo CPU 同步"一节有命令）。在 gdtensor 自己的代码里，直接调用集合通信的是 `gdtensor_conn/coord.py` 的 `GlooCoordinator`（真实 DCP 后端内部也会用到，那是 torch 的事）：提交共享文件前，各 rank 用 `all_reduce(MIN)` 投票，任一 rank 失败则所有 rank 一起放弃。这就是一次两阶段提交的投票阶段，你会觉得很眼熟。

### 3.7 DTensor：把"怎么切"写成数据

PyTorch 用 `torch.distributed.tensor.DTensor` 表示"一个逻辑张量 + 它在一组 rank 上怎么分布"。分布方式由两样东西描述：

- **DeviceMesh**：参与的 rank 排成的网格，比如 `(PP, DP, TP) = (2, 4, 4)`；
- **placements**：网格的每一维各对应一个放置方式。`Shard(d)` 表示沿张量第 `d` 维切开，`Replicate()` 表示每人一份完整副本，`Partial()` 表示每人持有一部分和、要相加才是真值。

模拟模型里一个列切的权重写作 `[Replicate, Replicate, Shard(0)]`：在 PP 维和 DP 维上复制，在 TP 维上沿第 0 维切开。`Partial` 不能落盘，代码会直接拒绝（`gdtensor/internal/dtensor.py`）。

**不整除时怎么切。** gdtensor 和 `torch.chunk` 一致：n 个元素切 k 份，每份 `ceil(n / k)` 个，最后几份可能更少，甚至为空。4 列切给 3 个 rank，每份 2 列，rank 0 拿第 0–1 列，rank 1 拿第 2–3 列，rank 2 拿到空分片。空分片是合法的，不报错，也不占字节。

**多个网格维切同一个张量维。** placements 里可以出现两个 `Shard(0)`，比如优化器状态的 `[R, Shard(0), Shard(0)]`：先按 DP 维把第 0 维切成 DP 份，每份再按 TP 维切成 TP 份，嵌套顺序就是网格维的顺序。

gdtensor 把 DTensor 的这套描述推广到了存储侧和缓存侧：计算侧的参与者是 rank（`ComputeMesh`），存储侧的参与者是文件里的 slot（`StorageMesh`），缓存侧的参与者是节点（`CacheMesh`）。三侧共用同一种张量描述，各自换一种 mesh，这是整个设计的主线。

## 4 文件格式（必需）

掌握到什么程度：能用十行 Python 手工解析一个 safetensors 文件；知道 DCP 的完成标记是什么；知道 gdtensor 在两种格式上各加了什么约定。

### 4.1 safetensors

HuggingFace 的单文件格式，结构极简：

```text
| 8 字节小端 u64：header 长度 N | N 字节 JSON header（空格补齐到 8 字节对齐）| payload |
```

JSON header 是一个字典，键是张量名，值是 `{"dtype": "BF16", "shape": [1024, 2816], "data_offsets": [begin, end]}`，`data_offsets` 相对 payload 开头，不是相对文件开头。可选的 `"__metadata__"` 键存一个字符串字典。gdtensor 写出的 header 里 `__metadata__` 是 `{"format": "pt", "writer": "gdtensor"}`（`gdtensor_conn/formats.py`）。手工读取只需要：

```python
import json, struct
raw = open("model.safetensors", "rb").read()
n = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8:8 + n])
payload = raw[8 + n:]          # 每个张量的字节在 payload[begin:end]
```

对第 10.3 节模拟 CLI 写出的 `model-00001-of-00002.safetensors` 运行这段代码，得到 `N = 2264`（8 的倍数），header 开头是：

```text
__metadata__                             {'format': 'pt', 'writer': 'gdtensor'}
model.embed_tokens.weight                {'dtype': 'BF16', 'shape': [8192, 1024], 'data_offsets': [0, 16777216]}
model.layers.0.input_layernorm.weight    {'dtype': 'BF16', 'shape': [1024], 'data_offsets': [16777216, 16779264]}
model.layers.0.self_attn.q_proj.weight   {'dtype': 'BF16', 'shape': [1024, 1024], 'data_offsets': [16779264, 18876416]}
```

同一目录下的 `.gdtensor_committed` 内容是 `{"format": "safetensors", "files": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]}`。

这和设计文档第 5.3 节的偏移模型一一对应：`contiguous_payload_offset` 就是 `data_offsets` 那一层，`get_file_offset` 再加上 `8 + N`。

大模型通常拆成多个文件（`model-00001-of-00004.safetensors` ……），外加一个 `model.safetensors.index.json`，其中 `weight_map` 记录每个张量在哪个文件里。gdtensor 只在每个张量都完整落在单个文件里时才写这个 index。

gdtensor 自己加了两条约定。第一，当一个张量在文件里的 slot 能按行优先顺序拼成完整张量时，header 用张量名作为一个条目；否则每个 slot 一个条目，名字是 `<张量名>/slot<w>`，形状是该 slot 持有的那块。两种写法都是合法的 safetensors。第二，所有分片发布之后，最后原子发布一个 `.gdtensor_committed` 标记文件，里面记录了已提交的分片集合；没有这个标记的目录，读端拒绝打开。

### 4.2 PyTorch DCP

PyTorch Distributed Checkpoint 是为分布式训练设计的格式：每个 rank 写自己的 payload 文件（真实 DCP 叫 `__<rank>_<n>.distcp`），所有 payload 写完后再写一个 `.metadata` 文件，记录每个张量的每个分片（chunk）的全局偏移、形状和它在哪个文件的哪段字节。`.metadata` 的发布就是完成标记。真实 DCP 的 `.metadata` 用 pickle 序列化，DCP 自己也支持读时换切分。

仓库里有两套 DCP 实现，容易混淆。`gdtensor_conn/formats.py` 里的 distcp 是一个仿真格式，`.metadata` 用 JSON，和真实 DCP 不兼容；`gdtensor_conn/dcp_backend.py` 则调用真实的 `torch.distributed.checkpoint` API 写出真正的 DCP 文件。README 里说"distcp 仿真实现使用 JSON metadata"指的是前者。

### 4.3 两种格式的发布方式对照

| | safetensors | DCP |
|-|-|-|
| 元数据位置 | 文件开头的 header | 独立的 `.metadata` 文件 |
| 写入顺序 | 先 header 后 data | 先 payload 后 `.metadata` |
| 单文件发布 | 写临时文件，fsync，rename | payload 文件可以直接写 |
| checkpoint 级完成标记 | `.gdtensor_committed`（gdtensor 约定） | `.metadata` |

## 5 gdtensor 自己的建模语言（需要）

掌握到什么程度：能读懂设计文档附录 B 的类型定义；能看懂 `make_builder_mixin`、`Ok`/`Err` 和 `@catch` 的作用；能手算一个小张量的 `CopyPlan`。

设计文档和代码有一套自己的写法，第一次看会觉得绕。它们都来自几个简单的想法。

### 5.1 代数数据类型和 viba

设计文档附录 B 用一种叫 viba 的 DSL 写类型定义（[lixinqi/viba](https://github.com/lixinqi/viba)），文档里统一把它的代码块标成 `bash`，那只是语法高亮，不是 shell 命令。你只需要认识三种构造：

- **Object（积类型）**：几个字段同时存在，相当于 struct 或 dataclass。比如 `GlobalTensorMeta` 有 `global_shape`、`dtype`、`itemsize` 三个字段。
- **Oneof（和类型）**：几个分支中恰好取一个，相当于 tagged union 或 Rust 的 enum。比如 `DTensorRange = RangeFromComputeView | RangeFromStorageView`。Python 实现里分支就是类本身，用 `type(x) is Foo` 判断是哪一支，不另设 tag 字段。
- **tagged 标量**：给一个整数贴上语义标签，比如 `WorldSize(4)`、`ConstantOne(1)`、`NumNodes(10)`。Python 实现是 `gdtensor/tag.py` 的 `Box` 子类，用来防止把"rank 数"和"张量数"这种都是整数的量混用。

### 5.2 Result 而不是异常

所有公开 API 都返回 `Result`：成功是 `Ok(value)`，失败是 `Err(error)`，error 是 `ConstraintViolationError`、`InvalidArgumentError`、`NotFoundError`、`CheckFailedError`、`GDTensorIOError` 之一，每个都带 `msg`（`gdtensor/internal/result.py`）。调用方必须检查返回值，测试里大量出现的 `assert type(result) is gb.Ok` 就是在做这件事。

内部实现为了写起来方便，仍然用异常：代码里 `raise Fail(err)`，函数外层的 `@catch` 装饰器把它转成 `Err(err)`；`must_ok(result)` 则反过来，取出 `Ok` 的值，遇到 `Err` 就抛 `Fail`。所以读代码时看到 `raise invalid(...)`，要知道它最终会变成调用方收到的一个 `Err`。

### 5.3 protobuf 风格的构造

所有数据类都按 protobuf 生成代码的约定提供 `new_*`、`get_*`、`set_*`、`add_*`、`mutable_*`、`clear_*` 方法和 `check_constraints()`。这些方法由 `gdtensor/internal/builder_mixin.py` 的 `make_builder_mixin` 根据字段清单生成为一个 mixin 基类，数据类再继承它：`class GlobalTensorMeta(GlobalTensorMetaBuilderMixin)`。实际代码里也可以直接用 dataclass 的构造函数，测试大多这么写。

`check_constraints()` 递归检查对象自身是否合规，失败时返回带约束名的 `ConstraintViolationError`；它不检查一次拷贝的区间是否不交、是否覆盖，那是内部 `check_*` 函数的事。

### 5.4 mesh 代数

mesh 是"参与者排成的网格"。gdtensor 不直接存一个整数数组，而是用几种构造组合出网格，这样能序列化，也能在任意规模下保持很小：

| 构造 | 作用 | 类比 NumPy |
|-|-|-|
| `ArrangedMesh(WorldSize(n))` | 0 到 n−1 的一维序列 | `np.arange(n)` |
| `LiteralMesh(ranks=[1, 3], max=WorldSize(8))` | 显式列出若干 rank | `np.array([1, 3])` |
| `ReshapedMesh(m, shape)` | 改变网格形状 | `m.reshape(shape)` |
| `SlicedMesh(m, slices)` | 取子网格，每维一个 `Slice` 或不切 | `m[0:1, :, 2:3]` |
| `PermutedMesh(m, perm)` | 交换网格维的顺序 | `m.transpose(perm)` |
| `SlotOuterProductMesh(tensor_slots, content_slots)` | 存储侧二维网格：张量行 × 内容列 | 外积 |
| `StCacheOuterProductMesh(temporal, spatial, concurrency)` | 缓存侧三维网格：step × 节点 × 缓冲槽 | 外积 |

模拟模型的计算 mesh 是这样拼出来的（`models.py` 的 `compute_mesh`）：先 `ArrangedMesh(WorldSize(32))`，reshape 成 `(PP, DP, TP) = (2, 4, 4)`，再用 `SlicedMesh` 把 PP 维切到当前 stage。存储侧的 slot 编号是 `t × W + w`，t 是张量行，w 是内容列，W 是列数（`gdtensor/internal/mesh.py` 开头注释）。

### 5.5 从切分到 CopyPlan

这是核心算法，搞懂它就搞懂了 IO 主路径。我认为这是整个代码库里最值得花一整天读懂的部分，缓存读链、统计函数和各种校验都建立在这四步之上。它分四步，全在 `gdtensor/internal/dtensor.py` 和 `plan.py` 里：

1. **参与者拥有一个盒子。** 根据 placements 和参与者在网格里的坐标，算出它拥有张量的哪一块：每个 `Shard(d)` 维按 `torch.chunk` 规则取一段，`Replicate` 维取全部。结果是一个多维的盒子（box），比如"第 0–1 行、第 0–3 列"。
2. **盒子变成区间。** 把盒子按行优先展平成全局元素下标上有序、不相交的区间列表。行切的盒子是一个连续区间，列切的盒子是每行一小段。
3. **求交集。** save 时，拿当前 rank 在计算侧的区间和每个存储 slot 的区间求交；load 时反过来，拿当前 rank 需要的区间和所有存储 slot 求交。每一块交集就是一次拷贝。
4. **换算地址。** 交集在计算侧用全局字节偏移表示（`RangeFromComputeView` 的 `global_tensor_offset` 是在整个逻辑张量里的偏移，不是本地 buffer 里的偏移，本地位置由 accessor 自己换算）；在存储侧用"该 slot 在文件里的起点 + 交集在 slot 内的前缀和"表示，也就是 `contiguous_payload_offset`。

save 还要去重：一个 `Replicate` 维上有多个副本时，只有一个坐标负责写，而且按张量顺序轮换（`plan.py` 开头注释："one coordinate writes, rotated by tensor order"），避免所有张量都压在同一个 DP 副本上写。load 没有去重的问题，每个 rank 要把自己拥有的每个元素都填满。

### 5.6 accessor 和 session

accessor 是裸字节读写接口 `DTensorDataAccessor`，只有两个必须实现的方法：`read_to_buffer(range, callback)` 和 `write_from_buffer(range, callback)`。回调的签名是 `(buffer_ptr, buffer_len) -> Result`，传的是一块内存的地址和长度。读的时候 accessor 把数据放进一块内存，再调用回调让对方取走；写的时候 accessor 准备好目标内存，调用回调让对方填进来。一个 range 可以分多次回调完成，读写两侧的分块大小可以不同，执行器会检查总长度对得上（`gdtensor/internal/api.py` 的 `_read`、`_write`）。

session 管文件的生命周期和偏移换算，不碰 payload。save session 是 `begin_save → 拷贝 → commit 或 abort`，load session 是 `begin_load → 拷贝 → finish`。`rank_save` 和 `rank_load` 是很薄的执行器（`api.py`）：生成 plan、校验 plan、打开 session、逐对拷贝、提交或关闭，出错时 abort 或 finish 后把错误原样返回。

## 6 一个最小例子（必需）

[tests/demo/test_walkthrough.py](../tests/demo/test_walkthrough.py) 用一个 3×4 的 I64 张量走完了 save 和换拓扑 load。元素值就是全局下标 0 到 11，每个元素 8 字节：

```text
        col0 col1 col2 col3
row0      0    1    2    3
row1      4    5    6    7
row2      8    9   10   11
```

计算侧有 2 个 rank，按 `Shard(1)` 切列：rank 0 持有第 0、1 列，也就是元素 0、1、4、5、8、9；rank 1 持有第 2、3 列。

存储侧有 2 个 slot，mesh 是 `SlotOuterProductMesh(ArrangedMesh(ConstantOne(1)), ArrangedMesh(WorldSize(2)))`，placements 是 `[Replicate(), Shard(0)]`，即按行切：slot 0 是第 0、1 行（元素 0–7），slot 1 是第 2 行（元素 8–11）。两个 slot 放在同一个文件里时，payload 区间分别是 `[0, 64)` 和 `[64, 96)`。

save 时，rank 0 的元素在行优先的文件里并不连续，`CopyPlan` 会给出三段：元素 0–1 写到 `[0, 16)`，元素 4–5 写到 `[32, 48)`，元素 8–9 写到 `[64, 80)`。按列切的张量写进按行展平的文件，必然产生这种跨步的小写入。

load 时换成 3 个 rank、按 `Shard(0)` 切行，每个 rank 正好分到一行：rank 0 读 `[0, 32)`，rank 1 读 `[32, 64)`，rank 2 读 `[64, 96)`，都是连续区间。

同一份文件，换一种切法读，IO 模式就从三段跨步的 16 字节变成一段连续的 32 字节。放大到真实模型，这个差别就是第 12 节写入碎片问题的来源。

测试随后把 3 个 rank 读到的数据再用 `rank_save` 写成另一份 checkpoint，检查文件字节与原始输入完全一致。这就是"换布局 = load 再 save"：设计第 9 节明确不支持存储到存储的直接复制，也不支持计算侧之间的直接换切分。

## 7 缓存读链（需要）

掌握到什么程度：能解释 spatial、temporal、concurrency 三个维度各自解决什么问题；知道读链的执行单位（segment）和它的生命周期；知道哪些地方设计与实现还没对齐。

### 7.1 为什么要缓存

同一份存储数据可能有几万个 rank 要读。设计文档第 3.4 节的例子：存储只有 1 份，计算侧有 24k 个 rank；中间放两级节点缓存，一级 10 个节点各从存储读一次，二级 100 个节点各从一个一级节点读，每个二级节点只服务 240 个 rank，是直连时的 1%。这和 CDN 的多级回源、镜像分发的 P2P 预热是一个思路。

### 7.2 三个维度

`CacheMesh` 是三维的：

- **spatial**（节点数）：放置方式固定为 `Replicate`，每个节点一份完整副本，层层放大副本数来分摊读者。
- **temporal**（`num_steps`）：放置方式固定为 `Shard`，把数据切成 S 段分批经过缓存，用时间换内存，S 越大单节点占用越小。
- **concurrency**（`concurrency_width`）：放置方式固定为 `Shard`，设计本意是多个缓冲槽，一个在填、一个在读，2 就是双缓冲。

节点和 rank 的换算由读链的 `num_ranks_per_node` 给出：节点 n 覆盖 rank `[n × num_ranks_per_node, (n + 1) × num_ranks_per_node)`。

### 7.3 执行模型

读链有四种：直连、一层缓存、两层缓存，以及按张量分段组合的 mixed。执行时，`gdtensor/internal/propagation.py` 先为整条链生成一份节点级的传播计划，内存统计、每个 rank 的拷贝计划和执行器都从这一份计划推出来，保证三者口径一致。它的术语值得记住（摘自该文件开头注释）：

| 术语 | 含义 |
|-|-|
| layer | 链上的一个缓存 view，layer 0 从存储读 |
| parent | 每个节点从上一层恰好一个节点读；同时出现在两层的节点本地读，其余按轮询分配 |
| demand | 一个节点必须持有的元素：末层是它服务的 rank 需要的并集，父层是子节点需要的并集 |
| cell | 一个 (step, 缓冲槽) 的盒子 |
| segment | cell 与 demand 相交后的一段连续字节，是填充、驻留和释放的单位 |
| budget | 每个 (layer, 节点) 的内存上限，执行器只在驻留字节不超预算时才允许填充新 segment |
| cohort | 读同一个 (layer, 节点) 的一组 rank；预算装不下全部 demand 时，它们必须同时在场 |

跨 rank 的协调状态默认放在一个文件里：真实部署是每个 rank 一个进程，进程之间看不到彼此的内存，所以默认的协调器 `_FileSlotTable` 把计划状态存成一个 pickle 文件，用 `fcntl.flock` 加锁，任何构建出同一条链（内容指纹相同）的进程共享它（`gdtensor/internal/api.py` 中 `_COORD_ENV` 附近的注释）。测试里可以用 `GDTENSOR_CACHE_COORDINATOR=thread` 换成进程内的实现。

一个 segment 的生命周期是：填充 → 所有消费者（rank 或下一层的 segment）读完并 ACK → 释放。消费者缺席时报错而不是挂起，等待时间由 `GDTENSOR_CACHE_WAIT_SECONDS` 控制（默认 60 秒）。这套状态机在 `formal/tla/CachePropagation.tla` 里有一个有界模型。

### 7.4 还没对齐的地方

设计第 3.4 节说 concurrency 维是把同一个 step 的数据切成几段，第 5.8 节又说多一个缓冲槽多占一份，两种说法给出的内存统计相差 `concurrency_width` 倍；当前代码按前者实现。缓存执行器目前还是串行的，"填一个槽的同时读另一个槽"尚未实现。这些都记录在 `docs/decisions-2026-10-27.md`，需要设计所有者拍板。读这一节代码时，以 `propagation.py` 的注释为准理解现状，以设计文档为准理解目标。

## 8 存储知识在这个项目里的具体用法（需要）

这些你都懂，这里只说这个项目怎么用它们，以及哪里有它自己的规矩。

**原子发布。** safetensors 的 save session（`formats.py` 开头注释）：每个 rank 以 `O_CREAT` 不截断的方式打开自己要写的临时文件，并把文件设成最终大小，这个操作是幂等的，多个 rank 对同一个文件做也没问题；文件的 leader rank 写 header，header 字节是 `(DTensorTable, PhysicalLayoutTable)` 的纯函数，任何 rank 算出来都一样；commit 前所有 rank 过一次 barrier 并投票，全票通过后每个文件的 leader 负责发布，最后由一个 rank 原子发布 `.gdtensor_committed`。投票失败则全部丢弃。

**commit 结果未知。** session 的状态机是 `open → committing → committed | unknown | aborted`。commit 过程中出错、无法判断是否已经发布时进入 unknown，此时 `abort` 不能删除任何东西，要保留现场。设计文档第 4.2 节把这条写成了硬性要求。

**同一个 root 不能并发 save。** 需要避免新旧分片混合时，写到新目录，不要原地覆盖已提交的 checkpoint。

**版本固定。** 对象存储接入（`SdkFileIO`）要求 `metadata(key)` 返回一个唯一标识不可变内容的版本号，每个 rank 第一次读某个分片时固定这个版本，之后的范围读都显式带上它。ETag 只有在 SDK 明确保证它是不可变内容版本时才能用。这和缓存一致性密切相关：缓存的身份（`cache_identity`）也要包含这个版本，源文件一变，旧缓存就必须失效。

**IO 后端协议。** README 的"文件系统 SDK 接入"一节列了 `FileSystemClient` 的八个方法和语义（`metadata`、`read_range`、`create_staging`、`write_range`、`sync`、`publish`、`discard`、`close_rank`），这是你最可能第一个动手改的接口。它是结构化协议（Python 的 `Protocol`），client 不需要继承基类，方法签名对上即可。

**IO 模式观测。** `gdtensor_conn/iobackend.py` 的 `IoTracer` 记录每一次 pread/pwrite 的 rank、文件、偏移、大小和时间；模拟 CLI 把它汇总成 `report.json` 的直方图，并导出 `read_trace.csv`、`write_trace.csv`。分析写放大、对齐和并发写者数，从这里入手。

## 9 工程工具链（需要）

### 9.1 Python 3.12

项目要求 Python 3.12 以上，用到了几个较新的语法，第一次见可能看不懂：

- `type Result[T] = Ok[T] | Err[Error]`：PEP 695 的类型别名语法（`gdtensor/__init__.py`）。
- `X | Y`：联合类型，等价于 `Union[X, Y]`。
- `typing.Protocol`：结构化子类型，只要方法签名对上就算实现了接口，用于 `FileSystemClient`。
- `abc.ABC` 和 `@abstractmethod`：用于 session 和 accessor 这种必须继承实现的接口。
- `@dataclass`：几乎所有数据类都是 dataclass。

并发方面，测试里大量用 `concurrent.futures.ThreadPoolExecutor` 和 `threading.Barrier` 模拟多个 rank，少数用 `multiprocessing` 的 fork 模式。

### 9.2 NumPy

测试和 connector 都重度依赖 NumPy，至少要会这些：

- `np.arange(12).reshape(3, 4)`：造测试数据，值等于下标。
- 切片 `a[0:2, :]` 和 `ravel()`：前者通常返回视图（不复制），后者展平。
- `a.view(np.uint8)`：把任意 dtype 的数组当成原始字节看，不复制；`a.tobytes()`、`np.frombuffer(b, dtype=np.int64)` 在字节和数组之间来回转换。
- `np.searchsorted`、`np.concatenate`：区间运算里用来查找和拼接。

### 9.3 uv、pytest 和静态检查

- **uv**：Python 的包管理和虚拟环境工具。`uv sync --locked` 按 `uv.lock` 精确安装依赖，`uv run <命令>` 在项目虚拟环境里执行命令。`--extra torch` 装 PyPI 默认的 torch，`--extra torch-cpu` 在 Linux 上装 CPU 版 torch，两者互斥。
- **pytest**：`-s` 显示 print 输出（测试的日志都靠 print），`-k 关键字` 选用例，`-v` 列出每个用例名，`-x` 第一个失败就停。`@pytest.mark.parametrize` 把一个测试展开成多个用例，`tmp_path` 是 pytest 提供的临时目录。
- **ruff**：格式化和 lint。项目只开了 `F` 类规则（未定义名、未使用导入等）。
- **ty 和 pyrefly**：两个类型检查器，CI 要求两个都零警告，需要装 torch extra 才能完整检查。
- **缩进规则**：`precommit.sh` 第 5 步会检查 core、connector 和 tests 的代码块最多三层嵌套（括号内换行不算），超了会失败。这是项目自己的规矩，写代码时要注意。

### 9.4 项目用到的环境变量

| 变量 | 作用 |
|-|-|
| `GDTENSOR_TORCH_TESTS=1` | 打开需要真实 torch 的测试，否则它们被跳过 |
| `GDTENSOR_CACHE_COORDINATOR` | 缓存协调器：`file`（默认，跨进程）或 `thread`（进程内） |
| `GDTENSOR_CACHE_DIR` | 文件协调器存放状态文件的目录 |
| `GDTENSOR_CACHE_WAIT_SECONDS` | 等待缓存消费者到齐的超时，默认 60 秒 |
| `GDTENSOR_REMOTE_HOST` / `_USER` / `_ROOT` / `_PROXY` | 远端验证主机的 SSH 参数，主机默认 `digest` |
| `GDTENSOR_REMOTE_TEST=1` / `GDTENSOR_FORCE_REMOTE=1` | 前者表示"已经在远端，直接跑"，在 macOS 上可以借它本机运行检查脚本；后者在 Linux 上也强制转发到远端 |
| `GDTENSOR_ADVANCED_FORMAL=1` | 让 `precommit.sh` 额外跑 Lean 和 TLC |
| `GDTENSOR_STRESS_*` | 随机压力测试的种子和规模 |

### 9.5 提交流程和 CI

- 提交信息用 conventional commits 风格：`feat(sim): ...`、`fix(cache): ...`、`test: ...`、`docs: ...`，看 `git log` 即可。
- `git config core.hooksPath .githooks` 启用 pre-commit 钩子，它调用 `scripts/precommit.sh`。
- 在 macOS 上，`scripts/precommit.sh` 和 `scripts/full_check.sh` 不在本机跑，而是自动把暂存树打包，通过 `ssh digest` 发到一台私有主机上验证（`scripts/remote_gate.sh`）。只有暂存的修改会被送去测试。没有这台主机的权限时，这两个脚本和 pre-commit 钩子都会失败，可以用 `GDTENSOR_REMOTE_TEST=1 bash scripts/precommit.sh` 让脚本以为自己已经在远端，直接在本机跑。我在本机这样跑过 precommit：格式、lint、编译、导入四步通过，pytest 部分 1113 个通过、1 个跳过，用时 7 分 53 秒。注意它用的是 `--extra torch-cpu`，会把 `.venv` 里的 torch extra 换掉，之后再跑 `--extra torch` 的命令时 uv 会再换回来。正式提交前是否必须过 digest，问一下维护者。
- GitHub Actions（`.github/workflows/ci.yml`）在 Linux 上直接跑锁定环境，分四个任务：Python 3.12/3.13 的格式、lint、非 torch 测试和行为变异；CPU torch 的类型检查和 LAWA；Lean 证明和 TLC 模型检查；以及汇总的 CI gate。
- 有两处钉死了源文件的 sha256：`tests/mutation_check.py` 里的 `FROZEN_API_SHA256` 钉住 `gdtensor/internal/api.py`，`formal/tla/MAPPING.md` 钉住 `api.py` 和 `propagation.py`（由 `scripts/formal_gate.sh` 检查）。改这两个文件时要同步更新这些哈希，否则变异测试和形式化检查会直接失败。

## 10 跑起来（必需）

下面的命令我在本机（macOS 14、Apple Silicon）实际跑过，耗时供参考。

### 10.1 安装

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv 官方预编译安装包，装到 ~/.local/bin
uv sync --locked --extra torch                    # 约 35 秒，uv 会自动下载 Python 3.12
```

不要用 `brew install uv`：我这台机器上 Homebrew 选择了从源码编译，开始下载 rustc 和 LLVM 的源码包，等不起。只跑非 torch 测试时可以去掉 `--extra torch`。

### 10.2 第一个要跑的测试

```bash
uv run --locked python -m pytest -s tests/demo/test_walkthrough.py   # 约 15 秒，2 个用例
```

它对 safetensors 和 distcp 各跑一遍第 6 节的例子。safetensors 那一遍的关键输出如下（测试夹具写的 header 里 `writer` 是 `test-fixture`，不是 connector 的 `gdtensor`）：

```text
source rank=0 indices=[0, 1, 4, 5, 8, 9] values=[0, 1, 4, 5, 8, 9]
source rank=1 indices=[2, 3, 6, 7, 10, 11] values=[2, 3, 6, 7, 10, 11]
header={'__metadata__': {'writer': 'test-fixture'}, 'weight': {'dtype': 'I64', 'shape': [3, 4], 'data_offsets': [0, 96]}}
payload_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
crc32 expected=a8e85d54 actual=a8e85d54 bytes=96
loaded rank=0 indices=[0, 1, 2, 3] values=[0, 1, 2, 3]
loaded rank=1 indices=[4, 5, 6, 7] values=[4, 5, 6, 7]
loaded rank=2 indices=[8, 9, 10, 11] values=[8, 9, 10, 11]
```

两个 slot 拼起来正好是完整张量，所以 header 里只有一个 `weight` 条目，`data_offsets` 覆盖全部 96 字节。想直接看 `CopyPlan`，可以调用内部的 `gdtensor.internal.plan.d_tensors_view_rank_plan_save`（只在实验脚本里这么做，测试代码禁止导入 internal）。我用同样的输入打印出 rank 0 的 save plan，正是第 6 节手算的三段：

```text
RangeFromComputeView(weight, global_tensor_offset=0,  size=16) -> RangeFromStorageView(model.safetensors, contiguous_payload_offset=0,  size=16)
RangeFromComputeView(weight, global_tensor_offset=32, size=16) -> RangeFromStorageView(model.safetensors, contiguous_payload_offset=32, size=16)
RangeFromComputeView(weight, global_tensor_offset=64, size=16) -> RangeFromStorageView(model.safetensors, contiguous_payload_offset=64, size=16)
```

load 一侧三个 rank 各是一对 32 字节的连续区间。

### 10.3 模拟 CLI

```bash
uv run --locked python -m gdtensor_conn.sim --root /tmp/gdtensor-ck \
  --hidden 1024 --ffn 2816 --vocab 8192 \
  --tp 4 --dp 4 --pp 2 --load-tp 2 --load-dp 8 --load-pp 2 \
  --layout hf --format safetensors                 # 约 20 秒
```

它用一个玩具 transformer 按 TP4×DP4×PP2 共 32 个线程 rank 写一份 checkpoint，再换成 TP2×DP8×PP2 读回来。`--root` 目录会被重建，报告和 trace 在 `/tmp/gdtensor-ck_report/`：`report.json` 是汇总，`write_trace.csv`、`read_trace.csv` 是每一次 pwrite/pread 的记录（时间、rank、文件、偏移、大小）。第 12 节的数字就来自这次运行。

仓库 `examples/` 下的两份报告是旧版 CLI 生成的，字段和当前输出不同（旧版有 L1/L2/L3 分层读统计），对照时以自己跑出来的为准。

### 10.4 更大范围的测试

```bash
uv run --locked python -m pytest tests/core tests/io tests/checkpoint tests/sim tests/demo -q
GDTENSOR_TORCH_TESTS=1 uv run --locked --extra torch python -m pytest tests/torch -q
```

第一条在本机用时 9 分 39 秒，结果是 1852 个通过、6 个跳过，另有 885 条警告，几乎都是 torch DCP 在单进程模式下的提示，可以忽略。第一次跑建议加 `-x`，遇到失败立刻停下。

第二条要启动真实的多进程 rank（最多 32 个），本机用时约 25 分钟，60 个用例里有 3 个失败：`test_gloo_coordinator.py` 和 `test_gloo_checkpoint_matrix.py` 的两组参数。这三个都用 `torchrun` 启动进程，失败原因是本机主机名解析不了（`gai error: 8 - nodename nor servname provided, or not known`），和项目代码无关：给同一个 Gloo 示例加上 `--local-addr=127.0.0.1` 后 5 秒通过。macOS 上主机名不在 `/etc/hosts` 里时很常见，可以先自查：

```bash
python3 -c 'import socket; print(socket.getaddrinfo(socket.gethostname(), 0)[0])'
```

报 `gaierror` 就把主机名加进 `/etc/hosts`（例如 `127.0.0.1 <你的主机名>`，需要 sudo），再重跑这三个测试。不修的话，README 里那条单机 `torchrun --standalone` 命令会一直卡住。

## 11 形式化方法（选修）

`formal/` 下有两类东西。TLA+ 是一种描述并发系统状态机的规约语言，TLC 是它的模型检查器，会穷举一个小规模实例的所有交错执行，检查不变量是否始终成立；这里有拷贝计划、原子发布、缓存槽生命周期和缓存传播四个模型。Lean 4 是一个定理证明器，这里用它证明了几条区间和行优先换算的性质。

需要知道的边界是：这些都是抽象模型，不证明 Python 实现正确。`formal/tla/MAPPING.md` 写明，只有 `CachePropagation.tla` 和当前运行时有逐字段的对应关系，其余是旧版设计的模型，作为数学回归保留。刚上手时不必读，等你要改缓存执行器或提交协议时再看对应模型。

## 12 你的背景最能发力的地方

下面几处问题来自仓库自己的记录和我在本机的复现，没有一处需要训练知识，全部是存储问题。

### 12.1 写入碎片

第 10.3 节那次模拟运行（hidden 1024、ffn 2816、BF16，TP=4、DP=4、PP=2）写出 2 个文件共 136,337,704 字节，用了 32,867 次 pwrite，99.73% 小于 64 KiB，单个文件最多有 17 个并发写者。按 `write_trace.csv` 统计大小分布，其中 16,384 次是 1408 字节，另有 16,384 次是 512 字节，两者占了 99.7%。1408 字节是 `down_proj`（形状 `[1024, 2816]`，按 `Shard(1)` 切成 4 份）一行里一个 TP 分片：704 个 BF16 元素；512 字节是 `o_proj`（`[1024, 1024]`，同样按 `Shard(1)` 切）一行里的 256 个元素。每种都是 1024 行 × 4 个 TP rank × 4 层 = 16,384 次。这就是第 6 节那个跨步写入的真实版本。本地 POSIX 盘可以靠 page cache 吸收，换成对象存储或远端文件系统 SDK，这种 IO 模式会直接变成请求数瓶颈。

模拟 CLI 有一个 `--aggregate-tp` 开关，帮助文字说它让 TP 组内的一个 rank 整块写出按第 1 维切的张量。我加上它重跑了一次，pwrite 次数、中位大小和碎片比例和不加时完全相同。原因是这个开关只改了存储表（`down_proj` 的存储放置从 TP 维 `Shard(0)` 变成由一个 TP 坐标持有整块），而 save plan 仍按计算侧的持有关系决定谁写什么，每个 TP rank 照样把自己的跨步碎片写进那个 slot。这是一处说明和行为不符的地方，适合作为第一个调查题目：先弄清是 plan 的设计如此（设计文档说业务层不做聚合），还是开关本身没接完。写端聚合、按 rank 重排还是换一种文件布局，哪种最划算，是一个典型的存储权衡。

### 12.2 读热点

同一次运行的 load 一侧换成 TP=2、DP=8：pread 共 1,090,920,576 字节，而文件只有 136,337,704 字节，报告里的读放大是 8.0，因为 8 个 DP 副本各自从文件读了同一份权重。32 个 rank 时这只是 8 倍，换成几万个 rank，就是设计文档第 3.4 节说的读热点。缓存读链要做的，就是让存储只被读一次，其余的副本在计算侧逐级扇出。

### 12.3 两层缓存的冷读

在 100 个线程 rank、Qwen3-0.6B 的测试里，热轮（缓存已填好）一层 27.4 秒、两层 29.1 秒；但两层缓存的冷轮要 661 秒，直连只要 204 秒（`artifacts/cache-fix-notes.md`）。笔记自己的结论是计划构建和两层冷填充仍是瓶颈。这类"填充流水线怎么组织"的问题，做过分布式缓存或镜像预热的人会很有直觉。

### 12.4 SDK 后端的语义边界

README 写明的限制：`SdkFileIO` 目前只覆盖 safetensors；client 必须支持 range write，只支持整对象 PUT 的对象存储需要在外部先拼好整个 shard；节点缓存仍写本地盘。设计第 9 节把对象存储 multipart、重试和分布式 barrier 明确划给 session 和实现层，核心不管。对象存储接入这一层几乎完全是空白，我认为这是你最该先啃的部分。

### 12.5 多机验证

现有的 torchrun 测试是同一主机上的两个 agent，README 和测试文档都写明"不是物理双机证据"；性能数字来自单主机的线程 rank，不是多物理机、GPU 或 NCCL。

## 13 测试方法论（需要）

这个项目的测试写法有自己的一套规矩，其中一部分由测试自己强制执行（比如禁止导入内部模块），改代码前先理解它们，新写的测试才能和现有体系对得上。

- **独立 oracle。** 期望值不能由被测代码算出来。测试用 NumPy 按 `torch.chunk` 规则独立切出每个 rank 的元素下标（`tests/support/oracle.py` 的 `slice_points`），再独立解析文件，按全局元素下标重建完整张量，要求每个元素恰好出现一次。
- **测试不导入内部模块。** `tests/core/test_api_conformance.py` 会递归检查测试和 connector 都没有导入 `gdtensor.internal`；测试文件之间也互不导入，共享的东西放在 `tests/support`。
- **CRC 用法有规矩。** 基线必须在操作前保存，或由独立 oracle 生成，不能对同一份输出算两次 CRC 再比较（`tests/README.md`）。
- **故意损坏。** 很多测试有一个负例：交换两个等长、内容不同的 payload 区域，要求 CRC 和读回值检查都能发现。
- **行为变异测试。** `tests/mutation_check.py` 在临时副本里故意改坏实现，要求至少一个测试失败，以证明测试真的能抓到这类错误。
- **线程 rank 和真实进程。** 文件测试大多用线程模拟 rank，真实 torch 多进程测试在 `tests/torch/` 下，需要 `GDTENSOR_TORCH_TESTS=1`。
- **日志格式。** 测试的 print 以 `[CRC+ARANGE]` 等标签开头，每条 `PASS` 都在对应断言通过之后才输出。

## 14 自测题

答案在每题下方，先自己想再看。

1. 一个 `GlobalTensorMeta([3, 4, 5], "F32", 4)` 的张量，元素 `(1, 2, 3)` 的字节偏移是多少？
   答：`((1 × 4 + 2) × 5 + 3) × 4 = 132`。
2. 一个 `[10, 6]` 的张量按 `Shard(0)` 切给 4 个 rank，每个 rank 分到几行？
   答：每份 `ceil(10/4) = 3` 行，rank 0–2 各 3 行，rank 3 只有 1 行。
3. 同一个张量按 `Shard(1)` 切给 4 个 rank，rank 3 分到什么？它在行优先文件里对应几段区间？
   答：每份 `ceil(6/4) = 2` 列，rank 0–2 各 2 列，rank 3 是空分片，0 段区间、0 字节。rank 0–2 各对应 10 段（每行一段）。
4. TP=2、DP=4、PP=2 时，rank 11 的 `(pp, dp, tp)` 是什么？
   答：`tp = 11 % 2 = 1`，`dp = 11 // 2 % 4 = 1`，`pp = 11 // 8 = 1`。
5. 为什么 `o_proj` 是 `Shard(1)` 而 `q_proj` 是 `Shard(0)`？
   答：Megatron 的 TP 把注意力和前馈的输入投影按输出维切（列并行），输出投影按输入维切（行并行），两者配对后每个块只需一次跨卡求和。PyTorch 权重形状是 `[out, in]`，所以输出维是第 0 维、输入维是第 1 维。
6. 一个 safetensors 文件前 8 字节是 `0x58 00 00 00 00 00 00 00`，第一个张量的 `data_offsets` 是 `[0, 64]`，它在文件里的字节范围是什么？
   答：header 长度 0x58 = 88，payload 从 `8 + 88 = 96` 开始，张量在 `[96, 160)`。
7. save 时一个张量在 DP 维上有 4 个副本，会写几次？谁来写？
   答：写一次。`Replicate` 维去重，由一个坐标负责，按张量顺序轮换。
8. 为什么核心没有"把 checkpoint A 直接转成布局 B"的 API？要做这件事怎么办？
   答：设计第 9 节裁掉了存储到存储的复制和计算侧之间的换切分。做法是 `rank_load` 到一个计算 view，再 `rank_save` 到新布局。
9. commit 过程中网络断开，session 进入 unknown 状态，这时调用 `abort` 应该删除临时文件吗？
   答：不应该。结果未知时可能已经部分发布，必须保留现场。
10. 缓存 mesh 的 spatial 维为什么只能是 `Replicate`？
    答：spatial 的目的是放大副本数来分摊读热点，每个节点都要有完整的一份；切分数据片是 temporal 和 concurrency 的事。
11. 为什么不能用 ETag 当作版本号，除非 SDK 保证？
    答：ETag 的语义由各家实现决定：有的是内容摘要，有的随加密方式或分段上传方式变化，有的不承诺和内容一一对应。读端固定版本后，之后每次范围读和缓存身份检查都要指向同一份字节；只有 SDK 保证"同一个 ETag 一定对应同一份不可变内容"时，才能拿它当版本用。
12. 测试为什么偏爱 `arange` 作为张量的值？
    答：值等于全局下标，读回的数据错位、重复或丢失时，打印出来一眼就能看出错在哪里；再配合独立实现的 oracle 和 CRC，能区分是偏移错、张量错还是分片次序错。

## 15 建议的阅读顺序

**第一天：把问题和词汇对上。** 读本文第 1–4 节，再读设计文档第 1、2 节和附录 A 术语表。不要从附录 B 的类型定义开始，那部分是给实现者查的。

**第二天：逻辑层和物理层。** 读本文第 5 节、设计文档第 3、4 节和第 5.3 节的偏移算例。然后读 [gdtensor_conn/models.py](../gdtensor_conn/models.py) 前 330 行，看每个权重被定成哪种 placement、存储表怎么从计算表推出来。

**第三天：跑起来，看字节。** 按第 10 节跑 walkthrough，对着打印的 indices 和 values 验证第 6 节的手算；跑模拟 CLI，带和不带 `--aggregate-tp` 各一次，比较 `write_trace.csv`。做一遍第 14 节自测题。

**第四天：执行层和一致性。** 读设计文档第 5.1、5.2、5.4 节，再读 `gdtensor/internal/plan.py`、`gdtensor/internal/api.py` 前 215 行；读 `gdtensor_conn/formats.py` 开头注释、README 的"文件系统 SDK 接入"和 `gdtensor_conn/iobackend.py`。

**第五天：缓存读链和未决问题。** 读本文第 7 节、设计文档第 3.4、5.8 节、`gdtensor/internal/propagation.py` 开头注释，再读 `docs/decisions-2026-10-27.md` 的"缓存容量口径"和 `artifacts/cache-fix-notes.md`。

LAWA 本身（设计第 10 节）只是业务例子，读一遍知道"读、算、写"三步怎么串起来即可。

## 16 外部资料

按优先级排，前三项建议在开始前读完：

1. **PyTorch DTensor 文档**（`torch.distributed.tensor`）：看 `DeviceMesh`、`Shard`、`Replicate` 的定义和示例。
2. **safetensors 格式说明**（huggingface/safetensors 仓库的 README）：和第 4.1 节对照着看。
3. **PyTorch Distributed Checkpoint 文档**：理解每个 rank 写自己的文件、`.metadata` 最后写。
4. **Megatron-LM 论文**（Shoeybi 等，2019）：只看张量并行一节的列切、行切示意图。
5. **ZeRO 论文**（Rajbhandari 等，2019）：只看三个 stage 分别切了什么。
6. **torchrun 文档**（`torch.distributed.run`）：环境变量和 rendezvous，读 torch 测试前看。
7. **TLA+**：Leslie Lamport 的 *Specifying Systems* 或官网的入门视频，只在需要读 `formal/` 时看。

## 17 常用术语速查

完整术语表在设计文档附录 A，下面只列最常碰到的：

| 术语 | 一句话 |
|-|-|
| tensor / 张量 | 有形状和元素类型的多维数组，落盘时按行优先展平 |
| rank / world size / local rank | 全局进程编号、进程总数、机内编号 |
| mesh | 参与者排成的网格；计算侧是 rank，存储侧是 slot，缓存侧是节点 |
| placement | `Shard(d)` 沿第 d 维切开，`Replicate()` 每人一份，`Partial()` 不能落盘 |
| box / interval | 一个参与者拥有的多维块，以及它展平后的元素区间 |
| PP stage | 流水线并行的一段，持有一部分层 |
| expert | MoE 模型里一个结构相同的前馈子网络 |
| ZeRO / 分布式优化器 | 把优化器状态在 DP 组里打散，每人存一片 |
| checkpoint | 某一时刻全部张量（含优化器状态）的落盘快照 |
| LAWA | 对最近 k 个 checkpoint 的权重做逐元素平均 |
| safetensors / DCP | 两种文件格式：前者 header 在文件头，后者按 rank 写 payload 再写 `.metadata` |
| view | 张量清单加逻辑表加 accessor 的句柄，存储侧还多布局表和 session |
| slot | `张量数 × world size` 表中的一格，可以为空 |
| `CopyPlan` | 源区间到目标区间的列表，plan 函数算出，执行器照做 |
| session | 文件生命周期和偏移换算；save 用 commit/abort，load 用 finish |
| accessor | 只读写裸字节的接口，通过 `(地址, 长度)` 回调交换数据 |
| 读链 | 带零到两层缓存的 load 数据流：storage → cache → compute |
| segment | 读链执行时填充、驻留、释放的最小单位 |
