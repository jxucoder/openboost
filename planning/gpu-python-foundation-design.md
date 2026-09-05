# GPU Python boosting foundation：设计草案

状态：P0–P3、P4.1–P4.3 已完成；下一项 P4.4 builder assembly，完整 GPU 扩展训练仍待实现。日期：2026-09-05。

本文件定义接口、边界、验证方法与执行顺序，供 medium 模型实施。
已完成阶段的实际行为与证据见执行清单；P2 基线不代表新扩展 API 的 GPU 路径已验证。
执行任务见 [执行清单](gpu-python-foundation-execution.md)。

## 1. 要验证的产品假设

> 一个 Python 研究者可以在独立包中改变 boosting 的目标、建树行为或更新规则，
> 复用 OpenBoost 的 CPU 参考实现、GPU 数据通路、预测与验证工具，而不维护自己的 fork。

这是有待验证的研究基础设施方向。Python 实现比例、GPU 标签和 API 数量不作为成功指标。
现有 [产品使命与证据规则](../AGENTS.md) 继续有效；NaturalBoost 是首个真实使用方，
本次不把整个仓库改名或重新宣传为通用替代品。

第一轮投资控制在建议的 6–8 周探索窗口内，按技术关卡推进；这是范围约束，非工期承诺。
先做到一个窄路径可信且可改，再决定是否扩展到更多树结构和算法。

| 目标 | 本轮可验证的证据 | 尚不能由此推导的结论 |
|---|---|---|
| Impact | 三类扩展实际改变训练结果，有独立数学/路由参考 | 算法新颖性、论文影响力 |
| Adoption | 两个独立 wheel 只依赖公开实验接口，能在干净环境运行 | 自编示例不能证明外部 adoption |
| Value | 实现工作量、端到端时间、质量、显存与迁移阻碍记录 | 未做访谈，不能推导付费意愿 |

继续扩大投入的产品关卡：至少两位外部开发者尝试扩展，其中一位在不改 core 的情况下
完成自己的方法；记录实际所需帮助和失败原因。联系与发送邀请需要用户另行授权，
执行模型先准备可运行材料，不自行发消息。软件技术验收不冒充这个产品关卡。

## 2. 起点：先整合已有工作

设计分支：`codex/gpu-python-foundation-design`，从本地 `main` 的
`82cf1e25b21a69093e85a270af7eb93c9ae7aa19` 创建。

2026-09-05 fetch 后，远端基线固定为
`6ebe3a8ced0e621b17e3cf63e31721af58471053`。
共同祖先 `504fdd0bfc60e5d8e7518250e087fb7e4766d1b4`；本地独有 12 个提交，
远端独有 9 个提交。此数字指创建设计分支时，不包括后续设计提交。

远端已有 `_trainer.py`、`_objectives.py`、FormulaBoost、WeibullAFT，必须复用。
本地有 categorical persistence/cardinality 修复、batch fail-fast、ScoringBench、
性能 CI 与 AGENTS/learnings，必须保留。不可用 reset 覆盖任一边，也不重新写一个训练循环。

只读 `git merge-tree` 发现文本冲突位于 `CLAUDE.md`、
`docs/getting-started/gpu-setup.md` 和 `docs/getting-started/installation.md`。
这不是完整的语义冲突清单；自动合并成功也必须检查 persistence、CUDA eligibility 与文档证据。
实际合并留给执行阶段 P0。

| 已核对的调用路径 | 设计后果 |
|---|---|
| 远端 `fit_boosting` 已按 channel 调 objective，但直接选择 native tree / `fit_tree` | 在此增加 builder 与 schedule 参数；显式扩展必须参与 dispatch |
| 远端 `TrainerConfig` 无 seed；现有 growth 存在全局 `numpy.random` 调用 | 将 seed/RNG 沿实际选中的路径传递，不能只给配置加字段 |
| 可扩展 primitive 把 histogram / sample node IDs 下载；`compute_leaf_values_gpu` 委托 CPU | 新增设备批量表示和设备 leaf reduction，不能把现有接口改名后宣称驻留 GPU |
| native builder 可原地更新 raw，旧 tree facade 以 host arrays 为持久化来源 | 明确 raw 更新归 trainer；树完成后允许小规模结构下载 |
| 远端以 distribution 类名选择 device kernel，并捕获宽泛异常后回落 CPU | 明确能力声明、严格模式与可见回落；禁止按名字误选内置实现 |
| 远端 `unit_hessian` 与 `sample_weight` 分开决定 | 非均匀权重可能被 native `const_hess=1` 覆盖，先复现再修复 |
| eval/callback 目前会触发 host raw 或预测下载 | 将其列为单独能力和成本，不声称所有训练配置全程驻留 GPU |

以上为源码检查；疑似 weighted-Hessian 问题尚未经 GPU 复现。
旧扩展测试本地基线为 **35 passed, 1 skipped**，不能证明上述 GPU 路径正确。
远端原设计中的速度数字不作为本设计已验证的依据。

## 3. MVP 支持边界

新增接口在 `openboost.experimental` 下；初期只在同一明确版本内承诺契约。
现有模型默认行为保留，NaturalBoost 接入并承担回归测试；暂不迁移所有模型。

| 项目 | MVP 决定 |
|---|---|
| GPU | 单 NVIDIA GPU，CUDA 12，Numba/CuPy RawKernel 内核；扩展数组使用 CuPy |
| CPU | NumPy oracle 与可运行的实验引擎；不要求与 GPU 位级一致 |
| 数据 | 稠密数值特征；sample-major 输入，bin 后 feature-major；一维 y，多 channel raw |
| 树 | level-wise、标量叶、每 channel 每 round 一棵树；深度至多 8 |
| 目标 | 任意显式提供 CPU/CUDA 实现的多 channel objective，首个验收是两参数 Normal |
| 权重 | 有限非负 sample weight，总权重大于零；只施加一次，包括零权重 |
| 采样 | 实验 GPU 首版仅 subsample=colsample=1；其他值明确拒绝，不默默忽略 |
| 正则 | 实验 GPU 首版 L2，reg_alpha=0；min_child_weight 与 min_gain 按契约执行 |
| 缺失/类别 | 首版实验 builder 明确拒绝；旧模型已有支持继续测试，类别上限修复不得丢失 |
| 元数据 | 首版 experimental fit 不接受 exposure/censoring；传入即明确拒绝，旧模型按原能力处理 |
| callbacks/eval | CPU 可先接现有流程；严格 GPU 首版仅无 callbacks/eval 的训练驻留声明；有评估的混合路径显式报告 |
| 预测/保存 | CPU raw prediction 必须可用；GPU prediction、GPU 保存后 CPU 加载须验证 |
| 不做 | Ray、多 GPU、out-of-core、GOSS、train-many、vector leaf、autodiff 框架互通、任意动态训练图 |

“纯 Python”指用户和维护者以 Python/CuPy/Numba 修改算法；不会承诺无需编译、
无需 CUDA runtime 或任意 Python 函数自动变成 GPU 内核。

## 4. 接口与训练语义

### 4.1 一个循环，三个显式扩展点

```text
NaturalBoost 等现有 facade       experimental.Booster
              \                 /
                 fit_boosting
                      |
        Objective.step(raw at round start)
                      |
           TreeBuilder.build per channel
                      |
       trainer applies StepSchedule coefficients
                      |
        tree storage / eval / persistence / report
```

以下是目标契约，不是现在可执行的 API。名字在 P3 确认后固定；不建立插件发现系统，
直接传对象即可。现有 Objective 由薄 adapter 对接，避免改动每个分布的数学实现。

```python
class Objective:
    channel_names: tuple[str, ...]
    supported_devices: frozenset[str]  # e.g. {"cpu", "cuda"}

    def init_raw(self, y, sample_weight=None, extra=None): ...
    # CPU initialization once; returns {channel: finite scalar}.

    def step(self, raw, y, sample_weight=None, extra=None, *, context): ...
    # Returns {channel: (grad, hess)} on the same device as raw.

    def loss_value(self, raw, y, sample_weight=None, extra=None, *, context): ...
    def constrain(self, raw, extra=None): ...

class TreeBuilder:
    supported_devices: frozenset[str]

    def build(self, binned, grad, hess, *, config, context): ...
    # Returns BuiltTree(tree, train_prediction=None).

class StepSchedule:
    def coefficients(self, round_idx, channel_names, base_learning_rate): ...
    # Returns {channel: finite nonnegative scalar}; full coefficient, not multiplier.
```

`experimental.Booster(objective=..., tree_builder=..., step_schedule=..., config=...,
device="cpu"|"cuda", fallback="error"|"warn")` 是现有 trainer 的薄 facade。
提供 `fit(X,y,sample_weight=None,eval_sets=None,callbacks=None,early_stopping_rounds=None)`、
`predict_raw(X)`、`save(path)`、`load(path)`；评估参数首版 CPU 支持，严格 GPU 在首次更新前拒绝。
不在第一版再复制一套 distribution prediction API。
用户可调用 `objective.constrain(booster.predict_raw(X))`。

`TrainerConfig` 增加 `random_state` 和缺少的 `min_gain`，已有超参数只设一个来源。
`ExecutionContext` 提供 `device`、`xp`（NumPy/CuPy）、本次 fit 的 `rng`、
`round_idx`、`channel`。新接口对象不读全局 backend 来猜执行位置；内部 legacy adapter
仍用 `backend_context`，fit 后恢复，禁止同进程混合 backend 并发。
objective 的 context.channel 为 None，builder 的为当前 channel；context 不由插件原地修改。
内置 adapter 消化新增 context 参数，再调用现有 objective 方法，保留现有数学实现。

### 4.2 不可含糊的数学/数组契约

1. raw/gradient/Hessian 每个 channel 均为连续 float32 `(n_samples,)`；key 必须完整，
   不允许额外或缺失 channel。channel 顺序固定。初始 y 维度不符合一维要求即拒绝，不能 ravel 隐藏错误。
2. `grad` 符号是损失增大的方向。默认叶值 `-sum(grad)/(sum(hess)+reg_lambda)`。
   `hess` 是供树优化使用的非负有效曲率，可以是 Fisher/preconditioner，未必是精确 Hessian。
   正则参数有限且非负；分母为零且 G=0 时返回零叶，分母为零但 G 非零时报错，不隐式除零。
3. objective 负责把 sample weight 同时乘入 grad 与 hess **一次**，trainer/builder 不再乘。
   默认 loss 为 weighted mean。所有权重为零、负数、NaN/Inf 明确报错。
4. 初版禁用自定义 objective 的常数 Hessian hint。内置优化只能在适用目标、无 sample weight、
   无改变 Hessian 的采样/转换时启用，并有对照测试；不能只依据 `natural=True`。
5. `step` 一次读取本轮开始时所有 raw，返回所有 channel 的统计；本轮按固定顺序建树，
   不在每个 channel 更新后重新算其他 channel 梯度。输入视为只读，返回缓冲区至少存活至本轮结束。
6. `F[r+1,k] = F[r,k] + eta[r,k] * tree[r,k](X)`；builder 不得修改 raw。
   `BuiltTree` 是 tree 加可选训练预测的简单容器，不带隐式“已经更新 raw”状态。
   trainer 负责且仅负责一次更新，并存储实际 `eta[r,k]`。
7. `StepSchedule` 首版只支持预定的逐轮逐 channel 系数，不能读/修改 raw 或整组旧树。
   line search、momentum、重新加权全部旧树不是本轮契约；这个接口有意窄于通用 UpdateRule。
8. CUDA 扩展收到 CuPy 数组，内部 Numba 通过 CUDA Array Interface 零拷贝视图互通。
   MVP 使用默认 stream，调用方自定义 stream 不在支持面；测试保留 owner 引用和生命周期，不能
   把 host ndarray 误判为 device。初始化/最终输出可复制，热路径不做隐式 `.get()`。
9. CPU seed 可重现且不改变全局 RNG 状态。旧采样路径接收同一个 scoped Generator；
   CPU/CUDA 对比可复用明确索引。没有 GPU 位级确定性的声明。

### 4.3 dispatch、错误与报告

显式 builder 必须优先，不能因输入适合 native path 而被绕过。默认 builder 可以选择
已验证 native kernel；adapter 传 `pred_gpu=None`，统一由 trainer 应用更新。
初版允许因此失去部分 fusion；测量其代价，不能为一个速度数字破坏更新契约。
`_models/_boosting.py` 的独立旧循环暂不整体重写。

CUDA 能力用显式声明和内置实现身份检查，不用 `type(...).__name__`。
`fallback="error"` 是实验 GPU 默认值：能力不满足在首次训练更新前报错。
`fallback="warn"` 只允许已知能力缺口在 fit 开始前选择并报告完整 CPU 路径；
不在某轮捕获任意异常、复制 raw 后悄悄继续。运行时数值错误、非法 shape、kernel 失败应保留原因并失败。
现有 facade 的合法混合执行可以保留，但必须在报告中清楚表示。

`fit_report_` 至少记录 requested/actual device、objective/tree/update/eval 的实际执行位置、
使用的 builder 路径、fallback 原因、seed、每 channel 树数以及计时同步边界。
传输计数仅声称覆盖 OpenBoost 的包装边界；外部 CuPy 代码可能自行复制，不能将计数当全进程证明。
驻留测试再用小规模 profiler trace 核对；不可用 shape 或 `cuda.is_available()` 代替。

## 5. 树 primitive：允许 Python 改法，保留 GPU 数据通路

现有 `dict[int, NodeHistogram]` 是 host API，保留兼容，新增实验批量 API；不伪造返回类型兼容。
底层优先复用已有 histogram/split/partition kernels，先证明边界再做性能优化。

| 批量表示/操作 | 目标形状和职责 |
|---|---|
| HistogramBatch | grad/hess `(node_slots, features, 256)` float32；样本计数 int32，active mask bool，同设备 |
| SplitBatch | 每 node_slot 的 feature/threshold/child IDs/gain/valid mask 数组，同设备 |
| `build_histograms` | 按真实 sample node IDs 聚合，零权重、空节点与未激活槽位定义清楚 |
| `find_splits` | 批量比较候选，排除 invalid child；相等增益按 feature、threshold 顺序确定 tie |
| `partition` | 根据 split 更新样本 node IDs，返回新数组或显式声明原地缓冲区 |
| `leaf_values` | 在设备完成 sum/reduce 和叶值规则；不下载 grad/hess/sample IDs |

为限制动态分配，level-wise 首版采用完整二叉树固定槽位与 device active mask，root=0，
left=2i+1，right=2i+2。leaf 的 children=-1。每层可固定循环，不为获取 active node 数下载样本数组。
最深 8 层，histogram 临时预算默认 256 MiB；超预算先明确拒绝，暂不实现 out-of-core。
bin 255 始终保留为 missing；即使该路径拒绝 missing，也不把该 bin 当普通候选。
常量特征、全零有效曲率、无有效 split、负/非有限增益必须有独立小例子。

`LevelWiseBuilder(leaf_rule=...)` 首先暴露最小树扩展：leaf rule 接收批量 G/H、config、context，
返回同设备叶值。用户需要进一步修改 tree 时可以自己组合公开批量 primitives；
不提供任意 Python callback 自动注入 compiled kernel 的承诺。

最终返回现有标量 `TreeStructure`：每棵树完成后允许 O(tree_nodes) 的结构/叶值下载，
保留 device cache 供本轮训练预测复用。这样复用 CPU prediction 与 serializer，
无需一次性重写所有树对象。允许的传输是初始化、每棵树的紧凑结构、必要的标量状态/错误检查、显式评估、最终输出；
热路径不允许 O(samples) 的 raw/grad/hess/node IDs 或整个 histogram 下载。

split gain 的标度和 min_gain 比较沿用已验证 CPU 实现，P3 将其写成独立公式测试。
如现有 CPU/native 公式不一致，先记为正确性问题解决，禁止临时放宽 parity 阈值。

## 6. 三个扩展实验，两个独立包

两个包放在 `examples/extensions/normal_fisher/` 与 `examples/extensions/bounded_leaves/`，
各有 pyproject、README 和独立测试。build wheel 后在新环境安装 OpenBoost wheel 与扩展 wheel，
从仓库外目录执行，禁止 editable/PYTHONPATH 注入、private import、复制 core 或 monkeypatch dispatch。
源码可同仓维护，但测试必须证明安装边界；不要宣称这是外部用户。

**A. 外部两参数 Gaussian/Fisher objective。** raw 为 mu 和 log_sigma；
`s2=exp(2*log_sigma)`，`g_mu=(mu-y)/s2`、`h_mu=1/s2`，
`g_log_sigma=1-(mu-y)^2/s2`、`h_log_sigma=2`，最后乘权重。
初始位置与方差使用 weighted mean，方差下限 1e-6；最小测试选有限范围，非有限状态报错。
梯度由独立 finite difference NLL 核对，Fisher 由解析参考核对。
CPU/CUDA 都实现；这是独立实现与扩展性实验，不声称 Gaussian Fisher 是新算法。

**B. 外部 bounded-Newton leaf。** 在 tree 生长期间使用
`clip(-G/(H+lambda), -c, c)`，无有效样本的叶值为零；finite c>0。
比较未裁剪参考，必须观测到真实叶值及下一轮 gradient 的变化。
检验不止“callback 调到了”；GPU clipping 和 reduction 在设备，保存后结果保持。
该方法沿用默认 split criterion，不能描述为重新优化了 clipped objective 的所有 split。

**C. 非常数逐 channel schedule。** 在 A 的包中实现
`eta[r,k] = base_lr * channel_scale[k] / (1 + r/tau)`，tau>0，
mu scale=1、log_sigma scale=0.5。用手算两轮例子验证每轮、每个 channel 的实际系数，
再验证训练 raw、重新 predict、early-stop 恢复及 save/load 相同。
它验证更新接口，不是 line search，也不构成新的训练算法贡献。

在 A/B 都完成后安排外部作者实验。如果真正的外部方法仍需 private import 或 fork，
先记录缺哪个接口；不凭空添加十个“未来可能有用”的 hook。

## 7. 持久化与兼容

保存 binner、base scores、channels、tree arrays、每棵树实际系数和版本信息。
新 coefficient 状态必须进入 predict、eval、early-stop 截断/恢复和持久化的同一逻辑；
不能只改变 fit。加载旧文件时若无 coefficient 状态，按旧 learning_rate 合成并测试。

实验 Booster 的 raw inference 不依赖自定义 objective/builder/schedule 代码。
不序列化 lambda 或任意训练对象来“解决”部署；load 后原始预测可运行，继续训练需要
重新提供 objective，warm-start/resume 在首版明确不支持。
现有内置模型的预测变换仍按现有模型处理。

若共享 serializer 格式改变，按当前实际版本递增，保留旧版本拒绝规则；
numeric、missing、categorical、symmetric/linear/vector 等被触及状态必须回归，
不能为了实验路径删除不认识的字段。

## 8. 正确性、性能与产品关卡

| 关卡 | 必须拿到的证据 |
|---|---|
| G0 整合 | 双方提交保留；本地正确性回归、远端新模型测试和 CPU suite 通过或明确现有失败 |
| G1 CPU 契约 | 独立数学 oracle、seed、不重复权重/更新、显式 dispatch、错误路径、保存预测一致 |
| G2 CUDA 基线 | 真实 GPU weighted/unweighted Normal/Poisson 端到端验证，已知故障不能带入新 API |
| G3 扩展驻留 | A/B 独立 wheel 在 CPU/CUDA 运行，C 改变更新；梯度、split、leaf、prediction 与任务指标均核对 |
| G4 工程价值 | 冷/热端到端时间、显存、传输与相同质量比较，原始结果 committed |
| G5 外部 adoption | 外部作者自己的扩展及阻碍记录；无人尝试/全部要 fork 不能算通过 |

微型 oracle 优先用可精确表示、无近似 tie 的数据：hist/grad/leaf 起始容差
rtol=1e-5、atol=1e-6，split topology 在这种数据上必须一致。
并列最优用专门 tie 测试；大数据浮点归约差异用有解释的预测与质量门槛，不能要求所有树完全相同。

预先固定 seeds=0,1,2；数值型真实回归数据先用 sklearn California Housing 的冻结版本/hash，
下载失败就记失败，不换成合成数据冒充真实证据。训练/验证/测试 split 与预处理只用训练数据拟合。
Normal 比较 held-out NLL、CRPS 和区间覆盖率；Poisson 用确定生成过程检验计数场景。
大样本合成 scaling 与真实数据质量分开报告，不冒充官方 ScoringBench。

先冻结 P2 基线，再看实现结果。预设筛选阈值：相同算法相同配置下均值 NLL 的绝对差不超过
`0.01*max(1,abs(baseline_NLL))`，CRPS 相对劣化不超过 1%，90% 区间覆盖率差不超过 1 个百分点；
逐 seed 报告，失败必须调查，三 seed 不包装成统计显著性结论。
微型 parity 是正确性关卡，以上宽一些的真实数据阈值不能拿来豁免它。

默认配置 GPU 端到端 fit 中位数相对整合后旧路径退化超过 20% 触发 profiling 和设计复查，
并不允许改质量或隐藏 warmup。这个数是设计预算，不是已测性能，也不是自定义算法的速度承诺。
优先给出“实现这个方法需要多少公开接口、多少额外代码、多少 GPU 成本”的可复现记录。

## 9. Modal 验证设计

P1 实施调整：使用独立 `benchmarks/foundation/modal_app.py`，保留旧 runner。
旧 app 注册了源码挂载和宽松依赖的其他作业，直接复用会破坏 wheel 隔离边界。现有 runner 只复制单个测试文件、
宽松依赖、部分入口只打印失败，不足以直接当本轮证据。
Modal 支持构建镜像时安装依赖与显式包含本地文件，也有运行测试的官方示例。
使用当前 SDK 的 uv 安装路径，不再新增随意 pip 安装。
来源：[镜像指南](https://modal.com/docs/guide/images)、
[CI 示例](https://modal.com/docs/examples/ci-on-modal)。

- Python 3.12、CUDA 12、固定镜像 digest 与 Linux 依赖锁；包含 numba-cuda、CuPy、pytest/xdist。
  在 Linux 镜像中验证 lock，不能复用 macOS 的已装包清单作为 Linux 环境。
- 本地从干净实现提交 build wheel，计算 SHA256；上传 wheel、必要测试/conftest/config、
  扩展 wheel 与 manifest。安装后验证 import 来源是 site-packages、wheel hash 相符。
  不上传工作区整个目录、.git、凭证或用户数据。
- 默认单 T4、并发 1。`foundation_smoke` 远端 timeout=300 秒，
  `foundation_correctness`=1800 秒，`foundation_benchmark`=1800 秒；pytest 子进程各留 60 秒收集结果。
  初轮 smoke 加 correctness；通过后才跑 benchmark。默认不自动增加 GPU 型号或矩阵。
- app 级 retry=0；收集累计远端运行秒数、失败与已有 run id，初轮计划最多约 1 GPU-hour。
  这不是费用硬上限；镜像构建/启动和平台内部重启须单独记录。Modal timeout 是每次执行的限制，
  平台还可能处理基础设施重试。[timeouts](https://modal.com/docs/guide/timeouts)、
  [failure handling](https://modal.com/docs/guide/functions)。
- 使用本地入口 `::foundation_smoke` 等控制结果；任何 pytest failure、缺失指定 GPU 测试、
  选中必跑测试被 skip、环境校验失败或结果无法取回都使本地命令非零退出。
  不把现有 CPU/CUDA 混合测试文件里合理的 CPU skip 当失败，必跑清单明确列 node IDs。
- 每个结果带 run id、源码 SHA/dirty、wheel hash、数据/version/hash/split、精确命令、依赖、
  CPU/RAM/threads、GPU/driver/runtime、CUDA 实际路径、fallback、同步/预热策略；返回 JSON、JUnit、日志。
  本地保存到 `benchmarks/results/foundation/<run_id>/`，明确修改 ignore 规则纳入审核后的冻结结果。
- correctness 不测宣传速度；benchmark 测完整 fit/predict，分别报告首次编译和 warm median、
  质量/峰值显存，比较 CPU/GPU 资源不同。至少 3 次热运行，始终同步 GPU 后计时。

本轮尚未验证 Modal 凭证、镜像可构建、GPU quota 或实际价格。执行时先做最小 smoke，
失败保留结果并修复具体原因；不循环重试长作业。

## 10. 方向调整条件

- 若两个独立扩展仍需 core 修改：接口假设未成立，先缩小或修正契约。
- 若 GPU 可扩展路径在目标工作负载上持续无经济优势：保留 CPU 研究工具价值，暂停 GPU 平台扩张。
- 若外部作者只需要自定义 distribution：把资源回到分布建模垂直产品，不为通用性扩大核心。
- 若外部作者能持续提交不同算法，且愿意让其包依赖 OpenBoost：再评估 split-gradient/leaf-gradient
  分离、vector leaves 和更深的更新接口；每项由实际算法提出。

竞争背景与更广泛路线见 [impact/adoption/value 研究](../learnings/2026-09-05-impact-adoption-value-strategy.md)。
本计划要获得的是一个可以被采用或被证伪的窄基座，尚不宣称已经拥有 ecosystem。

## 2026-09-05 goal review after P4.1

The larger objective remains useful, trustworthy distributional/risk modeling
and a shorter path from a research idea to a usable implementation. The GPU
foundation is one bounded hypothesis supporting that objective. Passing kernels,
more APIs and more Python code are not adoption or value evidence.

G0/G1/G2 and one histogram primitive have evidence. G3 (independent GPU
extensions), G4 (matched-quality end-to-end cost) and G5 (external author use)
remain open. Continue the smallest path through numeric split/routing, one
bounded leaf rule and two-channel Normal training to the two independent wheels.
Do not add tree families, custom split criteria or extra device support on the
way. The next product checkpoint is a reproducible method implemented through
public APIs, with implementation effort, installation obstacles and runtime
cost recorded; then an external author task, not another list of kernels.

Keep the existing stop conditions: if authors only need custom distributions,
return investment to the distributional product; if GPU gives no end-to-end
benefit, keep it optional. Prepare author materials without sending invitations
or publishing. External attempts and retention still require actual users.

Implementation clarifications: P4.1 uses a small CuPy RawKernel for its separate
layout/count contract; Python percentage and Numba-only kernels are not goals.
Its non-default stream test covers that primitive only, not the future whole
trainer. P4.2 numeric splitting requires positive curvature in both children,
including when min_child_weight=0; this avoids inventing information about empty
versus zero-weight bins from G/H alone. Zero-curvature nodes stay leaves; missing
and categorical builder support remains out of scope. These limits must remain
visible in the public contract and independently tested.

Sequencing adjustment from this review: after the minimal P4.4 builder works,
run P6's independent CPU wheel examples before completing P5's strict GPU
integration. Record public/private imports, installation failures, method code
and steps to correct output. Fix demonstrated API obstacles first. GPU parity
and independent GPU wheels remain mandatory afterward; no external adoption
is claimed from examples we write ourselves.
