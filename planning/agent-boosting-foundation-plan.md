# OpenBoost v1：可编程 boosting 基座设计、执行与验收

日期：2026-09-05。版本：**真正的 v1 规划基线**，由用户明确指定。
状态：**计划已形成；新 v1 实现与 eval 尚未执行**。这里的 v1 是本次产品/架构目标，
不是已有 PyPI 包版本或旧 P0–P7 工作已经完成 v1 的声明。
代码审阅基线：`3ac1552`，分支：`codex/gpu-python-foundation-design`。

本计划取代旧 GPU foundation 清单中的后续投资顺序，以及
[ScoringBench 优先计划](impact-adoption-value-next.md)。旧实验及其失败结论仍有效。
用户明确：foundation 是产品；use case 决定抽象；**不要求 backward compatibility，
允许重写 API、trainer、内部表示和持久化格式**。本轮只规划，不启动重写或 GPU 作业。

共同构成 v1 规格的文件：

- 本文件：产品目标、范围、架构、工作依赖、阶段交付与完成定义。
- [应用覆盖与案例契约](foundation-application-contracts.md)：应用维度与候选数据。
- [验收与 eval](openboost-v1-evaluation.md)：E0–E7、量化门槛、对照、失败与结果协议。
- [三大库最新 release 与公开 plans](boosting-release-review-2026-09-05.md)：带来源的
  竞争事实、已发布/实验性/计划中状态，以及对 v1 的影响。

保险、survival/AFT 是用户举例，**不是封闭清单或行业定位**。v1 的覆盖范围按
不同算法决策、数据/目标语义及真实使用方式选取；不能每次只围绕最近一个例子设计。

## 1. 重新审视产品假设

目标是让研究者与 Agent 能以较低成本，把一个 boosting 算法想法变成经过验证、
可以复用的实现。应优化的是 **从算法改动到可信结果的总成本**：实现、排错、验证、
反复训练和部署，而不只是单次 fit、Python 代码比例或者公开函数数量。

需要验证四个假设，不能合并成一个“技术测试通过”：

1. 有用的问题确实需要改变算法内部，而现有配置或扩展接口不能低成本满足。
2. 一组可组合组件可以显著减少这些改动的工作，同时承载不同类型的算法。
3. Agent 可以正确使用这些组件，失败时能获得足够的信息定位错误。
4. 完整研究工作流的收益足以覆盖运行开销、学习成本和新依赖。

第一批使用者是已经有算法修改需求的研究者、领域建模者及其 Agent。
普通表格任务的一键训练入口有助于试用；算法作者可发布的独立 recipe/package
才是本轮采用路径。研究者通过一个具体方法进入，再复用基础组件并发布自己的方法。
Impact 看独立研究或实际决策中的复用；adoption 看独立完成与重复使用；value 看
可信修改的总成本与任务结果。暂不从这些假设推导付费意愿或商业模式。

### 对竞争前提的修正

- XGBoost 已有 Python custom objective/metric；还提供内置 grow policy、updater
  和叶更新约束。自定义 loss、切换 depthwise/lossguide、简单裁剪叶值不自动构成差异。
  [自定义目标](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)、
  [参数](https://xgboost.readthedocs.io/en/stable/parameter.html)。
- LightGBM 提供逐轮 `update(fobj=...)`、rollback 和 `set_leaf_output`。
  “现成库完全不能修改训练或叶值”不成立。
  [Booster 文档](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)。
- NGBoost 已支持开发分布、score 和相应 metric；新增 Laplace 分布只是容易的对照任务。
  [开发指南](https://stanfordmlgroup.github.io/ngboost/5-dev.html)。
- Py-Boost 明确提供 Python GPU boosting 和多种自定义能力，是基座方向的直接对照，
  不能只和 XGBoost/LightGBM 比较。
  [官方仓库](https://github.com/sb-ai-lab/Py-Boost)。
- 当日正式版本为 XGBoost 3.4.1、CatBoost 1.2.10、LightGBM 4.7.0。XGBoost 3.4
  已扩展 vector-leaf `hist`，CatBoost 已有 GPU custom objective，LightGBM 4.7
  已发布新的 GPU 与数据互操作能力。具体版本、限制和公开后续工作见 release 核对，
  不能沿用“对手完全封闭、没有多输出/自定义 GPU loss”的前提。

以上是本日文档审阅，执行时仍须冻结实际版本并验证对应任务。机会在于改变算法的
成本和可验证性，而不是声称别人没有扩展能力。对照允许使用现成 hook、外层循环，
也允许修改源代码；不得人为限制竞争方案来制造优势。

“至少不比 XGBoost 差”应拆开：组件能构造标准算法，是表达能力要求；在任务上
接近成熟实现，是独立质量门槛；速度和工程完备性又是另一项工作。不能保证任意
Agent 修改都改善结果。提供可复现 baseline，用验证集选择，最后在独立测试集评估。
选择 baseline 保护工作流价值，但调用 XGBoost 不能算作自有组件重建成功。

## 2. 从 use case 提取变化轴

这些是同一基座的检验程序，不是同时开发四条完整产品线。

| Use case | 必须能表达的算法决策 | 最小可检验实现 |
|---|---|---|
| XGBoost / LightGBM 风格 GBDT | 统计量、候选分裂评分、叶求解、depthwise / best-first 生长、采样 | 固定 bins 上的标量二阶 boosting；两种生长顺序，共用统计与路由 |
| NaturalBoost / NGBoost 风格 | 参数链接、proper score、Fisher/其他曲率、方向、拟合弱学习器、步长接受 | 双参数 Normal；固定步长与基于训练损失的有限回溯；拒绝的更新不污染模型 |
| FormulaBoost | `theta(Z)` 与 `formula(theta, x)` 分离、参数耦合、结构输入、参数输出 | 双参数公式，独立 Jacobian/GGN 参考；组合现有树组件完成至少两轮 |
| Train-many | 数据复用、独立 run 状态/RNG/停止条件、执行分组 | 同一训练数据上多个 recipe/config；顺序参考，改变顺序不改变各 run 结果 |

当前 FormulaObjective 已能先计算耦合的 GGN 方向、再逐参数拟合树。
这和共享树结构、联合叶求解是不同问题，不能把前者说成现有系统无法表达。
当前 `fit_trees_batch` 已有共享 binned input 的顺序实现，应保留为语义参考，
不能把新名字或重复封装当作 train-many 进展。

除这些设计探针外，保留两个未参与接口设计的算法任务，检验是否只是为自己的示例量身定做。
最初允许用合成数据检验数学；采用与实际价值仍需要真实使用场景。

另一个维度是 **真实应用**：分类、回归、ranking、分位数、计数/正值目标、删失、
多输出、结构化模型与模型选择。保险/AFT 分别检验 offset/目标单位与删失语义；
ranking 检验组内关系；分类检验 link/class mapping；多输出检验共享树与参数轴。
算法结构 × 应用任务是设计矩阵，不要求第一轮运行全部笛卡尔积。

### v1 最小交付范围

这些是 v1 完成时的 required 能力；F1 可先完成小子集，但不能将它称为完整 v1。
CPU 是独立参考与全部下列 recipe 的执行入口；GPU 按明确子集验收。

| ID | Required recipe / 任务 | 最小行为与检验重点 | v1 CUDA 范围 |
|---|---|---|---|
| R1 | 标准回归、binary 与 multiclass | 平方误差/logistic/softmax；类别映射、权重、概率输出、缺失处理 | numeric dense（含数值缺失）required；原生类别输入 CUDA 可后续 |
| R2 | Quantile / robust regression | 至少一个加权 quantile recipe；叶求解能读取 routed residual/weights，不能只接收 G/H | CPU required；CUDA optional 并显式声明 |
| R3 | Group ranking | pairwise logistic 与 NDCG-weighted lambda 变体；group boundaries、pair generation 与 group metric | CPU required；CUDA optional |
| R4 | Count / positive / aggregate targets | Poisson offset；Gamma/Tweedie 固定声明参数的均值 recipe；目标与 weights/exposure 分开 | Poisson offset required；Gamma/Tweedie optional |
| R5 | Censored AFT | 一个固定噪声族/尺度，完整事件与右删失；规范 interval target，其他删失可拒绝 | 完整事件/右删失 numeric 路径 required |
| R6 | Natural / distributional boosting | Normal 双参数，Fisher/普通方向、固定/回溯步长、proper-score 评价 | Normal numeric 路径 required |
| R7 | FormulaBoost | 双参数 formula、link、结构输入、独立 Jacobian/方向，至少两轮 | CPU required；混合执行单列，不冒充全设备路径 |
| R8 | Multi-output / vector leaf | 独立树与共享 topology/向量叶各一条；分裂统计可与叶求解统计不同 | numeric squared-error/对角统计路径 required |
| R9 | Train-many | M=1/8/32，共享 prepared data，独立参数/种子/停止/失败，顺序参考 | 对兼容的 R1/R4/R8 组至少一条批量路径 required |

| ID | 跨 recipe 的基础能力 | 完成标准 |
|---|---|---|
| C1 | Typed data / targets | numeric、数值 missing、CPU categorical；mapping/unseen policy；weights、offset、group、interval、structure inputs；仅训练集拟合预处理 |
| C2 | 可组合 tree core | additive statistics + 至少一个额外统计；可改 split feasibility/score；depthwise、best-first、symmetric 三种 CPU policy；真实 routed rows |
| C3 | 可替换弱学习器与叶求解 | scalar/vector payload；Newton 与 weighted quantile；输出 schema 与参数轴独立；一个自定义 split constraint |
| C4 | Explicit runtime | scoped device/workspace/RNG，候选/接受/拒绝/停止语义，独立 run，明确 sync/transfer/fallback，无全局 backend 约束 |
| C5 | Model artifacts | raw 与用户预测空间区别、类别 mapping、base/link/offset 约定、树/系数/向量叶、新版本 persistence、可读诊断 |
| C6 | Evaluation / authoring | 独立 oracle、5 类算法修改、2 个未见任务、真实任务、完整失败记录和 wheel 外部包验证 |
| C7 | 可用工作流 | 安装→baseline→读/改 recipe→验证→保存/加载→CPU 推理；能力表、错误定位、可复现实验 |

Native categorical v1 可选定一种可验证的分裂方法；不要求复刻 CatBoost ordered
boosting/全部 CTR 组合。CSR/CSC 稀疏数据、文本/embedding 特征、完整 Cox/竞争风险/
截断生存、全部约束、线性叶、DART/GOSS、分布族目录、自动微分编译器、Ray、多 GPU、
out-of-core 等进入有理由的后续清单；不能默默接受相应参数。
F0 必须记录这些能力在三大库中的状态以及 v1 不实现的原因，保持边界可扩展。

## 3. 推荐架构：可直接编程的算法 + 明确状态 + 批量算子

不以现有 `Booster(objective, builder, schedule)` 为必须保留的外壳。
**Recipe 是普通 Python 算法程序，可以拥有训练循环；runtime 管理资源与状态，
不替所有算法规定训练顺序。** 便利模型是 recipe 的薄入口。

| 边界 | 应负责什么 | 不应提前写死什么 |
|---|---|---|
| Problem / parameter state | 数据角色、目标、权重、结构输入、raw 参数、link、初始化及评价 | 所有任务都只有一个 y；所有辅助量都是 sample weight |
| Recipe | score/方向、生长策略、弱学习器选择、参数更新次序、候选接受与停止 | 每轮必须每个参数各拟合一棵树；只有预先确定的 learning-rate schedule |
| Components / bulk ops | 分箱、按路由聚合统计、候选统计/评分、partition、叶求解、预测与更新 | 所有统计都只有 G/H；树一定是固定 511 槽位；loss 等于曲率 |
| Runtime / run state | 显式 device/stream/workspace、buffer 所有权、run RNG、候选提交、批量执行 | 进程全局 backend；参数轴就是模型轴；任意 Python 自动编译 |
| Artifacts / verification | 可保存的模型状态、声明过的推理程序、诊断与独立参考 | 任意训练闭包自动可序列化；通过验证就证明任意算法正确 |

名称用于讨论，不是现在冻结的类目录。优先少量数组、结构化状态和普通函数；
只有两个实际调用方需要同一边界时，再固化成公共协议。

### 算法应写到什么程度

下面是设计伪代码，不是已有或已确定的 API：

```python
with runtime.run(data, seed=seed) as run:
    state = initialize(problem, run)
    for step in range(budget):
        geometry = problem.geometry(state)          # loss、导数/曲率语义明确
        targets = direction_rule(geometry, state)   # 可跨参数耦合
        learner = grow(data, targets, split_rule, leaf_rule, growth_policy, run)
        candidate = propose(state, learner)         # 尚未改变 accepted state
        decision = accept(candidate, problem, run)  # 可固定步长，也可回溯
        state = run.commit(state, candidate, decision)
    artifact = export_model(state)
```

`grow` 本身也必须有可读的组合实现：aggregate → candidate statistics → score /
feasibility → choose → partition → solve leaves。作者既能替换一段，也能直接调用
这些操作写另一种 grow。不能“开放 grow”之后又把全部算法决策封进第二个黑盒。
没有必要为每个小函数都设计 callback/registry 或通用调度图。

### 必须明确的语义

- **统计与方向分开。** 精确 Hessian、PSD 近似、Fisher、pseudo-response 和其拟合
  权重分别命名；谁施加 sample weight 要唯一且可查。不要把所有方向伪装成同一种 G/H。
- **候选与已接受状态分开。** 明确联合更新或依次更新；回溯最多尝试预定次数。
  拒绝不能留下树、系数、raw prediction 或 best-state 残留。RNG 消耗规则按 run/step/
  component 定义，不能随调度顺序变化。实现不要求每次复制完整数据或模型。
- **共享统计可扩展。** 先支持 G/H/count 和一个额外的可加统计通道例子；声明 shape、
  dtype、归约含义、内存预算和支持的 score。暂不承诺任意 Python reducer 都能加速。
- **叶求解不限于加法统计。** Weighted quantile 等规则需要 routed rows/residuals；
  用显式 row view 或声明的额外统计提供，不强迫它伪装成二阶 Newton 更新。
- **Objective 可有样本间依赖。** Ranking 的 query/pair、Cox 的风险集等不能被
  逐样本独立协议排除；方向计算与树的 per-row 归约接口是不同边界。
- **模型轴与参数轴分开。** 一个 run 可有 K 个参数；M 个 run 可各自有不同 K、树数、
  配置和停止状态。先用独立 state 列表，兼容组再打包，不强制巨大的 M×N×K 张量。
- **弱学习器与参数更新分开。** 弱学习器声明输出 schema，recipe 决定映射到哪些
  参数；runtime 不要求“一棵树等于一个 channel”。树拓扑与叶 payload 分离；
  标量叶可先实现，但 v1 必须用 R8 的真实 vector-leaf recipe 验证，不能只留设计空间。
- **数据复用有身份。** 只有训练行、分箱策略、影响分箱的权重及元数据等一致才共享
  prepared data。不得跨验证折用全量数据分箱。样本索引、结构输入和权重必须对齐。
- **推理是显式能力。** 标准树/raw 参数可由核心 reader 推理；自定义 formula/link
  需要声明其推理依赖或受支持的表示，不承诺卸载任意自定义公式代码仍可预测。
  新格式可拒绝旧格式，不做兼容层；新格式自身的 round trip 仍须正确。

### GPU 与 Python 的取舍

Python 负责可修改的算法决策，大数组操作留在设备上；GPU 仍是设计目标，
CPU 是可安装的入口和独立参考，不应到最后才考虑设备执行。
第一版执行选型以 NumPy 参考 + CuPy 数组 / 小量现有 CUDA kernel 为起点，
只在剖析证明需要时换 kernel DSL。选型可重做，不把 Numba、CuPy 或 Python 比例
当作产品承诺，也不先写编译器或比较所有 GPU 框架。

允许已验证的常用组合有批量/fused 实现；它必须遵守与可读组合实现相同的语义，
改变 recipe 后应重新检查能力，不能悄悄忽略修改。任意 Python 循环仍可能同步或
无法加速，执行报告应说明。GPU baseline recipe 应在 fit 中保留设备树和预测状态，
按明确边界导出，不强制每棵树回 CPU 快照再遍历预测。

验证要分清开发期独立 oracle、运行时边界检查和导出检查。重新设计所有权与检查
粒度，而非保留低效路径再加一个“相信插件”的开关。便宜检查不能代替数学验证；
只读数组和 verifier 也不是隔离任意 Python 代码的安全沙箱。

## 4. 对现有资产的取舍

| 当前资产 / 限制 | 新计划如何使用 |
|---|---|
| 独立 CPU 数学/路由参考、真实 CUDA conformance、wheel 安装、失败复现 | 保留语义与原始证据；新 API 可重写对应测试，禁止为了通过而删掉正确性条件 |
| `_trainer.py` 按 channel 建树，schedule 只看到 round/channel/lr | 可以整体替换；不要求适配这个循环或复制其限制 |
| `TreeStructure` 精确类型、深度 0–8、固定槽位、host snapshots | 都是旧实现边界，不是新设计约束；新表示仍须明确支持范围和资源上限 |
| FormulaObjective、分布数学、`fit_trees_batch` 顺序参考 | 可复用已核实数学，也可重写；不得当作唯一独立 oracle |
| 旧 public API、加载器、实验插件 | 不需兼容，不做双写、弃用周期或迁移 shim；旧实现可从历史 revision 复现 |
| ScoringBench 构造参数修复 `3ac1552` | 保留有用修复；后续用于分布 recipe 质量，不再决定整个基座的排期 |

现有 [P7 T4 结果](../benchmarks/results/foundation/20260905T193308Z-5ebd75ab/README.md)
中，实验路径 warm fit 是同轮 legacy CUDA 的 **12.888 倍**；原定 1.2 门槛失败。
这说明当前路径成本不可忽略，并不证明“可编程 boosting 基座没有价值”。
旧门槛不改写；新算法/工作负载的质量和成本预算须在测量前另行冻结。
当前独立扩展组合还出现 proper score 变差，必须保留这个反例。

## 5. 执行顺序与验收

使用 F0–F5，避免和旧清单已完成的 P0–P7 混淆。每个子项应是可独立审查的小提交。
当前交付停在规划；下一次开始实施时从 F0.1 开始，按依赖与证据门槛推进。

### F0：先写“必须能够写出的算法”与判卷标准

- [ ] **F0.1 — 任务卡与替代方案审计。** 新建 `planning/foundation-tasks.md`：
  从四类结构探针出发，固定全部 required recipes 的输入/输出、两轮状态变化、
  支持边界和独立验收。
  明确 incumbent 的现成接口、可修改源码的路径及其成本；尤其纳入 Py-Boost。
  按 R1–R9 / C1–C7 形成矩阵，涵盖应用契约中的不同任务。保险与 AFT 是其中的
  语义探针；写清 release 已发布/计划中状态、scope 排除理由与数据/目标/评价。
  交付任务卡 + 简短接口草图；每项需求能映射到 E0–E6，不再新增一套空泛路线图。
- [ ] **F0.2 — 原始 NumPy 参考与反例。** 新建 `tests/v1/reference/`，
  独立枚举 split/逐样本归约/小矩阵求解，不导入待测 production 算法。
  写可失败的手算 fixture：重复阈值 tie、空/零权重子节点、非法候选、拒绝更新、
  参数更新次序，以及两个 run 交换执行顺序。标准 GBDT 与外部库比较时区分
  binning/并列选择差异；不能强求不同算法的整棵树逐位相等。
  加入 exposure offset/权重各施加一次、预测单位，以及事件密度和删失概率的独立
  loss/导数参考；边界标签中的合法无穷大不能被通用 finite-vector 校验误拒绝。
- [ ] **F0.3 — 冻结并实现比较协议。** 新建 `benchmarks/v1/` 的 manifest/runner/判卷器：
  固定库/Agent 版本、任务、公开示例与保留任务、预算、正确性容差、质量门槛、
  CPU/GPU 环境与预期结果矩阵。先用旧代码/参考跑 baseline，声明成本预算再测新代码。
  落实 [E0–E7 协议](openboost-v1-evaluation.md)，用缺项/污染缓存/失败记录验证判卷器。
  v1 主基线固定三大库新 release，先 CPU capability smoke，再预检合适的 GPU 变体。
  不继承过时阶段的“完成”标记。实际资源预算/数据 hash 未冻结时不能启动完整评测。

首轮开发探针包括：一个现成工具擅长的控制任务（新增 loss/分布）；一个深入组件的
任务（候选 split 必须满足各预定义 cohort 的最小 information mass，参考逐候选
枚举，`information_weight` 与训练 sample weight 分开）；一个更新过程任务
（对多参数候选做有界回溯并正确拒绝）。第二项是统计稳定性实验，不宣称算法创新
或真实需求已证实。若现成工具已轻松满足，也记录这一结果，不临时加难度。
完整 E2/E5 覆盖还包括叶求解和 run scheduling；上述探针不替代五类修改 eval。

F0 出口：任务可判对错、比较对象公平、修改目标明确；还没有理由建立一套大框架。
最先三个提交就是 F0.1、F0.2、F0.3，禁止夹带模型迁移或 kernel 优化。

### F1：CPU 基座，先闭合小路径，再完成 v1 覆盖

- [ ] **F1.1 — data/run/artifact 语义。** 显式 device/run identity，唯一权重语义，
  参数与模型轴、候选提交/拒绝、独立 RNG、输入所有权及新版本模型表示。
  用 F0 的 state fixtures 验证，不先搬迁全部旧模型。
- [ ] **F1.2 — 可组合建树。** 实现 aggregate、候选统计、score/feasibility、
  partition、leaf solve 与 predict；把两种 growth policy 写成普通 Python。
  先 numeric、有权重、标量常数叶形成最小提交，再完成 C1 的数值缺失/CPU 类别支持。
  独立的 symmetric policy 复用统计/route；不先复制三套完整 builder。
  额外 information mass 能通过同一统计/分裂管线使用，不能在 trainer 特判任务名。
- [ ] **F1.3 — 两个完整小 recipe。** 标准二阶 GBDT 和双参数 Normal；后一项包含
  固定步长及接受/拒绝。至少两轮，比较中间状态、最终预测与新格式保存/加载。
- [ ] **F1.4 — 两个早期结构探针。** Formula 两轮含结构输入与参数链接；train-many
  先 M=1/2，再 M=8 的独立状态，含不同停止轮数与一个 run 失败的记录。
  故障不能污染其他 run，最终汇总不能隐去失败。按稳定 run ID 改变排列验证结果。
- [ ] **F1.5 — 应用语义探针。** 同一基座组合 Poisson + log-exposure offset 与
  固定噪声尺度的右删失 AFT，各至少两轮并验证推理/新格式 round trip。AFT 事件与
  删失走不同 likelihood 项；interval/left censoring 首版若未支持须显式拒绝。
  若必须修改通用 runner 才能传入这些标签或 offset，先修正 Problem/state 边界。
- [ ] **F1.6 — 完整 CPU 覆盖。** 补齐 binary/multiclass、group ranking、weighted
  quantile、Gamma/Tweedie 均值 recipe、vector leaf；分拆为各自验证的提交。新状态、
  类别 mapping、输出语义及推理跨进程保存验证，完成 R1–R9 与 C1–C5 的 CPU cells，
  提供 C6/C7 的本地执行入口；Agent 比较和干净环境交付分别留给 F2、F5。

F1 出口：required CPU recipes 能通过相同语义组件表达。若 Formula 或 train-many
要绕开核心状态语义，先修改设计，不能称“以后再支持”而冻结接口。不承诺完整
XGBoost、LightGBM 或 NGBoost 功能/速度相同。
F1.5 与 F1.6 也必须通过才能冻结 v1 CPU 接口；更广删失支持及真实质量单独记账。
验收对应 E0 的 CPU 子集、E1、E2；先完成 F1.1–F1.3 只算最小架构里程碑。
F2.1 的发现性试用在 F1.3 后即可开始，避免补齐全部 recipe 后才发现边界难用；
F2.2 的正式比较仍等待所需 CPU 能力完整、接口与判卷器冻结。

### F2：Agent 改算法试验，决定接口是否值得固化

- [ ] **F2.1 — 发现问题的试用。** 安装 wheel，在独立工作目录完成 F0 任务，记录
  每次求助、core/private import、失败和修正；允许据此重设计 API，但这次不计成绩。
- [ ] **F2.2 — 冻结后的比较。** 同一 Agent 版本、工具与预算，用 OpenBoost、合适的
  incumbent 公共接口，以及必要时可改源码的 incumbent 路径做任务；可选原始 NumPy
  作为成本参考。至少三个独立尝试，顺序轮换，分别记录 setup、active time、token/
  计算量、人工提示、正确完成率与需要理解的代码范围，不只比较 LOC。
- [ ] **F2.3 — 保留任务。** 使用未参与 F1 改 API 的任务做检查；一旦据此修改接口，
  它就转成开发集，不能继续声称该任务是未见验证。独立 verifier 与最终评价记录
  不由候选实现覆写。小样本是方向性证据，不宣称统计显著或外部 adoption。

F2 出口以 E5 为准：5 类开发题、2 个保留题、固定尝试数和预算，正确率与至少两类
深层修改的成本收益均达标，并保留控制任务的比较结果。若优势只来自文档或任务
提示，应先修文档再公平重测。若持续必须修改 core，说明抽象不合适；不是补更多
包装后自动通过。接口语义改变后重跑受影响的 eval。

### F3：让已证明有用的组合在单 GPU 上成立

- [ ] **F3.1 — 设备常驻垂直路径。** 基于 F1 语义实现 bulk ops，沿一条完整 recipe
  验证梯度/有效曲率、统计、split、leaf、逐轮更新、最终质量和推理，不只测 kernel。
  首先复现旧 P7 任务，以原阈值报告；新 recipe 单独按 F0 的冻结协议评分。
- [ ] **F3.2 — 扩展确实走设备。** F2 至少一个非默认组件及多参数更新上 GPU；任何
  host fallback 和同步可见。动态回溯需要同步时计入成本，不能拿固定步长代替。
- [ ] **F3.3 — train-many 兼容组。** 在顺序结果一致后测试 M=1/8/32 的共享数据与
  批量执行；不同参数数目/不同轮数可以分组，不要求所有 recipe 都支持 fusion。
  记录吞吐、延迟、显存上限、失败、JIT/传输和端到端质量；先单 GPU，不做 Ray/多 GPU。

F3 出口：同一算法的可组合实现与优化实现语义一致，达到冻结工作负载预算。
用已允许的 Modal 做有超时和完整 provenance 的有界运行。未通过可保留 CPU 研究
价值，但不得宣传 GPU 成本优势，也不靠移除验证掩盖结构性开销。
最终完成 R1/R4/R5/R6/R8 的 required CUDA 子集、R9 批量执行与 E4；性能门槛和
全部同步/传输成本不能以单个 kernel 的加速比代替。CUDA optional 的 recipe 单列
其 CPU 结果与设备状态。

### F4：真实 use case 与独立采用

这一阶段不必等 F3 全部完成：F1 通过后即可准备 CPU 试用材料，F2 的接口修订完成后
即可开展获授权的外部试用。用户的实际阻碍应能反过来调整 GPU 优化顺序。

- [ ] 从有实际使用理由的 recipe 选一个真实数据任务；分布模型可用 ScoringBench，
  Formula 需要明确结构假设；train-many 需要完整模型选择成本。使用强 baseline、
  预注册质量指标、独立测试集、重复种子/折，发布失败到本地原始 artifact。
  按 E3 完成至少8个真实任务单元、至少6个独立来源，覆盖分类、回归、ranking、
  quantile、多输出、count/positive、survival。保险与 AFT 不能替代其余覆盖。
- [ ] 准备作者可直接安装的 wheel、recipe 源码、最小复现、组件契约与诊断实例。
  第三方定义的任务价值高于继续添加自己编写的 demo。
- [ ] 经用户授权联系后，让至少两名外部作者尝试，其中一人完成自己的方法并在
  后续任务复用。记录完成、阻碍和复用原因；当前没有外部尝试或留存证据。

F4 有两个独立出口：工程质量子任务按 E3 验收；外部采用按 E7 检验具体作者的
重复使用与可复现收益。内部 Agent 成功不计 outside adoption；无需等待全生态、
所有模型或完美 benchmark 才准备试用材料。不能把未获授权的外联当作其他工程
任务的阻塞，也不能在 E7 未通过时声称已经验证采用。

### F5：依据证据稳定产品边界

- [ ] 只稳定反复被用到的组件契约，整理独立 recipe 的打包与发现方式。
- [ ] 删除重写后无使用方的旧执行路径和重复概念；正确性 case 迁移为新语义测试，
  不要求保留旧 import/signature/文件格式。旧行为通过历史版本复现。
- [ ] 确认安装→运行 baseline→改算法→验证→保存/推理的完整路径；同步公共文档。
  公共 README 在实现通过前不能提前描述为已具备新架构。

### v1 完成定义与依赖

主依赖：F0 → F1 → F2 → F3 → F5；F4 的真实 baseline 准备可在 F0 开始，正式质量
在相关 recipe 稳定后执行，外部试用可在 F2 后独立推进。
每阶段 commit 附对应 E-gate、原始记录和未完成单元。F5 检查 E0–E6 全部 required
通过才称“工程 v1 完成”；E7 单列，决定能否声称外部 adoption/impact。
每个大阶段末重看任务与 component 的复用关系，不仅检查完成了多少函数。

计划中的新增实现目录以公共模块/普通函数组织：data/targets、stats/ops、tree、
runtime、recipes、artifacts；F0 用最小调用示例确定精确命名。可替换现有私有模块，
无需永久维护 v1/v2 两套 trainer。测试在 `tests/v1/`，评测在 `benchmarks/v1/`；
旧 baseline 通过固定历史 wheel/revision 复现，不为兼容旧调用牺牲新设计。

## 6. 执行纪律与停止条件

不要求兼容不等于要求无选择地重写。可以从零改变架构；复用已经正确且符合新边界
的计算实现是工程选择，不是义务。不要为了保留旧测试签名做 shim，也不要为“从零”
丢掉失败样例、证据、未提交用户修改或历史。

设计前期允许 breaking changes；F2 正式比较时冻结一个版本，防止评测随接口漂移。
新 API 不必永久支持任意插件，但每次 evidence 必须能定位到精确版本。

- 表达力、正确性、运行成本、Agent 修改成本、外部采用分别记账。
- 如果只在 control task 上有优势，先判断是否一个更好的教程/适配层就足够。
- 如果每个新用法都加一套专用 core，缩小可复用边界或重设计，暂停扩功能。
- 如果只有新算法质量改善、实现工作并未减少，记录算法结果，不冒充基座价值。
- 如果 GPU 开销持续超预算，定位设备驻留/批量粒度/同步；不立刻升级多 GPU。
- 前一关的证据支持不了下一关的投入时，提交当前事实和修订计划，不以更多代码代替。

每步先找最小独立失败样例，用 `uv` 运行相关测试和 lint，修改公共能力后检查文档与
打包，更新 learning，检查 staged diff 后提交。只有改动影响到的行为和语义才需要
迁移回归测试；对已主动移除的兼容接口不保留通过义务。
本轮规划的校验是 Markdown 链接及 `git diff --check`，没有新运行时行为需要测试。
无 push、release、外部消息或 leaderboard 提交。

给后续执行模型的起始任务：

> 先读 AGENTS.md、此计划、eval 协议、release 核对、应用契约和 v1 planning learning。
> 执行 F0.1：形成
> R1–R9/C1–C7 任务矩阵，使用最新三大库核对，覆盖不同应用及五类算法修改，
> 核对最合适的替代方案，明确每项算法改动的输入、输出与判错条件。不维护旧 API
> 兼容，不提前重写内核。检查链接与事实后提交；随后按 F0.2 建独立数学参考，
> F0.3 冻结比较协议。每次提交报告通过与未验证的部分，保留失败结果。
