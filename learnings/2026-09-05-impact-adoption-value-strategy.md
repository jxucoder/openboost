# 2026-09-05: OpenBoost 的 impact、adoption 与 value 路线研究

## Context

问题：OpenBoost 接下来怎样投入，才能提高研究影响、实际采用和用户价值？

本文是研究建议，不是已经通过的产品决策，也不修改 AGENTS.md 的使命和优先级。默认按小团队、未来 6–12 个月、先获得真实使用再放大研究与商业价值排序。团队规模、可用 GPU 预算、行业关系和现有用户留存未知；下文的数量和期限是建议的实验门槛，不是增长预测。

研究日期：2026-09-05。证据分为本地实测、线上源码/项目自述、外部一手资料和待验证假设。没有进行用户访谈、付费意愿测试或新一轮 GPU/第三方质量 benchmark。

### 必须先区分的版本

- 本地检查基于 `05cd8bc800595a2f40c4d08f51afb697968b9b3e`；本地 `origin/main` 是 `504fdd0bfc60e5d8e7518250e087fb7e4766d1b4`。开始时工作区干净，本地相对这个远端跟踪引用领先 11 个提交。
- GitHub connector 读取的线上 `main` 已有 FormulaBoost、WeibullAFT、统一 trainer 和更新后的概率建模定位；相关文件返回的修改日期是 2026-08-17/18。这些文件不在上述本地快照中。
- 线上 README、旧的 raw.githubusercontent.com 搜索缓存、本地 checkout 和 PyPI 入口不一致。研究优先使用 connector 返回的线上源文件，并把本地实测限定在本地 SHA。没有拉取、合并、重置或推送任何分支。
- 线上源码使用可变 `main` 链接，本次未锁定其不可变 SHA；不能把本地 26 项测试通过写成线上新架构已验证。

线上来源：[README](https://github.com/jxucoder/openboost/blob/main/README.md)、[统一 trainer](https://github.com/jxucoder/openboost/blob/main/src/openboost/_trainer.py)、[objectives](https://github.com/jxucoder/openboost/blob/main/src/openboost/_objectives.py)。

## Decision or Result

建议的主路线是：**围绕可检验的概率预测构建一个小而完整的产品，以 NaturalBoost 获得采用，以真实风险任务证明价值，用有预算上限的 FormulaBoost 实验探索研究贡献。**

对外可以逐步形成这样的定位：

> OpenBoost 帮助 Python 团队建模和检验表格数据的预测分布，并在需要时将领域公式纳入模型。

“校准优先”应代表训练、独立校准、诊断和部署验证的完整流程。仅有 `predict_interval` 或较均匀的总体 PIT，不能承诺模型对所有群体、尾部或分布变化都可靠。

现阶段最重要的问题是：**哪个外部团队会在自己的第二个任务上继续使用 OpenBoost，原因是什么？** 这比新增一个模型类更能判断路线是否成立。

### 1. 三个目标分别优化什么

| 目标 | 希望产生的结果 | 优先观察的证据 |
|---|---|---|
| Impact | 别人用 OpenBoost 完成此前困难的分析、方法或决策 | 独立复现、被外部项目集成、外部研究使用、真实案例 |
| Adoption | 外部用户能上手，并在后续工作中继续使用 | 首次成功率、首次得到有用结果的耗时、第二次使用、四周留存 |
| Value | 用户获得足以抵偿迁移和维护成本的收益 | 在匹配质量下减少计算/工程时间，或改善事先定义的业务决策损失 |

研究影响可用“独立使用团队数 × 持续使用程度 × 每团队实际改善”作定性判断，不能把它当成已估计的数学增长模型。Stars 和下载量用于观察传播；下载会混入 CI、重复安装等流量，不能直接等同活跃用户。

### 2. 已有资产与仍然缺失的证据

**本地已核对的资产：** NaturalBoost、分布参数预测、sample weights、部分分布的 exposure offset、NLL/CRPS/quantile/interval 评估、PIT/reliability/recalibration 工具及相应测试。26 项聚焦测试通过。实现见 [distributional](../src/openboost/_models/_distributional.py)、[utils](../src/openboost/_utils.py)，测试见 [distributional tests](../tests/test_distributional.py)、[utils tests](../tests/test_utils.py)。

**本地已有的第三方比较：** 一个三数据集、单 seed、单次计时的 CPU NaturalBoost/NGBoost 原始 JSON，反映的是接近的质量和有胜有负的速度，不足以推出普遍优势；其元数据也没有覆盖当前 AGENTS.md 要求的全部 provenance。[原始结果](../benchmarks/results/ngboost_comparison_20260720.json)

**线上新增的能力：** FormulaBoost 的调用链是 formula model → FormulaObjective → fit_boosting；目前公式损失仅支持 MSE，有限差分 Jacobian 和 GGN 在 CPU，树可以走 GPU。统一 objective 对 Normal/Poisson 有设备计算路径，不能继续把旧版“所有分布梯度都在 CPU”套用到线上版本。[FormulaBoost 源码](https://github.com/jxucoder/openboost/blob/main/src/openboost/_models/_formula.py)、[objectives](https://github.com/jxucoder/openboost/blob/main/src/openboost/_objectives.py)

**线上实验是有价值的线索，尚未成为本次独立验证的结论。** benchmark 页面报告了新的 GPU timing、8 个 UCI 数据集及 FormulaBoost/Weibull 合成实验，同时承认 3 个 UCI 数据集缺失、XGBoostLSS/LightGBMLSS 尚未比较。页面开头称结果来自 committed runs，末尾又说明 JSON 在被忽略的目录、表格来自任务记录转录。本次没有核验到与这些新表格逐项对应的冻结原始结果与完整环境信息，因此不把速度倍数作为战略前提。[线上 benchmark 源文件](https://github.com/jxucoder/openboost/blob/main/docs/benchmarks.md)

**采用入口仍然分裂。** 本地 README/quickstart 偏向通用 GBDT，线上 README 已转向 distributional regression，PyPI 默认页仍显示旧 stable 入口，release history 同时列出 `1.0.0rc1`。优先把源码、版本、安装命令、文档和可复现实例对应起来；达到门槛后再推进稳定发行。[PyPI](https://pypi.org/project/openboost/)、[线上 README](https://github.com/jxucoder/openboost/blob/main/README.md)

### 3. 竞争意味着什么

| 方向 | 已有替代方案与一手证据 | 对 OpenBoost 的含义 |
|---|---|---|
| 通用分布预测 | [NGBoost](https://stanfordmlgroup.github.io/ngboost/1-useage.html) 已提供分布预测、proper scores 和 survival | 提供预测分布本身不足以驱动迁移 |
| 丰富的分布族 | [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/) 有多种分布、mixtures、flows、多目标等 | 不宜以分布数量竞赛建立定位；必须纳入相关任务的比较 |
| GPU 概率回归 | [PGBM](https://github.com/elephaint/pgbm) 明确面向大规模概率回归并支持 GPU | 同一用户任务下它是合理候选基线，不能仅因自己的文档将它列为 reference 就排除 |
| 可修改的 Python GPU boosting | [Py-Boost](https://github.com/sb-ai-lab/Py-Boost) 已强调可扩展性、多输出和 GPU | 可读 Python 是开发体验优势，需要具体“新增方法省了多少工作”的案例 |
| 区间与风险控制 | [MAPIE](https://mapie.readthedocs.io/en/stable/) 有 conformal、校准、risk control；[skpro](https://skpro.readthedocs.io/en/stable/) 有概率接口、指标和 pipelines | 兼容与集成更有价值；避免同时再造一个通用 uncertainty 平台 |
| Gaussian uncertainty | [CatBoost](https://catboost.ai/docs/en/references/uncertainty) 支持相应不确定性预测 | Normal benchmark 不能只对比 NGBoost |
| 新一代表格模型 | [TabPFN-3 官方报告](https://priorlabs.ai/technical-reports/tabpfn-3) 强调规模扩展、推理效率和校准分布 | “表格 foundation model 只能处理小样本”已不能作为路线假设；官方性能自述仍需独立比较 |

竞争资料能证明替代方案存在，不能证明哪个市场有付费需求。下面的场景优先级是根据当前代码匹配度、可获得的数据和验证成本作出的判断。

### 4. 选择一个能证明价值的初始场景

**第一候选：保险频率/损失建模的模型开发与验证。** 初始用户是愿意试验的精算研究者、保险数据科学家、风险建模咨询团队，而非先做完整企业定价平台。

候选理由是 exposure、非负/计数目标、分布参数和尾部评估与现有实现契合，并有公开数据与成熟基线。scikit-learn 的 freMTPL2 示例已展示 Poisson 频率、Gamma severity 与 Tweedie 纯保费路线，可以直接建立可审查的比较框架。[官方案例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html)

必须问清楚用户究竟需要什么：若只需要期望纯保费，已有 GLM/GBDT 可能足够。OpenBoost 应重点验证“同时使用多个分位数/阈值概率、研究异质 dispersion、按 exposure 检查概率质量”等任务是否能减少工程工作或改善决策。单一阈值也可能直接用分类器解决，不能自动假定需要完整分布。

第一份真实案例建议先做 **Poisson 频率 + exposure**，然后比较更灵活的计数分布；severity 单独建模与验证。Tweedie 在本地的训练 dispersion 梯度是近似、quantile 的正值部分也用矩匹配 Gamma 近似，`nll()` 采用另一条密度计算路径，因此其尾部不能仅凭接口或若干 shape 测试背书。[分布实现](../src/openboost/_distributions.py)

特别要区分 exposure 的统计语义：当前 offset 验证的是均值按 exposure 缩放；这不等于已经验证 Gamma/Tweedie 的整个分布满足任意 exposure 聚合规律。应明确 count、per-claim severity、aggregate loss、annualized loss 的目标与权重定义，再比较 dispersion 和尾部。[当前 exposure 实现](../src/openboost/_models/_distributional.py)

| 场景 | 现在的用途 | 升级为重点的条件 |
|---|---|---|
| 保险频率/损失 | 首选真实价值验证 | 有外部团队提供评价目标、能运行基线并愿意重复使用 |
| NGBoost 用户的大样本概率回归 | 最直接的开发者采用入口 | 在真实数据、相近预测质量下有明显计算或工程收益 |
| 需求/库存概率预测 | 保险无法触达时的备选，或后续扩展 | 有合作数据与明确缺货/库存成本；使用时间切分，并与现有 forecasting 流程比较 |
| FormulaBoost 的结构化曲线 | 有投入上限的研究实验 | 真实任务确实有可信公式、变化充分的结构输入和可识别参数 |
| Weibull 时间到事件 | 有领域合作时再提升 | 独立真实生存数据、正确删失评估，以及 lifelines/NGBoost 等强基线 |

预测场景已有 [MLForecast 的区间工作流](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/prediction_intervals.html)。采用该备选时应提供可接入的回归组件，避免顺势扩张为完整时间序列平台。

首选场景要允许被用户触达成本推翻：若两周内没有有效保险合作线索，但有明确的需求预测或可靠性团队愿意投入数据与时间，应优先跟随可验证的需求。

### 5. 把 adoption 做成一个完整工作流

建议稳定的最小体验：安装 → 运行公开真实案例 → 接入自己的数据 → 训练/独立校准/测试 → 对比基线 → 保存并重载 → 复现报告。

1. **一个默认入口。** 用户选目标类型、提供数据即可得到第一份概率质量报告。研究 primitive、其他模型和实验 backend 放在后续路径。线上 README 已完成一部分定位调整，应以当前线上状态继续，不要重做旧版审计。
2. **一个可信的报告模板。** 输出 mean/quantiles/intervals、CRPS/NLL、coverage 与 width、预先定义的分组/尾部检查、模型和数据版本、运行 backend。它首先服务这一个工作流，不必立刻成为跨所有模型的 dashboard 产品。
3. **训练与校准分离。** 校准集不用于最终评分，也不重复承担不受控的调参；检查独立测试上的分布质量。当前 `PITRecalibrator` 只暴露 PIT/CDF level 的 transform，尚不等于可保存、可采样、可输出校准 quantile 的完整 predictive distribution 产品。[本地实现](../src/openboost/_utils.py)
4. **生产前的小边界。** 先保证声明支持的 seed、weights、exposure、缺失/类别处理、early stopping、CPU inference、save/load 和版本兼容；明确 backend fallback。保留用户可以在普通 CPU 上完成试用的入口。
5. **把已有生态变成入口。** 先完善 sklearn 兼容和 NGBoost 迁移案例，再按用户需求选择一个 skpro 或 MAPIE 适配。上游是否接受需要维护者确认，不应假设一个 PR 自动带来流量。

“15 分钟获得第一份有用报告”可作为设计目标，必须由新用户实际操作测量。避免把额外原始运行次数、复杂参数或不必要依赖当成用户必须理解的概念。

### 6. Impact 的研究方向：结构能带来什么

FormulaBoost 值得保留，因为它把领域给定的函数形式 `f(theta(z), x)` 与非参数的异质参数结合起来。研究问题应是：**在怎样的数据支持、公式误差和参数耦合条件下，它改善结构输入方向的外推、参数恢复或决策？**

不能直接把 varying coefficients 或 full natural gradient 当作原创。已有 [tree boosted varying coefficient 研究](https://arxiv.org/abs/1904.01058)、[gamboostLSS](https://search.r-project.org/CRAN/refmans/gamboostLSS/html/mboostLSS.html)，NGBoost 也已有 natural gradient。它们构成进一步查新和对照的起点，而不是本次已经完成全面 novelty review 的证明。

几个会影响论文和市场定位的具体纠正：

- 线上 survival 文档称 NGBoost 没有 censored likelihood；这与 [NGBoost 官方 survival 文档](https://stanfordmlgroup.github.io/ngboost/1-useage.html#survival-regression) 不符。线上 README 也把 NGBoost 写成 fixed catalogue，但其[开发指南](https://stanfordmlgroup.github.io/ngboost/5-dev.html)允许新增分布和 score。这些说法应在推广前修正。
- “Weibull shape 随 covariates 变化”不是整个生态里独有的能力。[lifelines ancillary regression](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#modeling-ancillary-parameters) 已支持 shape 建模，并提醒这种扩展通常不再具有标准 AFT 的形式。应强调并验证非线性参数面、计算和使用体验，而非无条件的唯一性。
- XGBoost custom objective 的 Hessian 输入确有对角结构约束，但官方也讨论替代曲率和近似。接口不接收完整矩阵，不等于任何外层算法都无法利用耦合方向。[XGBoost 官方说明](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html)
- 线上 FormulaBoost 表格自身显示 `diag` 与 `full` 的预测/外推误差接近，`full` 的线索主要在参数恢复；不能由合成例子直接推出 full GGN 对真实任务必需。[报告](https://github.com/jxucoder/openboost/blob/main/docs/benchmarks.md)

对当前标量 MSE FormulaObjective，还有一个可直接推导的事实。对单样本 Jacobian 向量 `j`、残差 `r`、正 damping `lambda`：

```text
G = j j^T
g = r j
(G + lambda I)^(-1) g = r j / (lambda + ||j||^2)
```

因此当前逐样本 full GGN 的数学方向可化为样本级梯度缩放。它与按分量分别缩放的 `diag` 不同，但该形式本身并不要求通用稠密矩阵求逆。本次独立数值检查在 K=2/5 下与 dense solve 的最大绝对误差为 `2.78e-16` / `5.00e-16`。这个结论只针对当前标量 MSE、正 damping 的逐样本数学形式；不是整个训练系统等价证明，也不覆盖多输出残差、聚合曲率或任意数值 safeguard。[对应源码](https://github.com/jxucoder/openboost/blob/main/src/openboost/_objectives.py)

研究对照应包含同一个 formula、相同初始化/调参预算下的 global、diag、full、梯度缩放实现、合理的结构化替代模型，以及不带公式的基线。增加公式设错、参数弱识别、结构输入变化不足、参数化变化和不同 noise 情况。单样本标量响应的 Jacobian rank 至多为一，矩阵预条件不会凭空补足参数识别的信息。

若得到正结果，再构建“结构化 boosting 在何时有用”的技术报告和独立可运行实验。软件论文可在外部使用与软件质量积累后考虑；[JOSS 要求](https://joss.readthedocs.io/en/latest/submitting.html)提供审查标准，但发表本身不能代替采用验证。若用户把论文影响列为首要目标，可提高此项投入，同时收窄到一个明确问题。

### 7. 证据设计：必须回答用户的切换理由

建议分别回答三个问题，避免混用结果：

| 问题 | 实验 | 能支持的结论 |
|---|---|---|
| 一般概率质量是否可靠？ | 完成 ScoringBench 正式协议，使用其公开比较流程 | 在该协议与完整数据集合中的质量表现 |
| 放大后是否划算？ | 至少三个真实数据集、多规模/重复、CPU/CUDA 与强基线 | 指定硬件、质量门槛下的成本与规模优势 |
| 用户为何采用？ | 一个真实 domain 案例，加外部团队复现/继续使用 | 具体工作流的工程收益、决策收益和迁移成本 |

[ScoringBench](https://github.com/jonaslandsgesell/ScoringBench) 是合适的独立质量入口；本地已有[适配器和两套协议](../benchmarks/scoringbench/README.md)。正式质量协议和放大样本的 extension 必须分别标注。质量 benchmark 完成不应成为开始用户访谈的前置条件。

比较应包括相关的 NGBoost、XGBoostLSS/LightGBMLSS、PGBM、CatBoost uncertainty 或 quantile + conformal；资源允许时纳入当前可获取的 TabPFN。先按目标/分布支持筛选候选，再分阶段缩减，记录任何不支持、失败和计算预算差异。

提前约定 primary metric、质量非劣界限、数据划分、调参预算和业务决策量；报告 paired differences 与不确定区间。`p > 0.05` 不能证明等价或非劣。校准改善还要看区间宽度、proper scores 与下游损失，不能通过无限加宽区间获得一个无用的“校准胜利”。

新 GPU 结果至少附带：精确源码 SHA/dirty state、数据版本/hash、split/seed、OS/CPU/RAM/threads、GPU/driver/CUDA、依赖版本、完整命令、冷启动/热运行策略、实际 backend/fallback。统计端到端 fit、prediction、内存和 transfer；若比较 GPU OpenBoost 与 CPU NGBoost，应同时记录资源和成本，并加入相关 GPU 候选。合成数据可检验机制，真实数据才能建立场景证据。

生存任务还需处理删失：真实数据不能直接用未知的完整事件时间算普通 coverage；根据可识别条件选择 censored NLL、适用的 IPCW/Brier/校准方法及独立删失假设。FormulaBoost 的情景外推也不自动具有因果解释，不能仅凭观测数据上的曲线把改变价格/剂量解释为干预效果。

### 8. 90 天执行计划

| 时间 | 主要工作 | 交付物 | 建议决策门槛 |
|---|---|---|---|
| 第 1–14 天 | 核对本地/线上状态；整理已知正确性与证据缺口；准备一个 domain walkthrough；开展用户发现 | 版本/能力表、一个可运行案例、访谈提纲和基线协议 | 接触约 10 个合适对象；至少 3 个愿意给出真实评价任务或投入一次对比 |
| 第 15–30 天 | 跑完整第三方质量实验；完成 Poisson/exposure 真实基线；观察新用户上手 | 原始质量结果、失败清单、安装到报告的实际记录 | 至少 2 个外部用户独立跑通；明确最常见的 3 个阻碍 |
| 第 31–60 天 | 按阻碍改善工作流；冻结 CPU/CUDA parity 与规模结果；做一个小规模 FormulaBoost falsification 实验 | 可重复使用的核心版本、真实 case study、结构化方法对照 | domain 有一项足以覆盖迁移成本的收益，或明确停止该场景；研究方向有证据才继续 |
| 第 61–90 天 | 根据结果准备稳定发行、一个生态集成、公开技术案例与合作试点 | 版本化教程/报告、上游提交准备、可复用 onboarding | 累计 5 个外部团队跑过真实数据，其中至少 3 个四周后继续或用于第二个任务；1 个独立复现/集成 |

上表的外部联系、上游提交和发行均为未来行动建议，本次没有发送消息、发布或提交外部 PR。正式 benchmark 的上游接受时间不可控，应区分“提交材料完成”与“已接受”。

小团队的默认投入建议：约 60% 做核心可靠性与完整体验，25% 做外部使用/真实证据，15% 做 FormulaBoost 的单一研究实验。数据/计算任务可在同一个人的工作安排中交错进行；这不是要求同时启动三个产品。

每个新增功能必须对应一个外部使用阻碍或一个预先写明的研究假设。GAM/DART/linear leaves、Ray/multi-GPU/out-of-core/train-many 不获得独立产品路线的投入，直到主线 evidence gate 满足且有具体需求。

### 9. 获得用户与商业价值

最初的传播材料建议只有三类：一个真实概率建模 walkthrough、一篇可复现的性能/质量比较、一份 NGBoost 迁移说明。标题围绕用户问题，例如“把 exposure、预测分布和独立校准评估放进同一条 Python 流程”。所有数字链接原始结果。

优先接触三类潜在用户：已有 NGBoost 项目的作者/维护者、做计数与损失建模的研究团队、愿意提供验证任务的风险建模咨询团队。提供具体的配对实验与复现帮助，记录他们放弃或继续使用的原因。无需先追求大规模发布流量。

访谈只围绕已发生的工作：最近一次概率预测怎样完成？最耗时/最不可信的一步是什么？输出实际影响哪个决定？现在用什么替代方案？什么结果值得切换？能否在两周内用自己的数据完成一次比较？口头赞同的证据弱于投入数据、工程时间和再次使用。

商业化建议按验证顺序推进：

1. 可审计的 benchmark/迁移/校准评估服务，检验是否有人愿意为缩短交付周期付费。
2. 若多个团队反复提出同类需求，再考虑维护支持、私有环境部署、版本与报告复现。
3. 出现明确持续付费需求后再评估托管训练/评估产品。

价值估计使用“节约的工程时间 + 计算成本变化 + 经验证的决策改善 − 集成/维护成本”。不要把 CRPS 百分比机械换成收入，也不在缺乏访谈和成本信息时编造 TAM、定价或 ARR。

核心开源库保持低摩擦使用。先靠对共同问题的可靠解决建立信任，再判断哪种服务值得收费；不要为了假设中的商业化提前增加使用门槛。

### 10. 会推翻路线的条件

| 观察到的结果 | 应采取的动作 |
|---|---|
| 通用概率质量与强基线接近，但没有计算或工程收益 | 收窄到确有差异的 domain/custom objective；停止“普遍更好”的叙事 |
| 用户只需要区间，已有模型 + MAPIE/skpro 足够 | 提供轻量集成或承认替代方案更合适；不扩张为通用 UQ 平台 |
| 用户需要完整报告但始终不愿替换训练器 | 先用小 adapter 验证评估组件的独立需求，再决定是否改变产品重心 |
| GPU 在真实端到端 workload 没有经济优势 | 把 GPU 降为可选能力，依据 profiling 优化；暂停多 GPU 扩张 |
| FormulaBoost 只在生成公式完全正确的合成数据上有效 | 保留研究例子；暂停将外推能力作为通用产品卖点 |
| full GGN 优势在相同公式、初始化和调参后消失 | 把贡献放回公式结构/体验；不将矩阵形式当作独立创新 |
| 连续几轮 onboarding 仍无第二次使用 | 回访退出原因，调整场景/体验；不靠再加模型种类解决 |
| 多个团队重复使用并愿意支付支持费用 | 增加该场景的维护与集成投入，再讨论更大产品 |

## Changes

- 新增本文，保存研究依据、版本差异、建议路线、实验门槛与尚未验证的假设。
- 在 learning index 添加入口。
- 本次没有修改模型、运行库、公开产品定位或 benchmark 实现。

## Verification

### 本地功能抽查

```bash
OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache \
  uv run --no-sync pytest tests/test_distributional.py tests/test_utils.py \
  -n 0 -q -k 'exposure or PIT or ReliabilityDiagram or ProbabilisticMetricsWithOpenBoost'
```

结果：`26 passed, 136 deselected in 7.21s`；Darwin、Python 3.12.12。只说明所选本地行为通过，不代表完整回归、线上新架构或 CUDA 通过。

### 局部可复现性发现

下面的独立进程实验使用固定数据，仅改变 NumPy 全局 seed；本地 NaturalBoost 构造参数没有 `random_state`，且树的 row subsampling 使用全局 `np.random.choice`。预测最大变化是 `0.1888485550880432`。这是本地已复现的随机性控制缺口，尚未据此判断线上版本是否已修复。

```python
import inspect
import numpy as np
from openboost import NaturalBoostNormal

rng = np.random.default_rng(7)
X = rng.normal(size=(240, 4)).astype(np.float32)
y = (X[:, 0] + 0.4 * rng.normal(size=240)).astype(np.float32)
preds = []
for global_seed in (1, 2):
    np.random.seed(global_seed)  # Deliberately probe reliance on global state.
    model = NaturalBoostNormal(n_trees=8, max_depth=2, subsample=0.7)
    model.fit(X, y)
    preds.append(model.predict(X))
print('random_state' in inspect.signature(type(model)).parameters)
print(float(np.max(np.abs(preds[0] - preds[1]))))
```

### FormulaObjective 数学检查

接续以上代码的 `rng` 状态运行，验证标量 MSE 下的 rank-one 恒等式；没有执行线上模型的完整训练：

```python
for k in (2, 5):
    J = rng.normal(size=(100, k))
    residual = rng.normal(size=100)
    damp = 1.0
    matrices = J[:, :, None] * J[:, None, :] + damp * np.eye(k)[None, :, :]
    g = residual[:, None] * J
    full = np.linalg.solve(matrices, g[..., None])[..., 0]
    simplified = g / (damp + np.sum(J * J, axis=1))[:, None]
    print(k, float(np.max(np.abs(full - simplified))))
```

结果：K=2 时 `2.7755575615628914e-16`；K=5 时 `4.996003610813204e-16`。这是代数检查，不是质量或性能 benchmark。

文档核查：9 个相对文件链接存在、2 段 Python 示例可编译、7 个模板章节齐全，代码块成对闭合；`git diff --check` 通过。本文不在 MkDocs 导航内，未修改生产代码，未将文档检查描述为模型回归测试。

## Failed Attempts

- 浏览器搜索索引中的线上 README、raw 文件和提交历史相互不一致；改用 GitHub connector 读取源文件，并单独记录本地测试 SHA。
- `git ls-remote` 因本地环境无法解析 GitHub 域名而失败；通过可用 connector 完成源文件读取，没有因此更改网络设置或仓库状态。
- 不能从新版 benchmark 转录表格确认完整原始 provenance；没有据此重复宣传 GPU 数字。

## Risks and Follow-ups

- 最大未知是用户需求与留存，不是候选模型数量；下一步优先得到可评价的外部任务。
- 核对线上与本地代码状态后再实施任何修复，保留已有未推送提交。
- 首批工程检查应覆盖 seed、概率输出与校准的一致性、Tweedie 近似边界、持久化和真实 CUDA parity。
- 本文没有宣称 FormulaBoost 的研究新颖性已成立、OpenBoost 已被 ScoringBench 接受、GPU 性能已独立验证或存在已验证的付费市场。
- 若主目标切换为研究论文或短期商业收入，应相应调整实验顺序；不要机械沿用默认投入比例。

## Commits

- 本文和 learning index 作为一个独立文档提交；提交 SHA 由 Git 历史记录。
