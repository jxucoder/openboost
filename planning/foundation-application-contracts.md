# OpenBoost v1 应用覆盖与案例契约

日期：2026-09-05。状态：用户要求纳入的设计与验收范围，**不代表新架构已实现**。
补充 [v1 主计划](agent-boosting-foundation-plan.md) 与
[E0–E7 eval](openboost-v1-evaluation.md)，代码审阅基线 `f30c2ed`。

用户明确：**以下用例都需要，全部属于 v1 必需范围，逐项交付和验收**。
每个用例有自己的目标语义、recipe、真实工作流和独立证据。保险/AFT 与其他用例
地位相同。保持 foundation-first 与不要求 backward compatibility 的方向。

## 0. v1 必需应用矩阵 A1–A13

以下每行都是 required，不能从中任选若干完成。数据来源仍可选择，F0 固定版本/
hash、许可、切分、特征可用时点和任务定义；数据选择不改变用例的必需地位。
各用例通过组合共同组件实现，不要求按行业复制 trainer。

| ID / recipe | 必需用例 | 数据选择与必须交付的行为 | 独立验收 / 主评价 |
|---|---|---|---|
| A1 / R1 | 连续量回归 | 已有 California Housing 冻结数据或其他真实回归来源；训练、预测、保存/加载 | numeric/missing、权重、泛化误差、RMSE；声明地理切分限制 |
| A2 / R1 | 二分类 | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult)；含类别/缺失和概率输出的完整流程 | category mapping、unseen policy、class weights/link；log-loss + 辅助 AUC |
| A3 / R1 | 多分类 | [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype)；类别映射、K 类参数和概率输出 | softmax、概率归一化、各类表现、multi-logloss；线程/显存 |
| A4 / R3 | Group ranking | [Microsoft MSLR](https://www.microsoft.com/en-us/research/project/mslr/)；pairwise 与 NDCG-weighted lambda 两条 recipe | query/pair、组内依赖、官方 folds、NDCG@10；保留数据使用条件，不跨 query 切分 |
| A5 / R2 | 分位数回归 | [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275)；预声明分位数与加权叶求解 | temporal split、weighted pinball；删除 casual/registered 等目标组成字段，核对特征可用时点 |
| A6 / R8 | 多输出 | [UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons%2Btelemonitoring) 两个 UPDRS 目标；独立树与 shared topology/vector leaf | subject 级切分、原数据评分/插值语义、各目标误差；基准预测不等于临床有效性 |
| A7 / R4 | 事件计数 / 频率 | frequency 数据上的 Poisson + exposure；交付 count 与单位 exposure 的 rate 预测 | offset/权重只生效一次、预测单位、Poisson deviance；exposure 缩放反例 |
| A8 / R4 | 正值金额 / 严重度 | severity 数据上的 Gamma；声明 claim 级或保单平均目标 | 正值域、样本选择、claim 权重、Gamma deviance；不能以 A7 通过代替 |
| A9 / R4 | 总损失 / 纯保费 | 关联 frequency/severity 数据；Tweedie 与 frequency × severity 流程 | 零值、power、aggregate/annualized 单位、Tweedie deviance；不能以 A7/A8 代替 |
| A10 / R5 | Censored survival / AFT | 真实随访或寿命数据；固定噪声族/尺度的事件与右删失流程 | 删失 likelihood、risk/time/survival 输出、censored NLL 与适用的概率/排序评价 |
| A11 / R6 | 分布预测 / NaturalBoost | 真实回归来源或正式 ScoringBench 子任务；Normal 双参数与方向/步长变体 | proper score、Fisher/普通方向、参数 link、calibration/width；不能只报 coverage |
| A12 / R7 | 结构化关系 / FormulaBoost | F0 选定有明确结构假设的真实数据；双参数 formula、link、结构输入、参数输出；另有参数恢复/错设实验 | 独立 Jacobian/方向、可识别性反例、真实任务预测质量及结构基线；合成实验不能替代真实任务 |
| A13 / R9 | 模型选择 / train-many | 在真实任务上训练并选择多个配置/目标/分组；M=1/8/32，报告所选模型 | prepared-data identity、独立 seed/停止/失败、所选模型质量、完整集合与模型选择成本 |

验收按 A1–A13 的身份逐项判断，不能用一个“至少若干任务”的计数替代覆盖。
合计至少6个独立来源；同一数据上的不同任务只算一个来源，不可用复制样本充规模。
数据不可用时，F0 选择具有相同语义要求的公开数据并留下理由；选定前该项待完成，
不能删除用例，也不能测完后换掉表现差的数据。每项 F0 任务卡补齐数据/目标/输出、
baseline、独立参考、CPU/CUDA 边界、质量与成本指标、实现阶段及证据路径。
下面细化 A7–A10 的目标语义；所有行适用相同的交付和验收要求。

## 1. 保险：频率、严重度与纯保费分别定义

| 任务 | 目标与预测单位 | 初始强基线 | 必须明确的语义 |
|---|---|---|---|
| 出险频率 | 给定保障期的 claim count；单位 exposure 的 rate | Poisson GLM；XGBoost `count:poisson` | count + log-exposure offset，或经过核对的 rate + exposure weighting；两个方案分别定义，不能重复施加 exposure |
| 赔款严重度 | 已发生且符合定义的 claim 的正赔款金额；或保单级平均每次赔款 | Gamma GLM；XGBoost `reg:gamma` | claim 级与保单平均 severity 的训练权重不同；零赔款、理赔次数与选择规则单独记录 |
| 纯保费 / 总损失 | 每单位 exposure 的预期损失；或保障期内的预期总赔款 | frequency × severity；XGBoost `reg:tweedie`；Tweedie GLM | 区分 annualized loss、aggregate loss 和零赔款；冻结 power、offset/weight 方案和预测单位 |

XGBoost 官方明确展示以 `base_margin` 表达 Poisson 的 log-exposure offset：
`E[count | X, e] = e * exp(F(X))`。基线在训练、验证和预测时都必须使用相应 offset；
报告 rate 时也应明确 exposure=1 的含义。
[官方 offset 说明](https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html#offset)。
Poisson/Gamma/Tweedie objective 的实际配置应按冻结版本核对。
[XGBoost 参数](https://xgboost.readthedocs.io/en/stable/parameter.html)。

候选真实数据为 freMTPL2freq / freMTPL2sev，分别对应 OpenML 41214 / 41215；
scikit-learn 的保险示例提供数据关联、目标定义和 GLM 比较起点。
该示例的截断、零值处理属于其处理选择，不能未经说明当作原始数据真相。
[官方保险案例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html)。

F0 固定数据版本/hash、claim-policy 关联、样本排除/截断、缺失/类别处理及切分。
同一保单/实体的关联记录不能跨训练测试泄漏；有有效时间信息时定义时间切分，
没有时明确限制，不杜撰时间验证。编码器和 bins 仅由训练数据拟合。

验收分两层：

- **数学与流程：** 非单位 exposure、非单位/零 sample weight、offset 与权重各生效
  一次；Poisson 中固定模型后 exposure 翻倍应使预期 count 翻倍，而 rate 不变。
  验证集、early stopping、save/load 后推理仍使用相同定义。
- **应用：** 相同单位下的适当 deviance、总量和预定义分组的预测/实际对照。
  对完整分布另评 NLL/CRPS/尾部等指标。仅预测均值的 Gamma/Tweedie objective
  不自动提供完整已校准分布；均值按 exposure 缩放也不证明整个分布的聚合规律。

A7、A8、A9 分别交付并验收。实现可按组件依赖排序，Poisson 小路径通过不意味着
Gamma、Tweedie 或组合工作流完成。

## 2. Survival / AFT：观察到的标签不总是事件时间

XGBoost `survival:aft` 使用下/上界表达完整观察及右、左、区间删失；AFT 噪声族和
全局噪声尺度由参数声明。必须使用冻结版本支持的标签入口，例如
`DMatrix` 的 `label_lower_bound` / `label_upper_bound`。
[官方 AFT 教程](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html)。

| 观察类型 | 标签语义 | 对连续时间模型的 likelihood 贡献 |
|---|---|---|
| 事件在 t 发生 | lower = upper = t，t > 0 | 事件密度 f(t) |
| 右删失 | lower = t，upper = +inf | 生存概率 S(t) |
| 左删失 | lower = 0，upper = t | F(t) |
| 区间删失 | 0 < lower < upper < +inf | F(upper) - F(lower) |

这些是待独立验证的数学定义，不以复用 production loss 作为唯一参考。
实现要使用稳定的 log-density、log-survival 和 log-CDF 差，测试极端尾部与窄区间；
相等端点必须按事件密度处理，不能计算零区间概率。声明 log-time 到 time 的 Jacobian、
梯度符号、曲率近似/裁剪策略；数学参考与数值稳定策略分开验证。

早期 recipe 固定一个噪声族及全局尺度，验证完整观察 + 右删失；F0 的目标 schema
必须能区分全部上述情况。未实现的删失类型显式拒绝。delayed entry / 左截断与
左删失含义不同，暂不支持时独立拒绝，不能把二者互相转码。

候选真实验证：XGBoost AFT 教程引用的 NCCTG lung，或 scikit-survival 的
Veterans' Administration lung cancer 数据。F0 选择并冻结一套数据、版本、事件编码、
时间单位、预处理和切分；它们是小型真实 smoke / 质量比较，不代表规模优势。
应用还可来自寿命、设备故障、合同终止等 time-to-event 问题，需按实际事件定义任务。

基线至少包括 XGBoost `survival:aft`、CatBoost `SurvivalAft` 和适当的参数 AFT；
CatBoost 的上界 -1 哨兵需按其接口转换，且其官方表列此 objective 为 CPU。
[CatBoost 目标与设备边界](https://catboost.ai/docs/en/concepts/loss-functions-regression)。
Cox 可作为排序/风险对照，
按它支持的输出比较。不要直接把 hazard/risk score 当作生存时间或概率。
Cox 的风险集涉及行间关系，因此 F0 审阅时不要把“所有 objective 必须逐样本独立”
写成通用协议；原生 Cox 实现不是 F1 的前置任务。

评价要求：

- 适用假设下的 censored NLL、时间上的生存函数单调性、有效概率与分位数一致性。
- 区分排序与概率质量：C-index 之外，针对声明支持的右删失评价使用适当的
  IPCW Brier / integrated Brier 与时间点校准，记录删失假设及可评价时间范围。
  删失分布估计与时间网格在训练/开发阶段固定，不从最终测试结果调选。
- 不能把右删失时刻当真实失败时刻算普通 RMSE/coverage；也不能把右删失的 IPCW
  工具未经论证直接用于左/区间删失。不可评价的单元报告原因，不能静默删除。
  [官方生存评价指南](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)。

OpenBoost 当前 `WeibullAFT` 的入口是 observed time + event，面向右删失；不能当作
全部区间标签支持的证据。其随 covariates 改变 shape 的能力属于更一般的分布式
生存回归，不能仅凭类名声称仍满足固定残差分布的标准 AFT 假设。

## 3. 对 foundation 和执行阶段的具体约束

1. **F0.1：** 覆盖 R1–R9/C1–C7/A1–A13，逐项填写输入、目标、输出、三大库
   配置、质量指标及数据处理协议。F0.2 含 group/多输出/link/加权分位数/exposure/
   删失、分布/formula 和多 run 等独立参考；每个用例都要能独立判错。
2. **F1.1：** Problem 保留 typed target、offset/exposure、sample weight 和标签边界；
   共用切分/索引必须保持对齐。上界 +inf 可合法，不能沿用所有标签必须 finite 的规则。
3. **F1.5/F1.6：** 接通各用例的数据/目标语义并完成全部 CPU recipes；至少两轮
   验证预测、接受状态和新格式 round trip；runner 无需按应用名增加专用分支。
4. **F2：** 任务变化可来自保险统计约束或 survival 更新/曲率规则；先比较既有配置，
   正确完成基础任务与新算法改善质量分别报告。
5. **F3：** GPU 能力表分别列 offset、各删失类型、评价与预测支持；只通过 Normal
   GPU 路径不能宣称保险/AFT 也通过。声明支持时跑对应 CPU/GPU 语义与任务质量检查。
6. **F4：** A1–A13 全部完成真实任务验收；每项记录质量、成本、失败和未支持边界，
   任意一项缺失或失败都不能宣布完整 v1 通过。

本次只更新设计、任务范围和执行门槛；没有下载数据、执行基线或修改训练代码。
