# OpenBoost v1：验收与 eval 协议

版本：**v1-plan-r2 / 2026-09-05**。状态：预声明的设计门槛，尚未执行新 v1 评测。
范围修订：用户明确全部用例都需要；以 A1–A13 逐项验收替代此前八个代表任务的门槛。
与 [v1 主计划](agent-boosting-foundation-plan.md) 联合使用。
数字是本次提出的验收标准，不是既有成绩；F0 固定具体数据、实现与资源 manifest 后
执行。测量后改阈值须新建协议版本，旧结果保留，不能回填成通过。

## 1. 结果状态与可交付物

每个 case 记录 `not_run / pass / fail / unsupported / error / timeout`。
required case 中任何非 pass 状态使对应 gate 未通过；optional 的 unsupported 仅能
支持能力边界说明。没有设备产生 `not_run`，不算 GPU pass。缺少预期 case 是失败。

F0 需实现 `benchmarks/v1/` 下的 manifest、runner 和判卷器。
Sprint 011 已交付[产物完整性检查](../benchmarks/v1/README.md)，仅验证声明矩阵/缓存身份/文件；
真实manifest冻结、runner及独立质量/E-gate判卷仍未完成，不能以integrity_pass代替gate通过。
每次 run 的不可变目录至少含：

- `manifest.json`：协议 hash、代码 SHA/dirty、数据版本/hash/行与 split ID、目标/
  预处理定义、版本及 wheel/build hash、OS/CPU/RAM/线程、GPU/驱动/CUDA、资源预算、
  完整 CLI、种子、A-ID→recipe/test/artifact 的映射、任务×模型×fold×device 预期矩阵。
- `cases.jsonl`：逐单元状态、有效构造参数、后端/回退、计时范围、预测/模型 hash、
  指标与失败原因；不能只存平均数。Agent case 另有 prompt/tool/model/budget hash。
- `report.json` / `README.md`：各 gate、比较表、最差 case、置信区间与未验证项。
- 独立 evaluator、完整 stdout/stderr/JUnit 或等价记录；引用 parent run 时核验 hash。

判卷器自身先用坏记录测试：缺 fold、换配置后复用 cache、非法 NaN、错误 backend、
超时、被跳过 GPU、worker 非零退出、重复 case、缺原始预测均应使 required gate 失败。
训练代码不能自行宣布自己通过。缓存键包含代码、数据切分、预处理、全部配置与协议。

## 2. E0：覆盖和接口语义

以主计划 R1–R9/C1–C7 和应用矩阵 A1–A13 为 required 范围。每项有接口、CPU/CUDA 状态、失败语义、
测试入口和 artifact 链接，**100% 有记录，required 项 100% 通过**。
每个 A-ID 都须在 manifest 和报告中出现，且有真实任务、对应实现和独立检查。
不执行没有意义的全部笛卡尔积；但不能挑若干 A 项替代其余用例。F0 列出的 required
cells 不能在看到结果后删除。能力支持与任务质量分开，不能互相抵消。

## 3. E1：数学、状态与持久化正确性

| 类别 | 独立参考 / 反例 | 默认验收 |
|---|---|---|
| Objective / geometry | float64 公式、有限差分、必要的小矩阵 solve；loss/gradient/有效曲率语义分别核对 | 平滑且良态的小 fixture：`rtol=1e-6, atol=1e-8`；非光滑点验证预声明 subgradient/叶最优性，不滥用有限差分 |
| Histogram / route / split | 逐样本归约，独立枚举候选；缺失、类别、空节点、零权重、额外统计、ties | 整数 count/routing 完全相等；CPU 数值 `rtol=1e-7, atol=1e-9`；按确定的 tie policy 比较最优候选 |
| Leaf / topology | 加权 Newton、加权分位数、标量/向量叶、三种 grow policy | 与独立参考一致；至少两轮，改动必须传到下一轮状态；不能只有 shape 测试 |
| Run / update | 联合/依次更新、回溯拒绝、失败重试、早停、RNG、M=1/8/32 排列 | 按稳定 run ID 比较；拒绝无残留；重排/分组不改变该 run 的 seed 语义；不能隐藏失败 |
| Persistence | numeric/missing/category/vector-leaf/coefficients、output schema、offset/link、formula 依赖 | 新进程 load 前后预测一致；损坏 shape/version/index 明确失败；旧格式可显式拒绝 |
| Dataset semantics | query/entity split、category mapping、unseen policy、标签上下界、offset 与 weights | 无跨 split 数据学习、无字段错位、无双重权重、无静默丢字段；合法删失 inf 与非法 NaN 区分 |

同 CPU 平台、线程、版本与 seed 的 deterministic reference 重跑应完全一致。
GPU 浮点归约不要求逐位相等：中间 float32 数组默认 `rtol=1e-4, atol=1e-5`，
final raw prediction 默认同一阈值；任务指标差 `<=1e-3 * max(1, abs(CPU metric))`。
极端数值 fixture 按稳定参考和范围单列阈值，在运行前冻结，不能因失败统一放宽。
近似相等候选的 GPU 拓扑允许不同，但需先定义 tie band、证明均为容差内最优，
并通过最终质量；非 tie 的分裂错误不能用 final score 接近掩盖。

## 4. E2：算法表达能力

R1–R9 的可运行 reference recipes 使用安装后的公开 v1 接口，不导入 private module。
同时检验以下替换：

1. 更换 loss 或参数几何；现成库已能做的任务保留为 control。
2. 自定义候选可行性/评分与生长顺序，复用 histogram/route。
3. 改变叶求解，如从平滑近似切到加权残差分位数；可取得所需 routed rows/statistics。
4. 多参数候选的自适应步长/拒绝，改变参数更新顺序。
5. 共享 prepared data 的多 run 执行，独立 RNG、早停、错误和结果。

验收：五种修改全部有数学/状态测试，无 core edit、无 private import、无对任务名
写死的 runner 分支。独立包安装通过，且自定义组件实际参与执行。
此 gate 证明表达力，不证明新算法更好或比对手更容易使用。

## 5. E3：真实任务质量与适用范围

### 数据覆盖与分割

F0 为 **A1–A13 每项**固定真实任务与验收入口：回归、二分类、多分类、ranking、
quantile、多输出、Poisson 计数、Gamma 正值金额、Tweedie/组合总损失、survival/AFT、
NaturalBoost 分布预测、Formula 结构化建模，以及 train-many 模型选择工作流。
A1–A12 各有真实预测质量结果；A13 有真实模型选择结果及 E4 的完整集合成本。
不能用 Poisson 代替 Gamma/Tweedie，也不能以普通回归分数代替分布或 Formula 验收。

数据合计至少6个独立来源。同一数据可承担多个目标不同的用例，但不算多个独立
dataset；同一组预测换标签/重复计数不算新用例。A13 复用数据也不增加来源数。
Formula 必须同时有合成识别/错设反例与真实任务质量比较；选定前该项未完成，
不能降级成以后再做的案例。数学模拟属于 E1，不能代替任何 A 项的真实任务证据。

候选来源由 [应用矩阵](foundation-application-contracts.md) 给出；F0 下载/核验许可与
版本后固定 dataset IDs 和原始 hash。没有可用数据就记录 unresolved，不能用复制
样本充作真实规模。数据选择可调整，但全部用例必需；unsupported/not_run 不能
使该项或 E3 通过。

一般任务固定 5 个 split seeds（0–4），train/validation/test = 60/20/20；按任务
使用 stratified/group/time split，官方固定 split 优先。ranking 不跨 query；重复
实体不跨 split；时间任务用预声明 rolling origins。5 次不是必然独立同分布样本。
最终 test 不用于选参数、预算、阈值或决定删除 case。

### 公平基线

候选版本为 XGBoost 3.4.1、CatBoost 1.2.10、LightGBM 4.7.0，参见
[release/plan 核对](boosting-release-review-2026-09-05.md)。每个任务选其实际支持的
objective；额外纳入合适 GLM/AFT、NGBoost、Py-Boost 或简单结构 baseline。
unsupported 不是劣分；例如不能要求 CatBoost SurvivalAft 跑其未声明支持的 GPU。

先保留一组显式有效默认配置；主比较每方法 **16 个预声明配置**，在 validation
选择配置和 baseline 方法，再解封 test。搜索空间按方法定义且核对是否合理，不强行
把对称树和 leaf-wise 树约束到相同 depth。当需匹配规模时比较叶预算及质量。
同时报告总模型选择成本；固定次数与固定时间预算的结论不能混为一个实验。
每 trial 的 CPU/GPU/时间/内存上限由 F0 manifest 固定；超时计失败，不免费追加重试。

### v1 质量底线（标准 recipe）

这里比较已选定的对手，而非事后挑 test 最优结果。分任务预声明主指标：

| 主指标 | 每个任务跨 split 的门槛 |
|---|---|
| 非负损失：RMSE/log-loss/pinball/CRPS/deviance/Brier | 候选/基线的 median ratio `<=1.05`，最差 split `<=1.15` |
| 可为负的 NLL / censored NLL | 统一参数化、单位、归一化与 likelihood 常数后，平均每行差 median `<=0.02` nats，最差 `<=0.10` nats |
| NDCG@10 | 基线减候选的 median `<=0.01`，最差 `<=0.03` |

ratio 只用于 baseline loss > `1e-8`；否则用绝对差 `<=1e-8`，并标记 near-perfect
case 不参加 ratio 总结。零/负值不能取不合法几何平均。以上是工程门槛，不能宣称
已经证明普遍不劣于三大库；A1–A12 各自的预声明质量门槛都需通过，不让简单任务
平均掉失败任务。A13 的所选模型须通过对应原任务的质量门槛，并验证选择过程无泄漏。

分布/生存补充 proper score 与校准；coverage 必须连同 width，生存按删失假设与
支持区间评价，不只看 C-index。量化指标不代替目标语义。报告每个 split、paired
差和任务级区间；小任务集合不推导全市场优势。

自定义新算法单独报告：数学通过而预测质量变差是合法的研究结果，但不能用于
质量胜利宣传。v1 研究价值至少需一个 E5 的修改成本收益；算法优于 baseline 的
宣传还需对应真实任务、冻结协议和可复现质量收益。

## 6. E4：GPU / train-many / 推理成本

同时测 semantic reference、优化的同算法实现及质量匹配的外部基线。
固定同机、线程/数据/缓存政策，first-fit 1 次 + warm-fit 3 次，至少 3 seeds；
禁止在正式 timing 中混入 profiler 或显存采样线程。报告分箱、目标计算、传输、
JIT、训练、验证、导出和预测；可另报各阶段但不能替代端到端。

必需 workload：小数据启动/单条预测；两个真实来源的中/大数据（至少一个达到
100k 行）；一个 K>1 输出任务；一个 M=1/8/32 的模型集合。F0 固定形状/资源，不能
看到结果后改变 N、K、M。旧 T4 P7 单独保留，原 1.2 阈值仍报告，不回写历史。

工程成本门槛：

- 设备支持表内的 required recipes 无静默 fallback，E1 CPU/CUDA 全部通过。
- 两个中/大任务的标准 recipe：相同质量门槛下，warm fit median 不超过最快合格
  外部 GPU baseline 的 **2 倍**；任一超出则该 GPU 性能 gate 失败。
- 公开组合路径相对同语义优化实现 warm-fit ratio **<=1.25**；不能以忽略插件获得。
- train-many：M=1 相对独立路径额外时间 **<=10%**。F0 固定至少一个真实模型集合，
  对同一集合分别测 M=8 与 M=32：至少一个 M 相对共享预处理的顺序参考有 **>=20%**
  的完整集合时间降低，另一个不得慢超过10%；其余预声明 required 集合也不得慢超过10%。
  每个 run 的质量/seed/停止状态一致，显存不超过固定上限。只有分箱复用的收益单列。
- 新进程 import→load→predict 与 warm batch/single-row prediction 均计时并记录。
  标准 CPU 推理相对最快合格 CPU baseline median ratio **<=2**；自定义推理依赖
  单独记录。运行次数、输入 batch size 和时钟开销校正写入 manifest。

未满足成本门槛可以交付 CPU/experimental 结果，不能标为“v1 GPU 验收完成”。
不要求所有 research recipe 达到相同速度，但其完整成本都要记录。精确显存峰值、
device-wide 采样下界、进程 RSS 是不同指标；缺项留空并说明，不能把 0 当作没用内存。

## 7. E5：Agent eval

从 E2 五种修改各选一个开发任务，再准备两个未参与 API 设计的任务。
所有题有任务卡、数学/状态 verifier 和资源上限，候选实现不能修改判卷器。
试用期间改过接口的题转为开发集；保留任务一旦用于改接口就不再算未见测试。

对照为 OpenBoost 与 **F0 选定的最合适现有实现路径**，允许对手使用公共 hooks、
外层循环或源码修改；不得剥夺其文档、已有配方或正常工具。Py-Boost 和 GPU custom
objective 能力参与选择，不能只比较弱默认接口。原始 NumPy 可作诊断对照，单列成本。

每任务/每 arm **3 次独立尝试**；5 个开发题共15次，2个保留题共6次。
固定 Agent 模型/推理设置、工具、初始文档、缓存与提示词；每次最多 **30分钟、
20k生成 token**，任一先到即停止（token 统计口径及服务能否计量须在 manifest 明确）。
轮换 arm 顺序，禁止跨尝试泄漏补丁。大规模执行前先跑一题 smoke，核对计量与成本。

记录首次正确完成时间、安装/活跃/阻塞时间、token/工具/计算费用、人工提示、
core edit/private import、失败原因和独立正确率。失败按完整预算计入时间汇总；
另外保留成功样本时间，不只筛选成功者比较。版本或设置改变要另立 cohort。

门槛：OpenBoost 开发题 **>=12/15** 正确且每题 **>=2/3**，保留题每题 **>=2/3**
（总计至少4/6）；
没有核心修改/私有依赖；至少两种深层修改的 capped median time 比对手低 **>=30%**，
总体正确完成数不低于对手。已等效存在的 control 不要求 OpenBoost 获胜。
这些是小规模工程决策标准，不宣称统计显著；不通过时先查任务/文档/API边界，
不能仅增加提示或减少对手权限来通过。

## 8. E6：安装、交付与可重现

- 干净 CPU 环境、单 CUDA 环境分别从 wheel 安装；不存在源码目录/private import
  依赖，CPU 安装不要求 CUDA。Python/OS 的声明支持矩阵逐项通过。
- 两个独立扩展包 + 全部标准 recipe 可运行；训练插件卸载后的核心 raw/tree 推理
  通过。自定义 formula/link 只承诺其声明且已安装的推理依赖，不自动序列化任意闭包。
- 文档、signature、能力表和测试一致；每个主要错误包含组件/字段/设备/形状及原因。
- 结果判卷器能从 committed raw artifact 重建 gate；运行时失败使进程非零退出。
- 发布内容具备版本与许可，无私密数据。旧格式拒绝不算失败；新格式 round trip
  必须通过。发布、push、外部 leaderboard 仍需用户提出该外部动作。

**工程 v1 完成：A1–A13 逐项证据齐全，E0–E6 所有 required gate 通过。** 若只完成子集，称相应 milestone
或候选版，不称完整 v1；状态报告列出每个 gate 的证据路径，不能凭总测试数宣布通过。

## 9. E7：adoption / impact，独立于工程完成

经授权联系后，至少两位独立作者尝试，其中一位实现自己的方法并在另一任务复用；
记录安装、帮助、失败、使用原因及真实依赖。内部 Agent/仓库作者不计外部作者。
Impact 另记独立研究复用、下游方法包或实际任务决策收益，不由 stars/downloads 代替。
CPU 试用可提前开展，不等待所有 GPU 门槛。E7 未通过时明确采用假设仍未证实，
即使 E0–E6 已通过也不能说已获得 ecosystem 或普遍产品价值。
