# XGBoost / CatBoost / LightGBM：v1 设计前的 release 与计划核对

核对日：2026-09-05。本文件是公开一手资料的快照，未安装或运行这些新版本。
正式对比前要固定版本、wheel/build hash、CUDA 变体和实际配置。
搜索摘要中的旧版本号、`latest` 开发文档和未合并 PR 不作为已发布能力证据。

## 1. 最新正式 release

| 项目 | 当日 `releases/latest` 指向 | 发布记录与近期实际变化 | 对 OpenBoost v1 的影响 |
|---|---|---|---|
| XGBoost | **3.4.1**，tag `v3.4.1`，`6fe8c54` | patch 修复类别容器的模型切片和 JVM sparse batch prediction；发布说明日期 2026-08-14，GitHub 页面显示 Aug 15 08:30 | 类别编码、切片和保存后的推理同样属于正确性；记录两种来源的日期，不假定文档日期就是上传时间 |
| CatBoost | **1.2.10**，tag `v1.2.10`，`b1bd2a6`，2026-02-19 | JVM 转置预测和 Spark 4.0/4.1 支持；相邻 1.2.9 才是近期 Python/数据接口变化的主要来源 | 单看最后一个补丁会漏掉数据与推理能力；必须比较完整安装版本 |
| LightGBM | **4.7.0**，tag `v4.7.0`，`8f7036f`，GitHub 页面显示 Jul 18 20:20 | Polars/Arrow 互操作、ROCm/HIP、NCCL 多 GPU、CUDA 13 构建及分布式修复；迁移到 `lightgbm-org`，默认分支 `main` | 不再使用“LightGBM 没有 CUDA/多 GPU”作为差异；CPU/OpenCL/CUDA/HIP 后端须分开记录 |

来源：[XGBoost 3.4.1](https://github.com/dmlc/xgboost/releases/tag/v3.4.1)、
[CatBoost 1.2.10](https://github.com/catboost/catboost/releases/tag/v1.2.10)、
[LightGBM 4.7.0](https://github.com/lightgbm-org/LightGBM/releases/tag/v4.7.0)。
LightGBM 行保留页面显示的月日，未以不可用的 API 时间戳推断精确 UTC 发布时间。

### XGBoost 3.4 的结构性变化

3.4.0 的 `hist` vector-leaf 实现被上游称为 feature-complete，仍标为 experimental；
涉及类别、约束、DART、分布式、模型检查和批量统计等支持。MAE/quantile 叶估计改为
平滑近似。默认二进制改用 CUDA 13.3，另有 CUDA 12.9 的 `xgboost-cu12` 变体。
[3.4 系列正式说明](https://xgboost.readthedocs.io/en/stable/changes/v3.4.0.html)。

设计推论：多输出/vector leaf 本身不能作为独占卖点；v1 要区分分裂统计与叶拟合
统计，允许改变叶求解算法，比较时也不能把旧版 quantile 的数学当作最新版参考。
T4 旧环境与最新对手 wheel 能否共存必须预检，不能静默降级对手版本或回 CPU。

### CatBoost 的已有能力不能低估

1.2.9 加入 Polars 数据输入（含辅助字段）、RMSPE、mmap 模型加载、Python 3.14
适配，优化 Lossguide 与数据初始化；这些是该版正式记录，速度自述未在本项目复测。
[1.2.9 release](https://github.com/catboost/catboost/releases/tag/v1.2.9)。

GPU custom objective / metric 已出现在 1.2.6 release 中。不能把“Python 自定义
GPU loss”称为 OpenBoost 独有；支持范围和调用约定仍须针对实际任务验证。
[1.2.6 release](https://github.com/catboost/catboost/releases/tag/v1.2.6)。

当前官方目标列表还包括 Poisson、Tweedie、MultiQuantile、RMSEWithUncertainty、
Cox 与 SurvivalAft，并逐项区分 GPU 支持；例如 Cox/SurvivalAft 列为 CPU。
CatBoost 的 AFT 上界哨兵与 XGBoost 不同，adapter 必须转换语义而非原样传标签。
[回归目标与设备支持](https://catboost.ai/docs/en/concepts/loss-functions-regression)。
有序 boosting / 类别统计也是必须纳入能力审阅的既有技术；v1 不以完整复刻为目标。
[官方参考资料](https://catboost.ai/docs/en/concepts/educational-materials-papers)。

### LightGBM 的成熟任务面与工程工作

其公开参数覆盖分类、多分类、ranking、Poisson/Gamma/Tweedie、quantile、约束与
custom objective；逐轮更新和叶输出修改有公开入口。
[参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)、
[Booster 接口](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)。
v1 对照需要比较相应 objective 与完整预处理/推理，而非只有一个回归 fit timing。
4.7 的加权百分位修复提示：quantile 等叶求解必须有非单位权重独立参考。

## 2. 公开 plans：按证据等级理解

没有把所有库描述成拥有统一且承诺交付日期的 roadmap。
下面分别标注正式 tracker、作者明确工作意图、待讨论 proposal 和社区请求。
这些状态是本日读取页面的状态，未来排期不做保证。

| 项目 / 来源 | 核对到的状态和内容 | 对 v1 的实际用途 |
|---|---|---|
| XGBoost [multi-output roadmap #9043](https://github.com/dmlc/xgboost/issues/9043) | Open、`type: roadmap`；正文已更新 3.4 `hist` 完整性；仍讨论多任务、更广输出和执行接口 | 已完成项用 release 确认；不能把页面旧 checklist 全当作缺失能力。模型轴、输出轴与统计轴须独立 |
| XGBoost [默认参数 RFC #12131](https://github.com/dmlc/xgboost/issues/12131) | Open；2026-03-26 提议调整学习率、采样和训练预算 | proposal 不等于当前默认；eval 同时记录有效默认和公平调参配置 |
| LightGBM [#2302](https://github.com/lightgbm-org/LightGBM/issues/2302) | 官方仓库集中 feature request / voting hub，无固定交付承诺；链接有不同状态 | 需求池可指示数据加载、routing、ranking、内存等痛点；投票数不是 adoption 或路线承诺 |
| LightGBM [预测效率 #7326](https://github.com/lightgbm-org/LightGBM/issues/7326) | Open、作者明确计划研究 Python predict/import 成本，作者已指派 | v1 记录进程启动、导入、模型加载、单条及批量预测，而非仅 warm fit |
| LightGBM [类别编码 #7361](https://github.com/lightgbm-org/LightGBM/issues/7361) | Open proposal；希望数据容器无关的类别映射与跨语言模型状态，暂无里程碑 | v1 的类别语义、mapping persistence 与 unseen policy 不能依赖 pandas 实现细节 |
| CatBoost [导出模型 CI #3173](https://github.com/catboost/catboost/issues/3173) | Open Task，2026-08-23；增强代码导出模型测试；页面关联 #3174 | 属于可见工程任务，不是完整产品 roadmap；v1 必须独立环境验证推理 artifact |
| CatBoost [C++ 导出兼容 #3172](https://github.com/catboost/catboost/issues/3172) | Open，2026-08-23；类别模型导出代码的语言标准问题，无发布日期承诺 | 说明 deployment 是实际用法；OpenBoost 自己仍不承担旧 API/格式兼容义务 |

本次未找到 CatBoost 最新的统一、带时间承诺的完整 roadmap；已查看 releases、
当前任务、contribute 与 milestones，不能把长期 `planned` / good-first-issue 标签
自动升级为即将发布。LightGBM 使用公开请求中心与具体工作条目，也不假设请求都获排期。
GitHub API 与部分带筛选条件页面本次抓取失败；结论基于成功读取的 release 与具体
issue 页面。检查记录不等于完整跟踪所有评论、PR、未公开计划。

## 3. 汇入 v1 的决定

1. **目标是可验证的算法修改。** 基础分类/回归、概率建模、GPU custom loss、vector
   leaf 都已有替代方案，直接纳入对照而非排除。
2. **算法覆盖与应用覆盖分开。** 保险和 AFT 是实例；分类、ranking、quantile、
   multi-output、结构化与批量模型选择都要进入覆盖矩阵。
3. **至少五种修改任务。** Objective/geometry、split/growth、leaf solver、update
   control、run scheduling，包含一个现有库擅长的 control 和未见任务。
4. **实际工程成本进入 eval。** 类别/缺失、权重、offset、分组、持久化、冷启动与
   推理，均有独立判错条件；不把库的 GPU 支持等同于每个 objective 支持。
5. **对手和协议版本冻结。** 先做 capability smoke，记录 package 名、版本/hash、
   backend/driver、实际模型参数和推理语义；以后更新 release 要另立实验版本。

本文件支持设计和 baseline 选择，不支持任何 OpenBoost 速度、质量或采用优势结论。
