# F0.2 独立参考验收映射

截至 Sprint 010。只评价参考证据；**F0.2 已关闭，F0.3未执行，F1生产未开始**。
来源：[任务契约](../planning/foundation-tasks.md)、[主计划](../planning/agent-boosting-foundation-plan.md)。
不得把此表的“已有”解释为对应A/R/C或E-gate已通过。

## 用例参考证据

| 范围 | 已有独立证据 | 尚待补齐或衔接 |
|---|---|---|
| A1/R1 | Sprint001 scalar数学、三种grow、两轮预测 | 公共组件conformance与外部差异解释在F1/F0.3 |
| A2/A3/R1 | Sprint003分类；Sprint009 native categorical完整grow与raw-transform两轮预测 | 公共组件与真实质量在F1/F4 |
| A4/R3 | Sprint004 pair/query归一化、lambda权重、两轮、NDCG | 后续真实query评测/采样估计协议；不是F0.2速度任务 |
| A5/R2 | Sprint004三q的加权quantile、routed叶、两轮；Sprint008惩罚叶；Sprint010三q模型预测组合 | 持久化与公共组件在F1 |
| A6/R8 | Sprint004 stump；Sprint009三policy多层vector、投影/K=1/两轮 | 公共组件与真实质量在F1/F4；两种结构继续保留 |
| A7/A8/R4 | Sprint005 Poisson/Gamma、exposure、权重、两轮；Sprint010 offset只应用一次、新exposure预测 | 真实adapter在F0.3 |
| A9/R4 | Sprint005 Tweedie、paid-count join；Sprint010 paid-count/Gamma各两轮训练→乘积预测 | 真实质量在F4，持久化在F1 |
| A10/R5 | Sprint005 event/censored、尾概率、单位、两轮 | output/persistence协议后续；真实IPCW由F0.3固定 |
| A11/R6 | Sprint006 Fisher/ordinary、NLL/CRPS、joint/ordered；Sprint008 D4精确规则；Sprint010版本化接受/best恢复 | 公共runtime和真实质量在F1/F4 |
| A12/R7 | Sprint006 Jacobian/GGN、三方向、重复Z两轮、错设/不可识别 | 真实质量与公式artifact在F1/F4；不能由合成参数推真实物理结论 |
| A13/R9 | Sprint007 K=1/2隔离、M=1/8/32顺序、best/失败/RNG | 批量执行、真实成本与多recipe整合不由顺序模拟证明 |

## 作者修改与跨用例边界

| 范围 | 状态与下一步 |
|---|---|
| D1 expectile | Sprint008已有tau=.8、加权base、两轮、tau=.5退化与r=0约定 |
| D2 cohort split | Sprint001已有逐候选可行性/无合法split/独立信息权重 |
| D3 penalized quantile | Sprint008已有断点+驻点枚举、subgradient、anchor/lambda、两轮routed叶 |
| D4 ordered acceptance | Sprint008指定六次alpha、反号/NaN拒绝；Sprint010两轮有序提交、best恢复、逻辑step派生key |
| D5 scheduling | Sprint007顺序/独立/重排/重组、故障重试、停止、ID seed与内容变化 |
| C1 identity/bind | Sprint009接通mixed转换、fitted identity与raw预测；生产typed contracts仍待F1 |
| C2/C3 tree/leaf | scalar、D3、类别与多层vector参考已有；不等于production通过 |
| C4 state/run | Sprint007/009/010覆盖run隔离、映射、offset/两阶段组合、版本化提交与best缓存/terms恢复；公共runtime仍待F1 |
| C5 artifacts | 序列化round trips属于F1构建要求；不可将内存snapshot当作持久化证据 |
| C6/C7 eval/workflow | F0.3冻结/判卷器、F2作者评估、F5安装工作流均未完成 |

## 出口审计与下一阶段

1. D1/D3/D4精确定义与反例已由Sprint008补充；不计作E5作者评测。
2. Sprint009已补类别/多层vector与混合数据；Sprint010补offset/双模型和有限状态组合。
3. F0.2参考准备已完成：288项测试通过，隔离进程禁止导入openboost仍能运行参考；不把F1要求提前标记通过。
4. 开始F0.3的真实数据manifest、capability smoke、预算/保留任务/judge冻结。

R1–R9/C1–C7/A1–A13始终全部required。GPU列的optional不能被误读为CPU用例optional。

可复现命令、组合反例与阶段反思见[Sprint010](010-reference-integration-exit.md)。
数学证据对应 `tests/v1/test_{scalar,tree,data,classification,extended,positive_aft,coupled,runs,author,mixed,integration}_reference.py`；
独立性检查为 `tests/v1/test_reference_independence.py`。所有表内“已有”均限小型CPU参考，
不证明外部库同构、真实任务质量、序列化、作者修改成本或CUDA正确性。
