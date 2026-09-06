# F0.2 独立参考验收映射

截至 Sprint 007。只评价参考证据；**F0.2 未关闭，F0.3未执行，F1生产未开始**。
来源：[任务契约](../planning/foundation-tasks.md)、[主计划](../planning/agent-boosting-foundation-plan.md)。
不得把此表的“已有”解释为对应A/R/C或E-gate已通过。

## 用例参考证据

| 范围 | 已有独立证据 | 尚待补齐或衔接 |
|---|---|---|
| A1/R1 | Sprint001 scalar数学、三种grow、两轮预测 | 公共组件conformance与外部差异解释在F1/F0.3 |
| A2/A3/R1 | Sprint003映射、missing、binary/softmax、两轮与换类顺序 | native categorical完整grow与raw-transform→模型预测的组合参考 |
| A4/R3 | Sprint004 pair/query归一化、lambda权重、两轮、NDCG | 后续真实query评测/采样估计协议；不是F0.2速度任务 |
| A5/R2 | Sprint004三q的加权quantile、routed叶、两轮 | D3惩罚叶未完成；完整三q预测组合/持久化后续 |
| A6/R8 | Sprint004共享stump、投影/完整叶、K=1、两轮 | 完整多层vector grow参考；独立与共享结构均须保留 |
| A7/A8/R4 | Sprint005 Poisson/Gamma、exposure、权重、两轮 | offset随接受项/预测组合的参考集成；真实adapter在F0.3 |
| A9/R4 | Sprint005 Tweedie、paid-count join与乘积定义 | 两阶段训练→组合预测的小型参考还未接通 |
| A10/R5 | Sprint005 event/censored、尾概率、单位、两轮 | output/persistence协议后续；真实IPCW由F0.3固定 |
| A11/R6 | Sprint006 Fisher/ordinary、NLL/CRPS、joint/ordered | D4严格6次步长与方向反号/拒绝后下一参数的完整fixture |
| A12/R7 | Sprint006 Jacobian/GGN、三方向、重复Z两轮、错设/不可识别 | 真实质量与公式artifact在F1/F4；不能由合成参数推真实物理结论 |
| A13/R9 | Sprint007 K=1/2隔离、M=1/8/32顺序、best/失败/RNG | 批量执行、真实成本与多recipe整合不由顺序模拟证明 |

## 作者修改与跨用例边界

| 范围 | 状态与下一步 |
|---|---|
| D1 expectile | **缺**：tau=.8、加权base、两轮、tau=.5退化与r=0约定 |
| D2 cohort split | Sprint001已有逐候选可行性/无合法split/独立信息权重 |
| D3 penalized quantile | **缺**：断点+驻点枚举、subgradient、anchor/lambda、路由后的两轮更新 |
| D4 ordered acceptance | 已有两轮顺序/回溯和拒绝；**缺**指定alpha=.1*.5^j、最多6次及反号探针 |
| D5 scheduling | Sprint007顺序/独立/重排/重组、故障重试、停止、ID seed与内容变化 |
| C1 identity/bind | 已有列转换+有类型hash+prepared行序验证；**缺**混合转换与算法的完整组合fixture |
| C2/C3 tree/leaf | scalar完整grow已有；类别、多层vector、D3参考待补 |
| C4 state/run | 不可变候选与best terms已有；**缺**跨offset/映射/两阶段的模型状态集成 |
| C5 artifacts | 序列化round trips属于F1构建要求；不可将内存snapshot当作持久化证据 |
| C6/C7 eval/workflow | F0.3冻结/判卷器、F2作者评估、F5安装工作流均未完成 |

## 收尾顺序

1. 下一 sprint 完成 D1/D3/D4 的精确定义和反例。
2. 然后补类别/多层vector与混合数据、offset/双模型的有限组合参考，复查两轮链路。
3. 按本表核对F0.2出口并记录剩余阶段归属；不把F1要求提前标记通过。
4. 开始F0.3的真实数据manifest、capability smoke、预算/保留任务/judge冻结。

R1–R9/C1–C7/A1–A13始终全部required。GPU列的optional不能被误读为CPU用例optional。
