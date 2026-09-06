# OpenBoost v1 执行与反思

这里记录用户要求的 v1 sprint 计划、执行结果和反思。目标始终是：**帮助研究者与
Agent 用公开、可组合组件完成正确的算法修改，并验证其成本和实际价值。**

设计依据：[主计划](../planning/agent-boosting-foundation-plan.md)、
[构建设计](../planning/foundation-construction-design.md)、[任务契约](../planning/foundation-tasks.md)、
[验收](../planning/openboost-v1-evaluation.md)。这些文件定义 scope/架构/门槛；本目录管理执行，
不复制一套不同的路线图。R1–R9/C1–C7/A1–A13 都必需，不因某个 sprint 成功而缩减。

## 当前执行位置

| Sprint | 对应计划 | 状态 | 交付与记录 |
|---|---|---|---|
| 001 | B01 / F0.2 的 scalar/tree 子集 | 完成，55 tests passed | [独立 scalar/tree 参考](001-scalar-tree-reference.md) |
| 002 | 用户要求提前退役旧生产代码 | 完成，namespace/build/docs 检查通过 | [从干净实现起点构建 v1](002-retire-legacy-production.md) |
| 003 | B01 / F0.2 的列转换与分类子集 | 完成；验收见记录 | [训练转换与分类参考](003-data-classification-reference.md) |
| 004 | B01 / F0.2 的 A4–A6 数学探针 | 完成；总计122 tests passed | [ranking/quantile/vector](004-ranking-quantile-vector-reference.md) |
| 005 | B01 / F0.2 的 A7–A10 数学探针 | 完成；总计183 tests passed | [正目标与 AFT](005-positive-aft-reference.md) |
| 006 | B01 / F0.2 的 A11/A12/D4 探针 | 完成；总计208 tests passed | [Normal/Formula](006-normal-formula-reference.md) |
| 007 | B01 / F0.2 的 identity/A13/D5 探针 | 完成；总计229 tests passed | [identity/runs](007-identity-runs-reference.md) |
| 008 | B01 / F0.2 的 D1/D3/D4 精确探针 | 完成；总计255 tests passed | [作者修改参考](008-author-mutation-reference.md) |

接下来完成 F0.2 的其他参考（类别和向量完整生长、组合模型与状态集成），再完成 B02/F0.3 的冻结评测。F1 公共 foundation 实现从 B03 开始，
按 B04–B10 接通可组合组件与全部算法；不把 reference 代码称为产品实现。
具体后续 sprint 在开始时按依赖选定有限范围，不能跳过现有 F 阶段出口。

## 执行规则

1. 每个 sprint 开始先写目的、映射到 F/B/A/C/E 的范围、独立失败样例、交付和验收。
2. 每个独立验证的切片提交；记录命令、结果、未验证范围与 commit，不只报告文件数量。
3. **每个 sprint 收尾、每累计三个实现提交、进入新 F 阶段，或出现架构/正确性反例时反思。**
   反思写在当前 sprint 中，回答下面的问题；一次反思可覆盖同时触发的条件。
4. 如果需要改变架构或执行顺序，先记证据与原因并同步设计；不悄悄放宽验收、丢弃失败、
   删除用例或转向多 GPU/功能目录。常规小修不要求重开总体规划。
5. `learnings/` 保留跨 sprint 的耐久结论并链接本目录；执行明细以本目录为准。

## Reflection 检查

- 当前代码减少了哪种正确算法修改的困难？若只是准备工作，具体为哪个组件提供依据？
- 是否沿 construction design 的依赖推进？是否偷渡旧 API 限制、专用 trainer 或性能优化？
- 数学/状态是否有独立证据？哪些结果只是内部模拟，不能支持性能、质量、采用声明？
- 结构不同的用例能否复用边界？是否开始只为当前一个例子设计？
- 全部 required 范围还缺什么？下一步最小可验证交付是什么？

记录格式为“观察 → 证据 → 决定 → 下一步”，不用无证据的“方向正确”结束反思。

## 全局完成状态

F0.1 规格与构建设计已提交。F0.2 进行中；F0.3、F1–F5 均未完成。
Sprint 001 已交付独立 numeric scalar/tree 参考；Sprint 002 已按用户要求退役旧生产代码。
Sprint 003 已补充列转换与分类；Sprint 004 已补充 ranking/quantile/vector 最小参考。
Sprint 005 已补充正目标/count、policy join 与 event/right-censored AFT 参考。
Sprint 006 已补充Normal/Formula；Sprint 007已补充identity/run隔离与模型选择。
Sprint008已补D1/D3/D4精确fixture。下一步补完整grow与有限组合链路；缺口及收尾顺序见
[F0.2验收映射](f0-2-acceptance-ledger.md)。
E0–E6 没有因为创建本目录而通过；E7 独立作者采用尚无新证据。每项 A 的任务卡已定义，
其 v1 production 实现和真实 eval 仍须逐项完成。
