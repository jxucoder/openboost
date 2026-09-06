# Sprint 001：独立 scalar/tree 参考

开始日期：2026-09-05。起点：`9700845`。状态：**进行中**。
计划映射：B01 / F0.2 的第一部分；A1/R1、C2/C3 的 scalar 子集、D2；E1 的参考准备。

## 目的与范围

为 F1 的 histogram/split/route/leaf/grow 建立独立判卷依据，避免新实现和 oracle
共同调用旧 production 算法而重复同一个错误。测试只在微型 fixture 上穷举、逐行归约。

- scalar 加权半平方损失、base、Newton 叶与半 gain；权重只乘一次。
- 固定 numeric bins 的所有候选、两种 missing route、ties、无合法候选和真实子节点。
- D2 的独立 cohort information mass；不能只检查总 H。
- depthwise、best-first、symmetric 的明确 topology 与预算；两轮平方误差更新 trace。
- 无 production import 的隔离执行。root `tests/conftest.py` 会导入旧 openboost，
  因此新参考的独立测试使用 `--confcutdir=tests/v1`，另用干净子进程检验 import 独立性。

不在本 sprint 声称完成：类别/向量/其他目标参考、生产 trainer、持久化格式、CUDA、
真实数据/基线评测，或整个 F0.2。其余用例继续按主计划实现，不降级为 optional。

## 执行清单

- [x] 读取设计、旧 split/插件语义及 tests；识别旧 gain 是两倍、旧测试入口导入 production。
- [ ] 先写手算测试并确认在参考模块不存在时失败。
- [ ] 编写 `tests/v1/reference/` 的独立 scalar 与 brute-force tree 函数。
- [ ] 验证两轮 trace、三种 grow、missing/weight/cohort/tie/非法输入及 import 隔离。
- [ ] 运行集中测试和 changed-file lint；审阅变更并提交。
- [ ] 写 reflection、结果和下一 sprint 交接，更新目录状态。

## 验收

1. `[-2,-2,2,2]`、两个 feature bins、lambda=1 的首轮叶为 ±4/3、净 gain=16/3。
   eta=.1 后第二轮残差必须改变，不能复用首轮梯度；具体两轮数字由手算期望检查。
2. 整数权重等价于复制行（固定 bins）；零权重不能制造可分节点；非法分母与负/非有限值拒绝。
3. 缺失左右路由均有最优反例；精确 ties 按 feature/candidate/missing-direction 排序。
4. D2 六行例中无约束选 cut 1，cohort 约束后选 cut 2；无可行项返回无 split。
5. 三种生长各有区分行为的 fixture；symmetric 先合并同一候选的层收益，不能拼节点赢家。
6. reference 只用标准库/NumPy，不调用 OpenBoost 的 objective、split、histogram 或 trainer。
   参考自身不计 E1 production parity；独立判卷仍需未来被测组件。

## 执行与验证记录

尚未完成；按实际结果补充，不预填 pass。

## Reflection

开工观察：上一阶段已把架构写具体；当前缺少能与未来组件独立比较的 executable oracle。
证据：`tests/v1/` 不存在；旧测试使用旧 gain/权重约定，且 root conftest 导入 production。
决定：先完成有限的 scalar/tree 参考，不修改生产 API 或提前做 GPU 优化。
下一步：用手算 fixture 驱动实现，收尾时核对覆盖与剩余 F0.2。

## Commits

- `9700845`：前置 foundation 构建设计。
