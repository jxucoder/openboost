# Sprint 001：独立 scalar/tree 参考

开始日期：2026-09-05。起点：`9700845`。状态：**完成本 sprint；F0.2 仍进行中**。
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
- [x] 先写手算测试并确认在参考模块不存在时失败。
- [x] 编写 `tests/v1/reference/` 的独立 scalar 与 brute-force tree 函数。
- [x] 验证两轮 trace、三种 grow、missing/weight/cohort/tie/非法输入及 import 隔离。
- [x] 运行集中测试和 changed-file lint；审阅变更并提交。
- [x] 写 reflection、结果和下一 sprint 交接，更新目录状态。

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

交付：[scalar](../tests/v1/reference/scalar.py)、[tree](../tests/v1/reference/tree.py)、
[手算与反例测试](../tests/v1/test_tree_reference.py)、
[隔离验证](../tests/v1/test_reference_independence.py)、[运行说明](../tests/v1/reference/README.md)。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1 --confcutdir=tests/v1 -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check tests/v1
```

- Red：仅有测试时，两处 collection error，原因是 `tests.v1.reference` 尚不存在。
- Green：**55 passed**，无 skipped；changed-file Ruff 全通过。
- 环境：macOS、本地 CPU，Python 3.12.12、NumPy 2.3.5、pytest 9.0.2。
- 子进程禁止所有 `openboost` imports，三个 policy 仍能运行两轮并得到手算结果。
- 第一轮 gain=16/3，第二轮叶=±56/45、最终 raw=±58/225；D2 cut 从1变2。
- 只证明这些 reference fixture 的行为，不是新 production parity 或运行成本证据。

## Reflection

开工观察：上一阶段已把架构写具体；当前缺少能与未来组件独立比较的 executable oracle。
证据：`tests/v1/` 不存在；旧测试使用旧 gain/权重约定，且 root conftest 导入 production。
决定：先完成有限的 scalar/tree 参考，不修改生产 API 或提前做 GPU 优化。
下一步：用手算 fixture 驱动实现，收尾时核对覆盖与剩余 F0.2。

收尾观察：三个 grow 可以用同一候选/真实路由数学描述，但 symmetric 必须在选择前
合并共同候选；节点各选赢家不能表达该算法。证据：左右节点分别偏好 feature1/2，
独立 symmetric 参考选择两者共同的 feature2，保留左节点零 gain 的合法候选。
决定：未来公共 candidate 层保留合法性与收益的区别，生长策略决定何时筛掉非正收益。

覆盖反思：完成的是 numeric scalar 子集；类别/分箱、分类、ranking、quantile/vector、
正目标/AFT、Normal/Formula、transaction/run 仍未闭合。下一轮继续 F0.2，不能因55个
内部测试通过就称 foundation 完成。用户在执行中明确旧生产代码整体退役、v1重新构建；
该清理单独记录/提交，保留已验证 reference 和历史实验，不改变数学/质量门槛。

## Commits

- `9700845`：前置 foundation 构建设计。
- `e76a2cd`：sprint 执行与反思机制。
- 实现切片：`test: add independent scalar and tree references for v1`。
