# Sprint 005：count、正目标与 log-normal AFT

起点：`ddf1143`。状态：完成本 sprint；F0.2进行中。范围 B01/F0.2、A7–A10 的数学与最小数据探针。

## 计划与验收

1. 先写失败测试：Poisson exposure、Gamma/Tweedie 导数、事件/右删失与稳定尾概率。
2. 独立 NumPy/stdlib 参考实现；手算、有限差分、权重复制、两轮树更新和输出单位验证。
3. 补 A9 小型保单/正赔款关联与两阶段乘积探针；回归、lint、reflection、提交。

- e 在 Poisson 中是 offset；annualized Tweedie 中是 weight，不能重复施加。
- Gamma y>0；Tweedie y>=0 且1<p<2；溢出/非法输入明确失败，不静默截断。
- AFT 接受有限正时间事件或正 lower/+inf upper 右删失；其他区间显式拒绝。
  固定 sigma>0；稳定 log-tail/Mills；median/mean/survival/quantile 单位区分。
- 每类 objective 通过两轮新 raw 的梯度、叶值及预测检查；输出保存/真实对手评测仍待后续。
- 数据探针列出不一致/孤立/非正赔款，不将缺失赔款自动填0；正赔付次数与均值配对。
- 不宣称产品、真实保险/生存效果或 E-gates 已完成；所有其他用例继续保留。

## 结果与 eval

本 sprint 有限范围完成。新增 `positive.py`、`survival.py` 与61项测试，扩展
生产 import 隔离检查。当前 **183 passed，无 skipped**，Ruff通过。

- Poisson：rate base=log(7/5)手算；全零有效计数的 minimum_rate 必须显式指定。
  e 翻倍仅使 count mean 翻倍；rate不变，weight不混入 exposure。Gamma base为
  log(weighted mean)，Gamma/Tweedie 手算 loss/g/h与有限差分一致，均返回未加权几何。
- 三种正目标的两轮 root leaf 用独立代数公式核对，整数权重与复制行 loss 一致；
  非法support/power/e/weight拒绝。Gamma在 y=1e308、F=log(y) 时通过 log-ratio
  保留有效几何；不可表示指数明确失败，无隐藏截断或最小curvature。
- AFT：event与right-censored分支均通过有限差分；lower必须有限正数，upper只能
  等于lower或为+inf，其他区间明确拒绝。混合事件/删失的两轮更新由erfc公式独立核对。
- Normal右尾：z<=8用erfc/log1p；z>8用300层连分式，并直接保留 Mills-z correction
  计算曲率。z=-10..30 与erfc核对，z=40/100与64点Gauss-Laguerre独立积分核对；
  切换点连续性通过。没有把右尾SF减到0后再取log，也不通过裁剪假造曲率。
- 输出：median/mean/quantile/survival分开检查；生存随时间递减。
  时间单位放大7倍时，event NLL增加log(7)（密度Jacobian），censored NLL不变，
  两者g/h均不变。输出序列化仍待后续，不以数学反变换代替持久化验收。
- A9 join：按policy ID汇总正赔款，保留孤立/非正/计数矛盾排除原因；只对零计数且
  无赔款保单填0。paid-count/exposure×paid-mean等于annualized amount；原ClaimNb
  不替换paid-count，实体顺序改变结果不变。这里只是小表数学，不是实数据ETL或模型。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

环境：本地macOS CPU、Python3.12.12、NumPy2.3.5、pytest9.0.2；无新依赖。
实现前测试因缺模块失败；实现后数学初批21项全部通过。扩展测试曾把标量loss/数组g/h
作为一个不规则数组比较而失败，改为逐项比较；lint变量命名问题已修正。
未运行真实dataset/对手、IPCW、CUDA、完整两阶段训练/保存、正式E-gates。

## Reflection：三个实现提交后的方向复核

观察 → Sprint003–005完成了不同目标几何的准备，但当前生产包仍只有namespace。
证据 → 所有新实现位于independent reference，隔离测试禁止生产import；没有性能、
真实质量或Agent修改成本数据。决定 → 不把测试数增加当作foundation产品价值；
这些反例须在F1成为public components的conformance测试，F2验证Agent修改成本。

观察 → 相似的exposure字段在A7/A9具有不同角色，删失/事件的相同时间也具有不同似然。
证据 → e翻倍、paid-count乘积与time-Jacobian测试明确区分这些情况。
决定 → 保持typed target/offset/weight/output schema边界，拒绝通用trainer隐式猜语义。
下一步 → Normal/NaturalBoost与Formula参考；然后补行身份、state/run、完整类别/
向量grow等F0.2缺口和F0.3冻结。所有A1–A13仍需各自真实实现与评测，不因本次选择
保险与AFT样例而给予它们范围特权。

## Commits

- 本切片：`test: add positive target and AFT references for v1`。
