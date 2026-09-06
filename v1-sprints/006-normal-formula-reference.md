# Sprint 006：Normal / NaturalBoost 与 Formula 方向和提交

起点：`c5e9899`。状态：完成本 sprint；F0.2进行中。B01/F0.2，A11/A12/D4 的独立数学和候选状态探针。

## 计划与验收

1. Normal 普通gradient/Fisher/natural direction 与独立NLL/CRPS评估；Formula稳定
   softplus、Jacobian、GGN及ordinary/diagonal/full方向：手算和有限差分。
2. 用独立scalar tree拟合负方向，明确fit weight只施加一次；两轮joint和ordered更新。
   固定步长与有限回溯均验证；无下降或非法候选拒绝时原始raw/term不改变。
3. 重复Z不同x、单x不可识别和公式错设反例；完整回归、lint、reflection与提交。

仅交付数学/最小更新参考，不是生产runtime或完整事务系统。Normal/Fisher不得称Hessian，
GGN不意味着可识别性；训练初始化不读取验证数据。持久化、真实任务质量、Agent成本、
CUDA与正式E-gates均待后续，全部A1–A13仍保留。

## 结果与验证

本 sprint 有限范围完成。新增 `coupled.py` 与25项测试，扩展生产import隔离检查。
**208 passed，无 skipped**；Ruff通过。使用本地macOS CPU、Python3.12.12、NumPy2.3.5。

- Normal：手算Fisher与natural direction，训练weighted mean/scale与显式scale下限；
  独立 evaluator 不调用objective，NLL与CRPS闭式值分别检验。两轮natural root叶
  用逐轮mu/sigma的代数式核对，非单位fit weight与复制行等价。
- Formula：softplus/expm1稳定求值，analytic Jacobian与finite difference一致，
  GGN=JᵀJ；ordinary/diagonal/full三个方向均有两轮joint测试。full使用显式2×2逆，
  无damping的秩亏GGN明确拒绝；ordinary不使用metric/damping作预条件。
- 每棵方向回归树的叶与同一raw快照的negative direction独立求和一致；预测由保存
  的tree/channel/coefficient重建。Z与x分开传入，测试含重复Z不同x及已知a/b生成数据。
- Normal两轮ordered更新每次重新算几何，与joint产生不同log-scale；固定有限步长
  可显式允许loss增加，有限回溯则要求严格下降。固定步长也不接受非法几何。
- 回溯依次拒绝非法1000步长、有限但变差的5步长，再接受.1；结果等于直接.1更新。
  全部拒绝时terms为空、raw_before=raw_after，修改调用方raw不会改变记录的snapshot。
- 单一x的两组不同a/b给相同输出，证明full GGN不等于参数可识别；相同配方下递减
  目标与饱和递增公式冲突的错设反例保留。不声称真实Concrete任务恢复参数。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

实现前missing-module测试收集失败；实现后数学初批12项通过。收尾lint修正测试排版，
代码review补充softplus下溢到0的明确拒绝，未改变正参数支持。无新依赖。
完整持久化/best state/early stopping、CPU生产组件、真实NLL/CRPS/Formula质量、
CUDA和正式E-gates均未验证。这里的step是小型参考，不是公共runtime或最终通用trainer。

## Reflection

观察 → 方向、方向拟合与接受规则可以分别表达，无需把Normal/Formula塞入标量Newton接口。
证据 → 两种目标复用已有scalar tree；Fisher/GGN在拟合前独立求解，fit权重只施加一次，
原始loss单独决定候选接受。决定 → 保持construction design的方向适配与事务边界。

观察 → 已有A1–A12的数学探针仍不能称完整F0.2。证据 → A13 run隔离/选择、行身份绑定、
完整类别/向量grow、D3带惩罚叶与完整组合/persistence参考仍有缺口。
决定 → 下一sprint优先state/run与identity参考，并建立剩余验收映射；不提前进入生产
优化，也不继续无限增加相同类型的数学测试。F0.3冻结真实评测后才进入F1公共组件。
所有A1–A13仍需各自production和真实eval；这些准备工作只提供后续可判错的依据。

## Commits

- 本切片：`test: add Normal and Formula directional update references for v1`。
