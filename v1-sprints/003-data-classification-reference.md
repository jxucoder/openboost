# Sprint 003：训练转换与分类的独立参考

起点：`cdce7d8`。状态：完成本 sprint；F0.2 进行中。对应 B01/F0.2，A1–A3/C1 的数学与数据语义子集。

## 计划

1. 先写失败样例：线性分位点、最小值 cut、missing/unknown、类别非连续语义、标签映射。
2. 实现独立列转换参考和 binary/softmax 数学；核对手算、有限差分、权重与两轮更新。
3. 更新参考范围、执行结果和 reflection，运行当前 v1 回归和 lint，独立提交。

## 验收与边界

- 训练拟合 cuts/dictionary；验证转换不改状态；越界数值进入端点 bin，inf 拒绝。
- 数据参考使用不可变列状态，显式 missing；类别按稳定字典 one-vs-rest，unknown 走 missing。
- 标签映射保存概率列含义，未知/缺失标签失败；binary 单类拒绝，初始化裁剪显式配置。
- binary 稳定 loss/g/h；softmax 精确 Hessian 与建树对角上界分开命名，有限差分验证。
- 权重只进入 loss 聚合与树统计，不预乘 derivatives；两轮使用新 raw 的全部输出方向。
- 不声称完成 PreparedData identity、row-ID bind、持久化、类别完整 grow 或所有 F0.2。
  本 sprint 为这些后续边界提供反例；生产 API、真实数据与 CUDA/E-gates 仍未实现。

## 结果与 eval

状态：**完成本 sprint 的有限范围；F0.2 仍进行中**。

- 新增 `tests/v1/reference/data.py`、`classification.py` 和40项测试；扩展隔离 import
  验证，仍禁止引用任何 `openboost` 生产模块。
- 数值：手算 `[0,2,4,6]` 的三 cuts 为1.5/3/4.5；重复值保留最小值 cut；
  常数/全缺失列、单 bin、端点越界、NaN、inf 与极端有限值插值均有反例。
- 类别：训练字典稳定、未知走 missing、两种 missing 路由；中间类别 m 的
  one-vs-rest 收益为64/15，严格优于任何 ordinal threshold，防止偷换类别语义。
- binary：零 raw 的 g=±1/2、h=1/4；非单位权重与复制行等价；±1000 loss 有限；
  正确方向 raw=±100 的非零小 loss 不被相消。初始化 clipping 必须可表示且显式给定。
- softmax：K=3 的 exact Hessian 与2p(1-p)上界分别手算，有限差分验证 g/H；
  shift/class permutation、权重、非法标签、超出 float64 的动态范围有明确检查。
- 两轮：binary 首轮叶±2/3，第二轮由 p=sigmoid(-1/15) 独立算叶；softmax
  加权 root 首轮向量(-3/11,0,3/11)，第二轮由完整 raw1 独立算全部方向。
  另测有分裂的 multiclass 树在换类顺序下两轮等变；训练/验证 raw 一致。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

最终 **95 passed，无 skipped**；Ruff 通过。环境：本地 macOS CPU、Python3.12.12、
NumPy2.3.5、pytest9.0.2。实现前最小测试收集出现两个 missing-module 错误，随后补实现。
没有运行 CUDA、真实数据对手比较、持久化或正式 E-gate；数字不代表95个产品功能。

## Reflection

观察 → 分箱与类别映射决定候选空间，分类输出轴决定几何；这些都不能隐藏在 trainer。
证据 → 最小值 cut、中间类别反例、softmax 同快照第二轮检验分别暴露三种错误实现。
决定 → 保留独立列转换/输出 schema/几何的边界，不为此引入通用 trainer 或生产兼容层。
当前参考只保存不可变列状态，并未完成 PreparedData identity 或 typed Problem；类目路由
也不能被计为完整 categorical tree。下一个 sprint 优先 ranking/quantile/vector 的
不同叶求解需求，并持续追踪剩余行绑定、类别生长、正目标/AFT、Normal/Formula、state/run。
所有 A1–A13 保留；本次只是为以后公开组件 conformance 提供判错依据。

## Commits

- 本切片：`test: add independent data and classification references for v1`。
