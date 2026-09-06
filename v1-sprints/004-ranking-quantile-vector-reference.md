# Sprint 004：ranking、quantile 与向量叶参考

起点：`e3d99ad`。状态：完成本 sprint；F0.2 进行中。范围 B01/F0.2，A4/A5/A6 的独立数学与两轮探针。

## 计划与验收

1. 写手算与失败反例：query 内 pairs/归一化、加权分位点左端、共享 split/vector leaf。
2. 实现 NumPy 独立参考；用有限差分、query shift、残差第二轮、K=1 退化、输出置换验证。
3. 回归、lint、记录 reflection 并提交；保留全部 F0.2 和 v1 未完成范围。

- Ranking 枚举全部 query 内严格 relevance pairs，pair loss 按每 query 的 pair 数平均，
  再乘 query weight；pair weight 乘分子，不改变分母。拒绝通用 row weight。
  Lambda 权重由当前排序冻结，ties 用稳定 row ID；IDCG=0 的 NDCG=1。
- Quantile 使用 routed residual/weight 的左端加权分位点，不使用 Newton leaf；
  pseudo h=1 只用于 topology。手算、非光滑最优性与两轮更新都须通过。
- Vector 用共享 stump 最小探针验证候选收益跨输出求和、完整向量叶、可选线性 split
  projection。另对照独立 scalar trees；标准化仅拟合训练集，常数目标 scale=1。
- 本切片不把 stump 探针宣称完整 vector grow，也不宣称 ranking sampling、真实质量、
  持久化、CUDA 或生产 foundation 已完成；E-gates 仍未通过。

## 结果与验证

**本 sprint 的有限范围完成；F0.2 仍进行中。** 新增三个独立参考模块和27项测试，
扩展阻止所有生产 import 的子进程检查。默认 v1 回归共 **122 passed，无 skipped**。

- Ranking：单 pair 的 query weight=3 给出 g=(-1.5,1.5)、h=(.75,.75)；
  三个 pairs 的 query 仍平均为 log(2)。普通 pair 与冻结 lambda surrogate 的有限差分
  均通过；query 常数平移不变、梯度守恒、row-ID ties/置换、零 IDCG、极端 scores、
  无 pair/零权重 query、未知 weight keys 与 row-weight 拒绝均有检查。
  两轮树首轮叶±.4、第二轮由 sigmoid(-.08) 重新计算；lambda 因新排序改变的反例
  单独保留。无采样、没有 NDCG 普通梯度的声明。
- Quantile：q=.1/.5/.9 的手算、左端 tie、正权重过滤、复制行与 pinball 最优性检查。
  pseudo h 始终明确为1，不称精确 Hessian；树只借 scalar reference 选择 topology，
  然后用每个叶的 routed residual/weight 重新求值。两轮高值叶为8和7.2，raw2=3.52。
- Vector：共享 stump 对所有输出 candidate gain 求和，完整 K 维叶独立求解；
  分裂可用投影 gP 和 diag(PᵀHP)，每列作为显式 sketch 通道求和，不将其当完整
  投影 Hessian。单位投影为默认。投影仅选择 topology，不压缩 leaf payload。
  fixture 共享树选 feature1，投影到首输出选 feature0；独立树分别选两个特征。
  K=1 与 scalar depth1 一致，输出置换、权重复制、无分裂 root 与两轮手算通过。
- TargetScale：训练集逐列无权重 population mean/std（ddof=0），常数列scale=1；
  保存 tuple 状态，训练输入改变不影响验证转换/逆转换。不是完整 output artifact。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

Ruff 通过。环境为本地 macOS CPU、Python3.12.12、NumPy2.3.5、pytest9.0.2。
初始测试在缺少 ranking 模块时收集失败。实现后11/12通过，剩余是下述样例假设错误。
修正验收样例、保留原失败情形，并补边界检查后最终全部通过。
CUDA、真实数据、外部对手、序列化、全深度 vector grow 和正式 E-gates 未验证。

## Reflection：反例触发与 sprint 收尾

观察 → 最初 quantile fixture 使用 y=[0,2,10]、w=[1,3,1]、base=2，预期高值叶为8，
实际没有 split、叶更新为0。证据 → equality 取1[y<F]-q的约定，使两子节点 pseudo
梯度同号，默认正则下无正收益。weighted residual quantile 正确并不保证 pseudo
分裂目标改善。决定 → 保留此“不分裂”反例，另用 w=[2,1,2] 验证有正收益下两轮
残差叶。没有改变 split 门槛或伪造树来满足预期；真实 A5 质量必须独立评估。

观察 → 三个用例需要不同的数据入口，却复用了已有行路由和 scalar 数学参考。
证据 → ranking 从 pair 归约到行；quantile 的叶再次读取 residual；vector 的 split
统计和 leaf 统计维度不同。决定 → 保持 construction design 的 objective→stats→
topology/leaf 边界。共享 stump 只是最小 probe，完整 grow/输出 artifact 不标记完成。
下一步 → 正目标/count 与 survival/AFT 参考，然后 Normal/Formula、typed identity、
state/run 与剩余完整 grow 检验；F0.3冻结评测后才能进入 F1。所有 A1–A13 保留。

## Commits

- 本切片：`test: add ranking quantile and vector leaf references for v1`。
