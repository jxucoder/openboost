# Sprint 009：混合特征与多层向量树

起点：`a6b3eeb`。状态：完成本 sprint；F0.2进行中。B01/F0.2，A2/A6/C1–C3 的参考集成缺口。

## 计划与验收

1. 训练拟合numeric/category转换器；树保留转换器，raw验证预测复用cuts/字典。
2. 独立逐行枚举多层scalar/vector树，支持depthwise/best-first/symmetric；类别按
   one-vs-rest相等判断，missing两方向，分裂可投影而叶仍使用完整输出统计。
3. 两轮binary混合输入与两轮多输出；K=1与已有scalar oracle比较，多层routing行守恒，
   unknown/missing、输出置换、权重复制、无合法split、冻结snapshot检验。
4. 完整回归/lint、验收映射和reflection；offset/两阶段模型留给下一有限集成切片。

不新增生产API或GPU路径。参考输出使用[N,K]，内部布局不是生产约束。已有scalar
oracle继续作为结构不同的对照，不将其替换成新实现以获得自洽测试。

## 结果与验证

本 sprint 有限范围完成。新增 `mixed.py`、24项测试及生产import隔离检查。
**279 passed，无 skipped**；训练转换、scalar/vector完整树和raw预测在同一fixture中连通。

- Transformer只由训练X拟合，保留names/kinds/cuts/类别字典；predict再次使用保存的
  转换器，unknown保持missing路由。identity同时绑定原始内容、row IDs与拟合转换状态。
- grow逐候选重新路由原始行并逐行归约，depthwise/best-first/symmetric均支持多层。
  类别是相等判断、numeric是阈值；没有histogram或生产backend依赖。没有leaf-budget
  配置，本参考按max_depth终止；不以参数缺省冒充所有生长配置均已覆盖。
- 多输出候选用所有split通道收益和；可显式投影gP/diag(PᵀHP)，叶仍求完整K维。
  三policy两轮四叶树逐叶对应一行，预测为y/2、.95*y/2，raw2=.0975*y。
  K=1在numeric/missing上与旧scalar oracle比较；输出置换、权重复制通过。
- 混合numeric/category三级树产生6个单行叶，lambda=0的声明fixture恢复精确向量；
  symmetric独立检查每层共享condition。原scalar/stump参考保留，未重写对照以求一致。
- binary mixed两轮梯度/树/raw预测串联，验证大于训练范围的numeric值、未见类别、
  missing；训练cuts/字典不因验证数据变化。中间类别m被单独隔离，unknown走同missing分支。
- 两种missing方向、zero-weight行、全缺失/零曲率、非法schema/projection/depth、
  输入修改不改变树、行守恒及无重复路由均有检查。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

本地macOS CPU、Python3.12.12、NumPy2.3.5。实现前missing-module失败；初批8/11通过，
三个失败来自下述错误预期，保留反例后采用正收益多层fixture。无CUDA、真实质量、
生产API或持久化结果。Transformer是不可变参考记录，不是生产布局或序列化格式。

## Reflection：向量分裂的正则代价

观察 → 最初y第一轴±1、第二轴±3的fixture预期第二层继续分裂，实际三policy都停在一层。
证据 → 第二层分裂的收益为第一轴+.5，第二轴因为重复正则化常数叶产生-1.5，总收益-1。
这符合按输出收益求和的契约，不是grow故障。决定 → 保留“不应继续分裂”测试，另把
第一轴设为±2使第二层收益为正，再验证完整两轮；未降低算法门槛。

观察 → 现在类别/向量参考从训练转换走到raw预测，并保留独立scalar/stump对照。
决定 → 对应F0.2缺口可记为已覆盖；仍不代表C1/C2/C3生产组件通过。
下一步 → 按验收清单完成offset/两阶段模型及best/per-run RNG有限状态组合，核对所有
未决F0.2项，再转F0.3。避免把已完成数学重做成更多孤立fixture，全部A1–A13仍required。

## Commits

- 本切片：`test: add mixed feature and full vector growth references for v1`。

提交前review补充：单个node score有限不保证左右gain之和有限。新增溢出反例，
对candidate/layer总收益显式检查，避免inf成为最优候选；最终计24项新增、279项通过。
