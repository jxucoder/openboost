# Sprint 008：D1/D3/D4 精确作者任务参考

起点：`6ae1960`。状态：完成本 sprint；F0.2进行中。B01/F0.2 的作者开发题参考；不计作E5作者评测。

## 计划与验收

1. D1：tau=.8 expectile、weighted base、正/负/零残差、tau=.5退化、两轮树更新。
2. D3：最小化 sum(w*pinball)+lambda*(v-anchor)^2/2；枚举残差断点和区间驻点，
   对照独立subgradient/最优性，检查lambda/anchor与两轮routed叶。权重和不归一化。
3. D4：Normal有序参数；每步alpha=.1*.5^j，j=0..5，有限且严格下降才提交；
   反号/NaN六次全拒绝、第二参数读已接受或保持的状态。复用已有不可变Update参考。
4. 完整回归/lint，更新验收映射、reflection并提交。

边界：public API作者扩展、真实作者工时、持久化和完整runtime/best/RNG集成仍待后续；
此处只交付独立判错依据，不把写出开发题解答当作E5通过。

## 结果与验证

本 sprint 有限范围完成，新增 `author.py`、26项测试及生产import隔离检查。
**255 passed，无 skipped**；Ruff通过。环境：本地macOS CPU、Python3.12.12、NumPy2.3.5。

- D1：expectile gradient/hessian按残差符号定义；r=0明确选h=2*tau，不能称该点有
  唯一经典二阶导。tau=.5回到half-square；tau=.8的[0,2] base=1.6，weights=[3,1]
  时base=8/7。独立枚举相邻目标间固定权重的驻点；正/负残差finite difference、
  zero-weight outlier、复制行、常数目标、两轮预测与手算叶通过。
- D3：sum(w*pinball)不除总权重；枚举全部残差断点及每个开区间的驻点（含两端）。
  [0,2,10]/[1,3,1]/q=.5/anchor0，在lambda=1时v=1.5，lambda=10时v=.15，
  lambda=.1时v=2。独立subgradient包含0，九组q/anchor网格比较全局最小值。
  权重复制等价；仅放大weight会改变解，同时放大lambda则保持解。
- D3 routed tree：只替换终端叶，重新读取原始residual/weight；高值叶两轮为8/7.2，
  低值叶为-2/-1.8，raw2=[1.62,1.62,3.52]。与默认quantile的数学差异单独测。
- D4：两参数按0→1有序更新，每参数最多六次(.1,.05,.025,.0125,.00625,.003125)。
  方向反号与NaN候选六次全拒绝，无terms/raw/loss残留；第一次成功后第二参数读取
  accepted raw，第一次拒绝则读取原raw；两轮均检查。输入修改不能影响旧snapshot，
  deterministic fixture不消费全局RNG。完整best-state/per-run RNG协议仍属后续集成。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

初始测试在模块缺失时失败；实现后初批15项通过，补足边界/权重语义后最终通过。
没有运行public API作者扩展、对手成本实验、真实任务、CUDA或持久化；这些测试不是E5。

## Reflection：三个实现提交后的收束

观察 → D1/D3/D4已经有各自可判错的参考，不应再用笼统“支持自定义目标/叶/更新”描述。
证据 → expectile改变目标几何；penalized quantile的最优值可在断点之间；ordered规则
必须看接受后的状态。这些修改轴复用同一独立树参考，但不共享错误的数学假设。
决定 → F1应让作者分别替换objective、leaf solver与update policy，F2再计真实修改成本。

观察 → 最近三个sprint填补了方向、run和开发题数学，仍没有production。
证据 → 当前仅tests/v1/reference新增实现；F0.3数据/预算/judge尚未冻结。
决定 → 按验收清单收束，不扩大开发题或增加无对应门槛的参考。
下一步 → 类别/多层vector完整grow及raw-transform、offset/两阶段模型的有限组合链路，
再核对F0.2出口并进入F0.3。F1–F5与全部A1–A13保持required，E-gates不提前标绿。

## Commits

- 本切片：`test: add exact author mutation references for v1`。
