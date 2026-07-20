# OpenBoost 计划:把诚实清单上的每个场景做到无可争议

> 范围:只做"什么问题应该用 OpenBoost"清单上的四个场景。不做:标准库野心、自调参赌注、
> 小数据 leaderboard、通用单调约束(那是 XGBoost 的地盘)。
> 依据:2026-07-20 四路并行代码审计(30 agents,major+ 差距全部经对抗核实)。
> 重要:REVIEW_FINDINGS.md 已过时 —— CRIT-1/2/4/7/8(部分)、HIGH-2/4/6/7 已修复;
> 下列问题是审计确认**今天仍然存在**的。

> **2026-07-20 修复记录**:Phase 0 全部完成;Phase 1/2/3 中的缺陷类条目已完成(标 [x]),
> 功能类条目(PIT、exposure、注册 API、GA2M 交互、平滑等)仍待做。全量回归 586 passed / 31 skipped。
> 数学核查:NegBin 梯度、链式法则、Tweedie 密度归一化、batch/对称门禁 —— 4/4 独立验证通过。
> 额外发现并修复:StudentT 的 df 梯度也是错的(两项符号错 + link 链式法则用错),已修并有 FD 测试锁定。

## Phase 0 — 正确性止血 ✅ 全部完成(2026-07-20)

按危害排序,全部经代码核实确认:

- [x] **修 NegBin 离散度梯度(CRIT-3 残留)**:`grad_r_raw` 整体符号错误且缺 `y/(r+μ)` 项,
      正确式:`d(NLL)/d(log r) = -r*(digamma(y+r)-digamma(r)+log p+(1-p) - y/(r+μ))`;
      同时更新 `_distributions.py:1307` 处误导性的推导注释(结论与代码不符)
      — `src/openboost/_distributions.py:1319`
- [x] **修 CustomDistribution JAX 路径缺 link 链式法则**:数值路径已修但 JAX 路径(默认!)
      在约束空间求导、未乘 link 导数,梯度落在错误空间 — `src/openboost/_distributions.py:1605,1667`;
      顺带补二阶链式法则(当前 Hessian 丢了 `grad*link''` 项)— `_distributions.py:1595`
- [x] **把 growth 参数接进 GradientBoosting**:LeafWise/Symmetric 已实现但模型层没有 `growth` 字段,
      文档示例 `ob.GradientBoosting(growth='leafwise')` 直接 TypeError
      — `src/openboost/_models/_boosting.py:168`,`docs/user-guide/models/gradient-boosting.md:85`
- [x] **对称树 GPU 路径:修复或显式关闭(CRIT-5 修复是表面的)**:`_build_symmetric_histogram_kernel`
      不接收 sample_leaf_ids,每层直方图相同 → 树退化为逐层重复同一分裂;短期先路由回 CPU 正确路径
      — `src/openboost/_backends/_cuda.py:3640,3990`
- [x] **CUDA batch 路径:修复或门禁(CRIT-6 残留)**:每轮复用初始梯度,GPU 批量训练实际不工作;
      CPU 路径的梯度重算还硬编码 MSE — `src/openboost/_core/_tree.py:958,1010,900`
- [x] **声明 scipy 为运行时依赖**:裸装 openboost 后任何分布式模型 ModuleNotFoundError,
      CI 因 sklearn 传递依赖而掩盖 — `pyproject.toml:27`
- [x] **publish workflow 加测试门**:当前 release 不跑测试直接发 PyPI — `.github/workflows/publish.yml:7`
- [x] **分布族数值正确性测试套件(锁住以上所有修复)**:仿照 test_loss_correctness.py,
      对全部 7 个分布族做有限差分梯度校验(对 RAW 参数、穿过 link)+ scipy.stats 参照 NLL
      + "NLL 随轮次下降"断言;现有测试只查 shape/正性,CRIT-3 这类 bug 全部漏过
      — `tests/test_distributional.py:93,695`
- [x] 刷新 REVIEW_FINDINGS.md:标注已修复项及修复 commit,避免后续按过时文档排查

## Phase 1 — 场景一:分布式预测 ✅ 全部完成(2026-07-20)

目标用户能走完的标准工作流:Tweedie 定价、分位数备货、厚尾风险。审计确认的堵点:

- [x] **sample_weight + exposure/offset 全链路**(保险硬需求,当前完全无法表达精算工作流,
      且 sklearn wrapper 静默吞掉 kwargs 不报错)— `_models/_distributional.py:126`,`_models/_sklearn.py:697,737`
- [x] **eval_set 升级**:支持 CRPS/coverage/interval_score/NLL 可选指标、多 eval set、
      每轮指标 history;sklearn wrapper 显式转发 eval_set/callbacks(当前早停在 sklearn API 下不可用)
      — `_models/_distributional.py:222`
- [x] **PIT 校准诊断**:连续预测分布最核心的校准工具目前缺失(现有 calibration 工具是分类用的);
      加 PIT 值/直方图/可靠性图 + 可选事后再校准 — `src/openboost/_utils.py:664`
- [x] **Tweedie 预测正确性**:quantile 用正态近似会给零通胀理赔数据负下界(恰好打脸目标场景);
      NLL 缺归一化常数不是 proper score,不能和 NGBoost 对比;sample() 是 O(n²) 双层 Python 循环
      — 换支撑集感知的分位数 + Dunn-Smyth/鞍点近似似然 — `src/openboost/_distributions.py:1168,1204,1176`
- [x] **非负分布族的目标校验**:负 y / 非计数 y 目前被静默 clip,给出貌似合理的错误结果;改为报错
      — `src/openboost/_distributions.py:775,1376`
- [x] **CustomDistribution 的 Fisher**:当前恒等矩阵 → NaturalBoost 对自定义分布退化为普通梯度;
      用 JAX 算经验 Fisher 或文档明示此限制 — `src/openboost/_distributions.py:1681`
- [x] **跑通并提交 NGBoost 对比基准**:速度 + NLL/CRPS,bench 脚本已有但无提交结果,
      "faster than NGBoost" 目前无凭据 — `benchmarks/compare_gpu.py:276`
- [x] **GPU 叙事对齐事实**:分箱数据确实上 GPU 建树,但梯度/Fisher 全在 CPU numpy;
      要么把梯度计算搬上设备,要么把文档口径改准确(不许可测量的夸大)

## Phase 2 — 场景二:研究可魔改性(可以并行于 Phase 1 启动)(3–5 周)

custom loss 路径是真优势(文档准确、CPU/GPU 都通)。其余审计确认的差距:

- [ ] **注册 API 三件套**:`register_loss` / `register_distribution` / `register_growth_strategy`,
      终结"扩展必须改库源码"— `_core/_growth.py:1145`,`_distributions.py:1782`,`_loss.py:24`
- [x] **重写 custom-distributions.md**:当前示例的签名、参数顺序、参数名全错,照抄必崩
      — `docs/user-guide/naturalboost/custom-distributions.md:12`
- [ ] **扩展 cookbook**:每个扩展点一个可复制、被测试覆盖的完整示例(自定义损失/分布/生长策略)
- [x] **自定义 split criterion:实现或收回宣传**:营销反复以此为卖点但代码零扩展点
      (gain 硬编码在 primitives + CUDA kernel 里);先给 CPU 版 criterion hook,GPU 路径文档化
      — `docs/releases/ANNOUNCEMENTS.md:29`,`src/openboost/_core/_split.py`
- [ ] **设备端自定义损失 + 真实 loss_value 回调**:当前每轮 3 次 H2D/D2H 往返;
      自定义目标的日志/早停走 Taylor 代理而非真实损失 — `_models/_boosting.py:716`,`_loss.py:85`
- [x] **JAX 路径进 CI**:jax 目前只在 cuda12 extra,默认梯度路径在 CI 中从未执行过
      (这就是链式法则 bug 存活的原因)— `pyproject.toml:42`

## Phase 3 — 场景三/四:LinearLeaf(外推)与 GAM(可解释)(4–8 周,可裁剪)

### LinearLeafGBDT
- [x] **用断言证明外推**:现有测试显式不断言优于标准 GBDT(只查有限值),核心卖点零证据;
      改为 assert 趋势数据上界外 RMSE 显著更低 + 提交对比图 — `tests/test_linear_leaf.py:71`
- [x] **predict 向量化**:纯 Python 逐样本循环,生产规模不可用;先 NumPy 向量化,再评估 CUDA
      — `_models/_linear_leaf.py:87-109`
- [ ] eval_set/早停/callbacks;sklearn wrapper 停止静默吞 kwargs — `_models/_sklearn.py:960`
- [x] 修文档:`ridge_lambda` 参数不存在(实为 reg_lambda_linear),"继承全部 GradientBoosting 参数"不实
      — `docs/user-guide/models/linear-leaf.md:29`

### OpenBoostGAM
- [ ] **成对交互项(GA2M/FAST 风格)**:EBM 有、我们没有,监管场景(年龄×收入)是刚需;
      这是 GAM 最大能力缺口 — `_models/_gam.py:29`
- [ ] **相邻 bin 平滑 + 可选单调形状约束**:当前形状函数是逐 bin 原始 Newton 步,稀疏 bin 噪声大;
      文档教人调不存在的 max_depth"控制平滑度"— `_models/_gam.py:190`,`docs/user-guide/models/gam.md:23`
- [x] **诚实化 EBM 基准**:文档三行表(25x/39x/43x)无提交产物支撑;唯一数据点关掉了 EBM 的
      bagging 和交互,且 R² 0.66 vs 0.74 的精度差距未披露 —— 重跑固定种子基准、连精度一起公布
      — `README.md:149`,`benchmarks/compare_gpu.py:382`
- [x] plot_shape_function:加 `ax=`(文档示例现在会 TypeError)、x 轴映射回原始特征值、密度 rug
      — `_models/_gam.py:293`
- [x] n_bins 与硬编码 256 解耦或显式校验(>256 会越界)— `_models/_gam.py:129`
- [ ] eval_set/早停/callbacks(与 LinearLeaf 同一套改造);补 sklearn wrapper 测试 + GAMClassifier

## 贯穿:信任基础设施(只做诚实清单需要的)

- [ ] **GPU CI(nightly,Modal)**:跑 gpu/parity 标记套件;CRIT-5/6 能存活至今就是因为 CUDA 路径
      在 CI 中从未执行 — `tests/modal_gpu_tests.py` 已存在,接上即可
- [ ] XGBoost 差分测试已有(test_numerical_agreement.py),扩展覆盖到 growth 策略接线后的 leafwise/symmetric
- [ ] **文档-代码一致性检查**:本次审计发现 4 处文档示例照抄即崩(growth 参数、GAM max_depth、
      plot ax=、custom-distributions 签名)—— 把 docs 示例纳入 doctest/CI

## 成功标准(每场景一句话)

- 分布式预测:保险用户能带 exposure 跑 Tweedie、用 PIT 验证校准、拿到有据可查的 vs NGBoost 数字
- 可魔改性:三个扩展点都能不改库源码完成,cookbook 示例全部被 CI 执行
- 外推:测试断言 + 提交图表证明界外优于标准 GBDT
- 可解释:GAM 有交互项与平滑,基准连速度带精度一起诚实公布
- 信任:分布族有限差分测试全绿、GPU CI 每晚跑、发版有测试门

## 明确不做(non-goals)

- 标准库/生态位争夺叙事;自调参融合训练主线(保留 batch 路径修复,但只到"不再是坏的"为止)
- 通用单调/交互约束(GAM 的形状级单调除外)、小数据 TabPFN 对标、SHAP/ONNX 大生态工程
