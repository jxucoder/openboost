# GPU Python foundation：medium 执行清单

状态：P0、P1、P2、P3 已完成；P4.1–P4.2 已完成，下一项 P4.3。对应 [设计契约](gpu-python-foundation-design.md)。
P1 结果：本地结果协议 12 passed，真实单 T4 smoke 2 passed / 0 skipped，
wheel 来源和设备调用验证通过；[P1 learning 与原始结果](../learnings/2026-09-05-foundation-p1-modal.md)。
P0 结果：CPU 回归 749 passed / 34 skipped，加载器定向回归 21 passed，
lint、文档和打包通过。详情见 [P0 learning](../learnings/2026-09-05-foundation-p0-integration.md)。
本文件里的新模块、测试和 Modal 入口都是计划目标，不能当作已经存在。

## 使用方式

在 `codex/gpu-python-foundation-design` 上按依赖顺序执行，一次完成一个小任务：
读调用路径 → 写最小会失败的测试 → 实现 → 验证 → 更新 learning → 检查 staged diff → commit。
普通实现选择自行处理，不重复请用户批准已经授权的工作。用户已允许 Modal 用于本计划验证。
不 push、不 release、不改主分支，不自行联系外部作者。

如发现设计假设被证伪，先用最小复现说明事实，更新设计对应段落；不扩大支持面掩盖问题。
每次报告区分“代码完成”“CPU 通过”“真实 GPU 通过”“外部 adoption 尚未验证”。
若执行模型的资源窗口有限，完成一个已验证提交并标记下一个任务，不宣称整条路线完成。

## 当前快照

| 项目 | 规划时状态 |
|---|---|
| 原本分支 | `main`，干净 |
| 当前分支 | `codex/gpu-python-foundation-design` |
| 原始 HEAD | `82cf1e25b21a69093e85a270af7eb93c9ae7aa19` |
| 已 fetch 的远端 | `6ebe3a8ced0e621b17e3cf63e31721af58471053` |
| 本地/远端独有提交 | 12 / 9，尚未合并；后续设计提交不计在此数字中 |
| CPU 扩展基线 | `tests/test_extensibility.py`：35 passed, 1 GPU skipped |
| Modal | 已获用户授权；本轮没有启动、验证凭证或计费 |

执行前重新 `git status --short --branch`；不要假设用户未继续修改文件。
优先合并上面的固定 SHA 以保持计划可复现；之后新增远端提交另行审阅，不能无条件拉最新覆盖。

## P0：整合远端 unified trainer 与本地修复

**P0.1 — 固定基线并解决 merge。**

- 读本地 AGENTS、相关 learnings 与远端 `planning/unified-engine-design.md`。
- 验证当前分支包含本地原始 HEAD 与设计文件，工作区无未解释修改。
- 使用 merge 保留双方历史，不 rebase/reset：

```bash
git merge --no-commit --no-ff 6ebe3a8ced0e621b17e3cf63e31721af58471053
```

- 已知冲突：`CLAUDE.md` 保留指向 canonical AGENTS 的指引；GPU 安装文档结合远端新模型
  与本地限制描述，不能恢复没有 raw evidence 的性能宣传。
- 语义审阅 `_persistence.py`、`_array.py`、`_core/_tree.py`、`_models/_boosting.py`、
  CI、ScoringBench 与 examples，确认 categorical/save/load/batch guard 没被覆盖。
- 合并提交前运行 P0.2；这两步合为一个验证后的 merge 提交，不提交未解决冲突。

**P0.2 — 整合回归。**

最小检查先跑本地修复与远端新模型：

```bash
OPENBOOST_BACKEND=cpu uv run pytest tests/test_categorical.py tests/test_persistence.py tests/test_batch.py tests/test_extensibility.py tests/test_formula.py tests/test_survival.py -n 0 -q
```

再执行 CPU suite、生产代码 lint、文档 build 和 package build：

```bash
OPENBOOST_BACKEND=cpu uv run pytest tests/ -m "not gpu and not benchmark" --tb=short
uv run ruff check src/openboost/
uv run mkdocs build
uv build
```

执行时先用 `uv sync --locked --extra dev` 确保依赖；如环境平台限制某个可选依赖，
记录具体限制，不伪造完整通过。将已有失败与 merge 引入失败分开；与本计划相关的正确性失败先修，
其他遗留问题保留最小复现与明确影响范围。G0 未满足不能宣传新基座可用。

验收：两个父提交均为新 HEAD 的祖先；AGENTS/全部本地修复仍存在；新模型可导入、测试可运行。
提交建议：`merge: integrate unified trainer and local correctness fixes`。

## P1：建立可失败、可追溯的 Modal 测试通路

**P1.1 — 结果协议与测试包。** 主要文件：`benchmarks/foundation/`（独立 app，避免旧源码挂载）、
新 `tests/foundation/`、必要的依赖锁/测试配置、`benchmarks/results/.gitignore`。

- 先写本地单元测试：远端结果 failure、timeout、缺失报告、必跑 GPU test skipped 时，本地入口退出非零。
- manifest 生成不读取或打印 credentials；固定 source SHA、dirty、wheel/data hashes 与 argv。
- 打包必要 tests/conftest/config，锁 Linux/Python 3.12/CUDA 12 依赖，加入 CuPy 与 xdist。
  使用 uv 安装；不得让 benchmark 环境直接指向工作区 src。
- 仅显式允许 `benchmarks/results/foundation/` 中审核过的证据文件受 git 管理；
  保持其他临时输出忽略，不能用广泛 `git add -f` 把全部日志打包。

**P1.2 — 最小真 GPU smoke。** 新入口 `::foundation_smoke`；单 T4，300 秒，retry=0。

- 核对 CUDA 实际 device、Numba/CuPy 零拷贝视图的值与 owner 生命周期、wheel 安装来源。
- 运行一个内置 Normal 的极小 fit/predict，回传完整环境、JUnit 与状态；不做速度结论。
- 正反两种入口结果本地都能测试，不能只有“GPU available”布尔值。

计划命令（P1 实现后才存在）：

```bash
uv run modal run benchmarks/foundation/modal_app.py::foundation_smoke
```

验收：本地收到与源码 wheel 对应的真实 GPU 结果，失败状态能传播至 CLI。
记录 GPU 时间、完整命令和限制；任一失败优先修具体原因，不增加机型或长作业重试。

## P2：先修正确性，冻结旧路径基线

**P2.1 — weighted constant-Hessian 回归。** 主要文件：`_trainer.py`、`_objectives.py`、
对应 tests；路径均相对 `src/openboost/`，执行时以实际实现为准。

- 首个红测试：固定 binned X、固定 raw/grad，sample_weight 包含 0 与非均匀正值；
  比较 CPU 与 native CUDA 的 weighted histogram、Newton leaf 和一轮预测。
- 端到端 Normal/Poisson weighted fit 接着验证，不能仅用“传入了参数”的 mock 代替。
- 优先修复 const_hess eligibility；实际非恒定 hess 必须读数组。即便 uniform weight，
  首版也可关闭 hint，先保正确。排查初始化与 gradient 的权重语义，但不顺便改变所有模型算法。
- 静态怀疑如果未复现，记录为什么与保护测试；不能把本设计的“可能”写成已经修好的 bug。

**P2.2 — 能力、fallback 与 seed。**

- 红测试：自定义对象与内置 distribution 同名时不能命中内置 CUDA 数学；
  kernel RuntimeError 不能被宽泛 except 吞掉；unsupported capability 的回落须可见。
- 对相关 fit 增加 scoped RNG 路径，验证同 seed 重复 fit 一致、不同 seed 影响实际采样、
  调用前后全局 NumPy RNG 状态不变。MVP GPU 未支持采样时要拒绝；旧 CPU 采样需接 seed。
- 不支持参数在首次更新之前失败；测试 fitted attributes 不留下看似完成的半模型。

P2 验收：真实 T4 共 14 passed / 0 skipped；12 个基线配置、24 次拟合完成，
三组 seed 的质量门槛、fallback、callback/eval 和双向保存加载全部通过。
[基线与原始结果](../benchmarks/results/foundation/20260905T084129Z-2574e387/README.md)。

**P2.3 — 冻结整合后可用基线。**

- 用 `::foundation_correctness` 跑 Normal/Poisson 的 gradient → split → leaf → raw → metric；
  单独验证 CPU fallback、callback/eval 下载边界与 CPU/GPU 保存预测。
- 固定真实数据 hash、split 和 seeds；无 eval 的训练驻留与带 eval 的端到端任务分开记录。
- 将已有 default trainer/native 路径的正确性和冷/热时间存为 baseline，后续 P7 同配置对照。
  baseline 自身未满足质量/正确性时先修复，不能把坏基线作为新代码“通过”依据。

验收：G0/G2 的相关正确性通过；baseline 源码 SHA 干净、结果 committed。
首次基线采集可用限定 benchmark 入口，但不要重复运行完整矩阵。

## P3：CPU 上冻结最小公开契约

已完成：CPU facade、严格 objective、显式 builder、逐 channel schedule、
插件无关 raw inference 持久化和 coefficient-aware early stopping。
182 项相关测试通过；3 项旧 GBDT 长训练测试中断，未计为通过。
干净 `f414b8c` wheel 在隔离 Python 3.12 环境验证预测完全一致。
[契约、安装限制与复现](../learnings/2026-09-05-foundation-p3-cpu-contract.md)。
P3 只支持 CPU；native extension adapter 留到 P5。默认 CPU builder 暂拒绝
`reg_lambda=0/min_child_weight=0` 组合。G1 在这个明确边界内通过。

**P3.1 — 实验 facade 与 objective contract。** 主要文件：新
`src/openboost/experimental/__init__.py`、薄接口/类型模块、`_trainer.py`。

- 红测试：独立实现的两 channel objective 能通过实验 facade 运行；错 shape/device/keys、
  非有限值、权重错误、缺失能力、unsupported parameters 失败；现有 objective adapter 通过。
- 不复制训练循环。Booster config、context、BuiltTree、报告只承载设计里列出的内容。
- 冻结数学约定、float32/device/ownership，并给出 CPU 无近 tie oracle。
- 不把整个 `_core` re-export；只公开本计划真实用到的类型/函数。

**P3.2 — builder 与 schedule dispatch。**

- 红测试：特征恰好符合 native eligibility 时，显式 builder 仍被调用；
  两 channel 两轮的每个系数都实际作用一次，train raw 与重新 predict 完全对应。
- builder 不获取 raw 写引用；native adapter 使用 `pred_gpu=None`，trainer 统一加更新。
- 用户返回同一缓冲区并覆盖此前 channel 的错误通过契约测试发现；不得“修复”为每轮全量 host copy。
- 默认 schedule 回归旧 learning_rate；自定义 schedule 的值必须有限、非负且 channel 完整。

**P3.3 — 系数持久化与 callbacks。** 主要文件：`_persistence.py`、`_callbacks.py`、trainer/facade。

- 红测试：非常数逐 channel schedule 的 fit raw / predict / save-load 一致；
  early stopping 截断和 restore 同时处理 tree 与 coefficient。
- 实验模型保存后，在不安装扩展包的干净 CPU 环境仍可 raw inference；不 pickle 训练插件。
- 旧模型没有 coefficients 的文件按旧 learning_rate 加载；实验 facade 拒绝旧 categorical state；共享旧加载器原有警告策略保持不变。
- 共享 persistence 被触及的所有 tree state round trip 测试通过，特别是 missing/categorical/specialized leaves。

验收：实验 CPU 契约已实现、默认模型无相关回归、G1 通过。
P3 每步独立提交；不要等 GPU 完成后才发现预测/保存语义不一致。

## P4：按 primitive 逐步打通设备路径

主要新增文件建议：`src/openboost/_core/_batch_primitives.py`，以及现有 CPU/CUDA backend。
文件名可因代码组织调整，但职责与接口不能随意增加。

**P4.1 — batch histogram / CPU oracle。已完成。**

CPU 74 项相关测试通过；干净 `cf61611` wheel 在真实 T4 上 3 passed / 0 skipped。
G/H 独立 sample oracle 最大误差分别 9.835e-7 / 1.252e-6，计数完全一致。
[证据与边界](../benchmarks/results/foundation/20260905T150940Z-a5c80f7f/README.md)。
该提交只完成 histogram primitive；后续 split/routing 验证见 P4.2。

- 独立用直接按样本求和作为 oracle，不能用生产 histogram 函数生成 expected。
- 测 weighted/zero-weight、empty node、inactive slots、constant feature、保留 missing bin、memory budget。
- GPU 聚合结果留 device；不通过 legacy dict host wrapper。样本计数与 H 分开。

**P4.2 — split / routing。已完成。**

`b75b95a`：90 项相关 CPU 测试通过，真实 T4 上 4 passed / 0 skipped。
分割拓扑、精确 ties、gain 边界、实际样本 routing 和下一层统计通过独立 oracle。
[原始证据与限制](../benchmarks/results/foundation/20260905T151819Z-e5eb30b7/README.md)。
数值 L2、正曲率子节点；Booster GPU 集成仍待 P4.3/P4.4/P5。

- 对极小矩阵穷举所有合法分割，验证 gain、min_gain/min_child_weight、tie 顺序与无合法 split。
- 设备 partition 后的 node IDs 与 CPU oracle 对应；下一层 histogram 必须来源于实际 routed rows。
- 不使用父 histogram 按比例近似 child。默认完整固定槽位，空槽明确 masked。

**P4.3 — leaf reduction / rule。**

- 真正的 GPU sum/reduce；非均匀权重与零有效节点独立核对。
- 公开 leaf rule 返回同设备数组。bounded leaf 的值与下一轮梯度都要改变。
- 测不发生 grad/hess/node IDs 的下载，完整 histogram 不落 host。

**P4.4 — LevelWiseBuilder assembly。**

- 将以上 primitive 组成默认实验 builder，树完成后只下载紧凑 tree arrays。
- 保留 device prediction cache；默认 stream/视图生命周期测试覆盖一次 fit 后预测、释放临时 buffers。
- CPU/CUDA 上先验证一棵树、两轮两 channel，再扩大 n；不在这一步新写 vector leaves 或 leaf-wise。

每个 P4 小任务必须有真实 GPU parity 才能标记该 GPU 项通过。
共享一个已有镜像/小测试 selector，避免每改一行都跑完整 Modal suite。
GPU 不可用时可以完成 CPU 与测试准备，但不得跳过 GPU 关卡继续宣称设备路径完成。

### 使用价值检查提前

目标复查后的顺序调整：P4.4 的最小 builder 可运行后，先执行 P6.1/P6.2/P6.3
的 **CPU 独立 wheel** 部分，再完善 P5 的严格 GPU 集成。原有 GPU 验收门槛不降低。
记录公开接口之外的依赖、方法代码量、安装障碍、正确结果所需操作、保存后插件可移除性。
如仍需改 core，先修最小接口问题；不要用更多 GPU 功能掩盖使用障碍。
CPU 自编包仍不是外部 adoption；G5 保持未完成。随后完成 P5 和 P6 的 GPU 验证。

## P5：集成严格 GPU 执行与报告

**P5.1 — 实际 dispatch 与驻留。**

- 同一 trainer 能运行默认 builder / 外部 builder；strict GPU 所有能力开训前检查。
- objective/raw/y/weights 常驻设备；更新在设备；tree finalization 与显式 eval/输出复制记入报告。
- 确认没有 native fast path 绕过显式扩展、双重 raw 更新或双重 sample weight。
- host transfer wrapper 测试覆盖库内部路径；再用一个小 run profiler trace 核查外部 CuPy/内部 kernel。
  profiler 不可用时报告证据缺口，不能写“证明无传输”。

**P5.2 — 回归与负向测试。**

- numeric default NaturalBoost CPU/CUDA 对照；用户同名 objective、broken kernel、错误设备输出失败。
- categorical/missing/exposure/eval/采样/正则超出实验能力时拒绝或执行已声明完整 CPU fallback；
  不能静默降级成部分参数生效。合法旧 facade 的混合执行如保留，报告精确区分各阶段。
- GPU fit → save → CPU load prediction、非恒定 schedule、early-stop state 与已有 persistence 回归。

验收：G3 的核心通路通过，fit_report_ 与实际 trace 相符；修复必要的编译/内存生命周期问题。
若默认路径显著变慢，先保正确，进入 P7 profiling；不把融合写回 builder 的隐式 side effect。

## P6：两个真实安装边界与可复现示例

**P6.1 — normal_fisher package。** 按设计 A/C 实现独立 objective 与 schedule，
给 finite-difference gradient、Fisher analytic reference 和两轮更新例子。
使用公开类型/NumPy/CuPy；不要注册一个内置对象别名当作外部方法。

**P6.2 — bounded_leaves package。** 按设计 B 实现 leaf rule，
证明 clipping 在训练时生效、设备上运行、后续 raw/gradient 改变，保存后重现。

**P6.3 — wheel conformance。**

- 在 fresh venv 安装 OpenBoost wheel 与两个独立 wheel，从 repo 之外运行测试。
- 检查真实 module 路径；扩展源码只能 import 文档列出的 public OpenBoost 模块。
- 至少一个组合运行 A+B+C，验证扩展可以共存；分别运行以便定位错误。
- 卸载扩展包后加载保存模型进行 CPU raw inference；记录 build/lock/source hashes。
- README 给出安装、运行、CPU oracle、Modal GPU 验证以及实际支持限制，例子必须被测试运行。

验收：两个包均无需 private import、fork 或手动修改 core 即可完成；
三个扩展点都改变预期行为。此时只是技术 adoption 条件满足，尚无外部采用事实。

## P7：测价值、收敛文档并交付

**P7.1 — 匹配质量的性能记录。**

- 同一 Modal GPU、相同数据/split/config 对照 P2 baseline 与新默认路径。
- 分别记录冷启动/编译与 warm median、端到端 fit/predict、峰值显存、传输、逐 seed NLL/CRPS/coverage。
- 按设计预注册阈值检查，不因为失败而换 seed、删数据集或放宽阈值。
- 标准路径退化超过 20% 就 profiling；修复后仅重跑受影响比较，不扩展机型刷更好数字。
- 外部扩展计算不同，分别报告其质量与成本，不要求不同算法与默认时间相等。

**P7.2 — 开发者材料与文档。**

- 给一页“如何实现新 objective / leaf / schedule”、capability matrix、真实失败示例与复现命令。
- 为外部作者准备一个不附答案的扩展任务和记录表：到首个正确结果所需时间、求助次数、
  private imports/core changes、GPU结果、愿不愿意在独立包中依赖 OpenBoost。
- 不修改产品使命来预先宣布平台化成功；如果未有外部尝试，G5 显式未完成。

**P7.3 — 最终验证与结论。**

- 运行 CPU regression、必要 CUDA correctness、changed-file lint、docs build、wheel install conformance。
- 冻结 raw artifacts 与最终 source SHA 的关系；如果结果之后又改了相关代码，原结果不能为新代码背书。
- 更新本路线 learning，写清楚通过哪些 gate、失败原因、未支持参数、下一次外部验证动作。
- 工作区只留下可解释状态；给出各实现提交，不自动 push/merge main。

## 证据文件约定

建议冻结结构（新目录必须先加入精准 ignore 例外）：

```text
benchmarks/results/foundation/<run_id>/
  manifest.json        # source/wheel/data/env/argv/device provenance
  results.json         # per-test or per-seed raw results, includes failures
  junit.xml           # correctness jobs
  transfer_summary.json  # only when actually measured; declares coverage
  README.md           # reproduction, source commit, interpretation/limits
```

失败日志保留必要上下文，去掉敏感信息；不要提交完整 credentials/environment dump。
collect 时间与 measured 时间分开；自报的 GPU label 不能代替设备实际执行证据。
实现提交先完成，再从干净 SHA 运行，最后单独提交 artifact，避免循环引用源码 hash。

## 给 medium 模型的启动指令

> 继续当前 `codex/gpu-python-foundation-design` 分支。先读 AGENTS.md 与
> `planning/gpu-python-foundation-design.md`、`planning/gpu-python-foundation-execution.md`。
> 按 P0 起逐项执行，复用远端已有 unified trainer，保留本地正确性修复。
> 每个小任务先写有独立依据的失败测试，再实现、验证、更新 learning、检查 diff 并提交。
> 已授权用 Modal 做计划内单 GPU 验证；先做可追溯 smoke，再做 correctness，最后 benchmark。
> 不扩大到多 GPU/通用训练图，不跳过持久化，不用 CPU fallback 冒充 GPU 通过。
> 不 push 或合并 main。遇到证伪结果就据实记录并调整最小设计；明确区分技术验收和外部 adoption。
