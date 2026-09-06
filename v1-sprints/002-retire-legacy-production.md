# Sprint 002：退役旧生产代码，清空 v1 实现起点

日期：2026-09-05。起点：`50acfc6`。状态：**完成本 sprint；F0.2 继续进行中**。
触发：用户明确选择“旧生产代码，重新构建 v1”。这是用户要求的清理时序调整，
将原本 F5 的旧代码退役提前；F0.2/F0.3 与 F1 构建依赖、全部 required 用例和门槛不变。

## 计划和边界

1. 删除 `src/openboost/` 的旧 trainer/models/core/backends/experimental/distributed 等生产实现；
   只重建标明 under construction 的包命名空间和 typing marker，不提供旧 API shim。
2. 保留 `tests/v1/`、历史数学测试、benchmark 原始 artifacts、learnings 和设计。
   默认测试明确只验证新 v1；旧测试用固定 revision 复现，不能作为当前通过/跳过成绩。
3. 同步 README、包描述/依赖、默认测试、CI 和文档入口。旧模型示例不再作为当前 API
   宣传；旧 GPU/发布流程退役，不能把只有 oracle 的包当成可发布产品。
4. 检查 import/包内容、55 个 reference tests、lint、build 与文档构建；记录 reflection。

## 验收

- 当前包中不存在旧训练/模型/设备模块；可 import 的仅为新命名空间。
- 旧源码完整保存在 `50acfc6` 及之前的 Git 历史；历史实验和 tests/v1 未被删除。
- 默认测试与 `pytest tests/` 不再依赖旧生产包；输出明确是当前 v1 子集，不能称完整 v1。
- wheel 不含旧模块；README/文档不展示已经删除的可运行训练 API。
- 清理不新增算法/性能承诺，不把 F1 或 E-gate 标记完成。

## 验证记录

旧包清点为47个 tracked 文件（46个 Python 模块、1个 typing marker）；整体退役后
重新建立 `__init__.py` 和空 `py.typed`。最终包只含这两个文件，没有旧模块或兼容入口。
字节码/Numba 缓存也已清掉。版本改为 `1.0.0.dev0`，避免将空命名空间描述为旧 RC 产品。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv lock --check --offline
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv build --offline
```

- **55 passed，无 skipped**；默认入口仅收集当前 v1 tests，历史 suite 不计入结果。
- Ruff、lock check、严格文档构建通过；成功构建 sdist 和从 sdist 生成的 wheel。
- 检查 wheel 内容仅有两个包文件；`python -I` 直接从 wheel import，新旧 namespace
  不混用，所有旧 trainer/model/core/backend/distributed/experimental 模块均不可发现。
- 14 个变更 Markdown、76 个本地链接及代码围栏检查通过；`git diff --check` 通过。
- 4 个 workflow 的 YAML 结构检查通过。旧 GPU/发布入口只保留明确失败的手动状态说明，
  无定时 GPU 任务、无自动发布；docs workflow 只构建当前 `docs/v1/`，没有部署动作。
- `git diff HEAD -- benchmarks tests/v1` 为空：清理未修改历史 benchmark 或已提交 reference。
- 初次 `uv lock --offline` 因跨 Python 版本的元数据未缓存失败；联网生成 lock 成功，
  之后离线 check 通过，共96个解析包。没有通过手改 hash 绕过失败。
- 上述是本地验证；GitHub CI matrix、CUDA、真实模型质量、正式 E-gates 未运行。

## Reflection

观察：用户希望摆脱旧生产结构，而不是继续维护过渡实现。
证据：当前可独立运行的55个 reference tests 已提交；旧代码仍包含全局 backend、
旧权重/gain、多个模型/实验入口和不在 v1 目标内的分布式实现。
决定：一次退役旧生产层，保护数学反例与证据；当前库暂时没有训练 API，明确记录这个状态。
下一步：回到 F0.2 剩余参考，不用清理代替实际 foundation 构建。

收尾观察：移除旧生产层后，同一套55个独立参考无需修改仍全部运行。证据：reference
目录 diff 为空、隔离 import 检查和默认测试均通过。决定：保留这个独立边界；未来
production conformance 另写比较测试，不能把 reference 直接包装成宣称优化过的产品。
下一 sprint 从 typed data/binning/category/classification 参考开始，再按依赖补齐其余
F0.2 数学与 state/run；清理没有使任何 A 的生产实现或 E1/E3/E4 通过。

## Commits

- `50acfc6`：最后一个包含旧生产实现、同时有 v1 独立参考的 revision。
- 退役切片：`refactor: retire legacy production code for the v1 rebuild`。
