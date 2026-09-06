# Sprint 013：A1/A11 Housing 五个split

起点：`dd2e9ad`。状态：完成；数据hash/五split子集，许可仍待核实。

## 计划与验收

1. 复核本地缓存archive的旧冻结hash，独立实现列映射/household比率/目标单位适配。
   最小测试用可手算原始行检验三个比率，防止房间/卧室/人口列错位。
2. 核对X/y组合hash和seeds0–2旧split hash；补齐3–4，记录member与全部split hash。
   检查每split互斥/完整、seed隔离、非法输入、float32溢出。
3. 真实文件重放、回归/lint/文档与提交。A1和A11共用数据但质量结果必须分别产出。

不训练模型、不运行GPU；公开来源的许可若未核实则保留unresolved，不编造授权标签。

## 结果与验收

实际缓存archive通过固定SHA256。新适配器独立读取原始九列，按households分别计算
AveRooms/AveBedrms/AveOccup，保持目标100,000 USD与little-endian float32。
20640×8的X/y组合hash与旧记录一致，旧0–2的九个split hash逐项一致；新增3–4六个hash。
[冻结记录](../benchmarks/v1/datasets/housing.json)保存member hash、五split和源码/环境provenance。
各split为12384/4128/4128，窗口内部完整互斥，随机划分不表示地理外推。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.housing build/foundation_data/cal_housing.tgz --verify benchmarks/v1/datasets/housing.json
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

真实数据重放匹配，损坏seed4 hash后CLI非零退出。新增20测试，总计380通过、无跳过；
非法household/count/shape/float32溢出拒绝，列/比率手算、无跨行学习、全局RNG不变通过。
本地macOS/Python3.12.12/NumPy2.3.5。原始数据未加入Git，未运行模型或查看test metrics。

## Reflection

观察 → 旧记录可复用，但只含三个seed，且完整许可信息没有随archive保存。
决定 → 保留原hash语义，独立验证后补齐两个seed；明确许可unresolved，不能把公共下载
推断为特定许可。A1与A11必须分别评价RMSE和NLL/CRPS，不能重复计作两个数据来源。
失败 → 实现前模块缺失导致collection失败；早期lint的单行分号由formatter修复。
下一步 → Adult分类/类别数据及其官方test切分，再补其他required数据；许可核实、
基线capability、预算、保留题和完整runner仍待完成。F0.3保持未完成。

## Commits

- 本切片：`data: freeze A1 A11 housing with five splits`。
