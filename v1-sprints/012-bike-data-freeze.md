# Sprint 012：A5 Bike Sharing 数据与日期切分冻结

起点：`e2f2c6b`。状态：完成；仅A5数据准备。

## 计划与验收

1. 获取UCI原始zip，记录许可、zip/member hash与实际行数。只读hour.csv，calendar白名单，
   排除实测天气、casual/registered、instant与日期；原始instant仅作行ID。
2. 日期rolling origins训练前50/55/60/65/70%，后接10% validation和10% test；
   边界定义为floor(D*p/100)，固定每个split的日期/行数/row ID hash。
   最小失败测试：同一天不能跨split，天气/分项计数不能影响模型输入。
3. CLI验证冻结archive，生成可重现数据准备记录；反例、回归/lint、学习与提交。

不训练/评估模型，不使用test结果选协议；本轮仅A5数据准备，其他全部required用例继续保留。

## 结果与证据

[冻结记录](../benchmarks/v1/datasets/bike.json)包含原始zip/member、X/y/row IDs hash、
五个窗口的日期/行数/hash和运行provenance。实测17,379行、731天、7个calendar特征；
UCI页面标注17,389，保留差异，不以页面数字覆盖原始解析。
源码以adapter_sha256固定；记录诚实保留父revision与dirty=true，不冒充已提交代码运行。
本记录不是integrity-v0运行manifest，未触发完整训练评测。

五窗口(train/validation/test)行数依次为：
8645/1744/1750、9529/1746/1752、10389/1750/1752、11275/1752/1752、12139/1752/1752。
窗口内row ID和日期均不重叠；跨origin重叠符合rolling设计，不能按独立随机fold解读。
日期百分比按731个完整日期取floor，不按行数百分比分割，不填充缺失小时。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.bike /tmp/openboost-v1-bike.zip --verify benchmarks/v1/datasets/bike.json
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

实际archive重放完全匹配全部数据字段；手动损坏frozen test行数后CLI非零退出。
新增24项测试，总计360通过，无跳过；覆盖日期隔离、不均匀小时数、泄漏列、非法calendar/count、
重复ID/timestamp、archive损坏与adapter源码身份。macOS/Python3.12.12/NumPy2.3.5。

## Reflection

观察 → 来源网页行数与实际文件不同，且每日期小时数不固定。
决定 → 使用字节hash及实际解析，不把网页元数据或行数比例当作数据切分真值。
失败 → 沙箱内curl DNS不可用，获准网络下载后成功；初始测试模块不存在，后续新增参数化
用例时补回此前被lint移除的pytest import，最终全量通过。
下一步 → 接入其余数据，优先复用A1/A11 Housing已有原始hash并补seeds3–4；
再推进其他全部用例的数据/基线能力与预算。A5仍无模型结果，F0.3不关闭。

## Commits

- 本切片：`data: freeze A5 bike sharing inputs and rolling splits`。
