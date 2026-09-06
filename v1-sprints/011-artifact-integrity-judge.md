# Sprint 011：F0.3 评估产物完整性判卷

起点：`bc68ab9`。状态：完成；仅F0.3完整性判卷子集。

## 计划与验收

1. 建立 `benchmarks/v1/` 的版本化完整性协议：显式预期cells、运行身份、缓存键、产物hash。
   最小失败测试：删除一个required fold，即使其余case写pass也必须失败。
2. 实现离线judge和CLI，严格JSON、缺项/重复/未知case、缓存污染、非法NaN、错误backend、
   worker错误、超时、未运行GPU、缺预测/损坏文件反例。测试用合成产物，不假装真实数据冻结。
3. 全量回归/lint与文档，记录边界并提交。仅输出integrity_pass，不输出任何E-gate通过。

本轮不运行训练、不实现真实质量evaluator、不冻结尚未取得的数据hash/库版本或资源预算。
F0.3总体仍未完成；下一切片须推进真实数据和baseline准备。

## 结果与验收

新增48项测试，完整suite共336项。覆盖缺第二fold但全部A-ID仍在、重复/未知case、
八类输入改变后的缓存污染、非有限metric/prediction、假pass、非零worker、缺文件、
hash损坏、路径/symlink越界、optional GPU未支持和CLI退出码。
协议显式要求每个A-ID有required CPU cell；不把这一结构检查当作实验设计充分性。
每个已声明cell必须有记录，optional失败状态仍保留。通过只输出integrity_pass，gate_results为空。

缓存键绑定完整manifest与cell。仅dirty=false可用：dirty布尔值无法区分未提交补丁，
后续要支持dirty run必须加内容digest。当前不宣称核验代码、环境或数据声明的真实性。
有限矩形预测JSON只做完整性检查，未核对目标行/单位或重算metrics。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

本地macOS CPU、Python3.12.12；无真实训练、CUDA或质量结果。测试manifest明确为临时合成输入。

## Reflection

观察 → 数学参考已完成，但只看训练程序写的pass无法排除缺fold/缓存污染/worker故障。
决定 → 先建立可独立执行的完整性检查，并把报告字段与正式E-gate隔开。
失败尝试 → 实现前测试collection因模块不存在失败；实现后反例通过。
审核补充 → 单纯删除唯一A13记录不足以验证多fold，增加A1第二fold缺失而全部A-ID存在的反例。
下一步 → 推进真实数据来源/许可/hash与split adapter，不继续把schema扩展当作真实eval进展。
全部A1–A13仍required；预算、基线能力、保留题和完整judge未冻结，F0.3保持未完成。

## Commits

- 本切片：`test: add v1 artifact integrity judge`。
