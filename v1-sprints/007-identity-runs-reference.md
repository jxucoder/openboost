# Sprint 007：identity、run 隔离与模型选择

起点：`c0e0b2c`。状态：完成本 sprint；F0.2进行中。B01/F0.2、A13/D5/C1/C4 的独立参考。

## 计划与验收

1. 明确有类型的内容hash与row-ID绑定；同shape不同内容、fold、schema、cuts、字典
   不能复用；target/weight/offset属于各自problem身份，不能只用prepared身份。
2. 顺序run参考执行真实小型scalar/vector平方损失树，独立预算、seed、best/early stop。
   每轮按稳定key采样，独立/顺序/重排/重组结果一致；故障保留且不污染其余run。
3. 只按validation选择已完成run；保存best terms与raw可重建。固定RNG派生fixture。
   核对F0.2剩余项，完成回归、lint、reflection与提交。

本切片的hash是独立语义参考，不是生产PreparedData、持久化格式或安全签名。
重组是顺序调度的语义模拟，不是批量GPU执行；不声称速度/显存优势。注入失败是故障处理
测试，不能把真实required run失败算作任务通过；当前E-gates均未完成。

## 结果与验证

本 sprint 有限范围完成。新增 `runs.py` 和21项测试，当前 **229 passed，无 skipped**；
Ruff通过。生产import隔离子进程也执行了一个真实run和固定seed断言。

- DataIdentity保存typed SHA256内容digest与不可变row IDs；hash覆盖值、行序、schema、
  transformer版本/cuts/字典。NaN规范化，None/string/int/float区分，mapping键顺序
  不改变digest。bind验证prepared行序及每个具名角色；不把hash当target合法性验证器。
- 修正审查发现的缺口：仅传digest不能验证调用者重新声明的行序，因此identity同时
  保留row IDs。即便所有fields同样反序，也不能绑定到原prepared顺序。
- 顺序执行真实平方损失树，K=1/K=2各自raw/预算；每轮joint提交所有输出。
  M=1/8/32保留全部记录；独立、顺序、反序、拆分重组结果完全一致。重组不是并行。
- validation严格改善才替换best；平局保留早先轮次。初始base作为round0候选。
  patience分别生效；训练损害validation时best回到base；存下的terms/coef重建best_raw。
- 故障run保存错误和已完成轮次，其余run不变；重试同ID首轮样本与故障前一致。
  config错误、无problem和全部失败均显式处理；真实required失败不会因为可选模型存在而通过。
- RNG是typed UTF-8 JSON的SHA256前64位，key=(seed,run_id,round,component,purpose)，
  无attempt/scheduler位置；fixture `(7,'run-α',2,'tree','rows')` → `6175955064790668999`。
  当前PCG64抽样仅作CPU参考，不承诺不同后端/未来NumPy默认Generator逐位一致。
- 选择器只读成功run的validation best；不同problem identity拒绝混排选择，即使loss
  都是数字。相同problem平局按run ID。并列异构运行不等于异构指标可以直接比较。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

环境：本地macOS CPU、Python3.12.12、NumPy2.3.5、pytest9.0.2。初始缺模块收集失败；
实现后首批7项通过，再扩展边界/固定key/M数量并修正lint。未运行GPU、生产cache/runtime、
真实模型选择成本、持久化或正式E-gates。当前run recipe只覆盖单位权重平方损失；通用
bind可包含weight/offset等role，不意味着这个run probe已经执行其他recipe。

## Reflection

观察 → K和M的区别、prepared和problem身份的区别均会影响正确性，不能只靠数组shape。
证据 → 异构K运行互不污染；改变目标/validation会改变problem ID且拒绝不可比选择；
统一反序fields仍会触发prepared行序校验。决定 → 公共foundation必须保存明确的
数据/目标/输出契约，调度只负责执行，不能猜指标是否可比。
下一步 → 已新增[F0.2验收映射](f0-2-acceptance-ledger.md)，明确D1/D3/D4精确fixture、
类别/向量完整grow及有限组合链路的缺口。先补这些，再关闭F0.2进入F0.3；不把已有
229项测试视为v1产品完成，不继续无限增加相同数学类型的测试。

## Commits

- 本切片：`test: add identity and isolated run references for v1`。
