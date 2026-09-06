# Sprint 010：有限组合链路与 F0.2 出口审计

起点：`972f80a`。状态：完成。补齐A5/A7/A9/C4的参考组合，完成F0.2出口审计。

## 计划与验收

1. 不可变scalar ensemble保存base/terms/系数；Poisson raw不含offset，两轮与新exposure
   预测；两阶段paid-count rate×Gamma severity真实拟合组合；三q模型并列预测。
2. Normal有序更新接版本化proposal、train/validation caches、best snapshot恢复；
   stale parent/NaN validation拒绝，失败不改变状态，恢复同步tree/coef/cache/step元数据。
   RNG仍由run ID/seed/logical step派生，重试不消费以后key。
3. 回归/lint、审计A1–A13与D1–D5的F0.2参考证据；只关闭有证据的阶段，明确F0.3/F1后续。

不实现序列化或公共runtime，不运行真实质量/速度/作者成本；这些分别仍在F1/F0.3–F4。
参考组合可使用已有reference组件，不能导入production或把内存恢复称为artifact round trip。

## 结果与验证

新增9项检查，总计288项通过，无跳过。Poisson两轮raw始终为log-rate，新exposure
只改变count mean；paid-count join接Poisson/Gamma真实参考训练，ClaimNb不能替代正赔付记录数。
三个q分别保留base、树与系数并重建预测。Normal两轮四次有序更新接版本化状态；
base和非零best均恢复完整terms、train/validation缓存及逻辑step，新version拒绝旧proposal。
反号拒绝、非有限validation、外来run、损坏raw、可广播的行数/shape错误均不改变原状态。
逻辑step派生key在重试/恢复后稳定；这里没有设备或全局RNG状态。

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

本地macOS、Python3.12.12、NumPy2.3.5；测试及lint通过。独立性测试在隔离进程
阻止所有openboost导入，仍能运行参考组合。严格文档构建通过。
完整范围映射见[F0.2验收表](f0-2-acceptance-ledger.md)。

## Reflection：结束数学准备，推进可评估的foundation

观察 → A1–A13与D1–D5的独立数学、反例及有限两轮组合已有证据。
决定 → 关闭F0.2参考准备，避免无限增加孤立fixture而迟迟不构建产品。
这没有关闭任何production conformance、持久化、真实质量、作者成本或GPU gate。
审核中修正了非零best测试的比较方式：初始化与后续更新必须使用同一metric，
否则best不具有可比意义。又补显式shape检查，不能依赖allclose阻止广播。

下一步 → F0.3建立真实数据manifest/hash、capability smoke、预算、保留任务和judge，
逐项保留全部required用例；F1再把通过独立oracle检验的组件实现为公共foundation。
参考Track是有限Normal状态探针，不是生产事务协议或磁盘checkpoint。

## Commits

- 本切片：`test: integrate reference models and close v1 F0.2`。
