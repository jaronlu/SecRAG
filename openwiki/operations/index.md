# 文件

- [数据与离线作业：样例数据、证券数据抓取、持仓与每日扫描](data-and-jobs.md) - 覆盖 SecRAG 的数据资产与离线批处理：data/raw 样例知识库与真实证券数据产物的结构与用途、fetch_real_securities_data.py 的批量抓取（运行时成分股解析、内容哈希幂等、失败隔离、水位文件）、按 user_id 隔离的持仓/关注池持久化，以及 run_daily_scan 每日扫描、P0/P1/P2 事件分级与去重/水位不变式。
- [观测与运维：审计、指标、追踪、缓存与限流](observability.md) - SecRAG 的部署与排障入口：SQLite 审计模型与 outbox 降级、Prometheus /metrics 指标清单与 /health 摘要、Langfuse 追踪的 metadata 白名单与导出层脱敏两层防线及 fail-open 语义、答案语义缓存运维语义（默认关闭、角色隔离、只缓存成功终态、TTL 24h、命中补审计）、进程内滑动窗口限流，以及 docker-compose 单机部署形态与已知边界。
