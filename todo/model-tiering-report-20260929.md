# Ark 模型分层选型报告（ISSUE-20 产物）

> 2026-09-29 · 工具：`scripts/benchmark_models.py`（流式 TTFT / 总时长 / 吞吐，中位数取值，
> `trust_env=False` 直连绕开本机死代理；凭据仅从 .env 读取）

## 一、TTFT / 吞吐实测

固定两档提示词：small（模拟理解/计划 JSON 短输出，max_tokens 256）、long（约 200 字生成）。

| model | small TTFT ms | small total ms | long TTFT ms | long total ms | 备注 |
|---|---|---|---|---|---|
| deepseek-v4-flash（现役主模型） | 1866 | 2052 | 2093 | 4693 | 可用 |
| doubao-seed-1-6-flash | - | - | - | - | HTTP 404 UnsupportedModel（coding 端点不支持） |
| doubao-1-5-lite-32k | - | - | - | - | HTTP 404 UnsupportedModel |
| doubao-1-5-flash-32k | - | - | - | - | HTTP 404 UnsupportedModel |
| deepseek-v3 / v3.2 | - | - | - | - | HTTP 404 UnsupportedModel |

**结论：当前账号的 coding 计划端点（/api/coding/v3）仅 deepseek-v4-flash 可用。**
分层小模型（更快的 lite/flash 档）在订阅内无候选，无法启用第二档；
`OPENAI_PLAN_MODEL` 机制已落地，账号侧扩容模型清单后按本脚本复测即可切换。

## 二、已落地的分层机制

- `Settings.openai_plan_model`（.env: `OPENAI_PLAN_MODEL=<模型名>`）：理解/计划合并调用
  （ISSUE-11 节点）以请求级 `model` 参数覆盖主模型；空串回落主模型。
- `LLM_PLAN_MAX_TOKENS=1024`（ISSUE-14）：小任务输出预算，超限走 JSONDecodeError 回退。
- 实测 deepseek-v4-flash 的 small 档：TTFT 1.87s、总时长 2.05s —— 规划轮耗时被
  TTFT 主导，换更小模型的主要收益正是压缩这一段。

## 三、P0+P1 落地后端到端复测（本报告同日，端口 8001 新服务）

基线（2026-09-29 审计）：74.1s / 87.8s（验证误判重跑 + 3-8 次串行 LLM 往返）。

| 查询 | 修复后 | 说明 |
|---|---|---|
| 货币基金的风险等级是多少？ | **43.7s** | 两跳（首轮可用<2，二轮达标），reason 16.2s |
| 示例稳健增利理财产品的风险等级是多少？ | **23.0s** | 两跳，reason 3.4s |
| 什么是量化交易？（SSE） | 首字 35.3s / 188 个 delta / 总 37.8s | 真流式生效，reason 15.7s |

结构性改善（audit.db 瀑布佐证）：
- 验证失败重跑消失（ISSUE-13）：全部请求 reason iterations=1。
- 回环不再为凑证据数触发（ISSUE-12）：仅 0/低召回二跳。
- 恒失败工具轮次消失（ISSUE-10/19）：无 rerank/market 失败轮。
- reason 单轮 16.4-50.1s → 3.4-16.2s（ISSUE-13/14 预算与一次成功通过）。

## 四、残余差距（不满足设计的部分，如实记录）

1. 批次目标 P95 ≤25s：多跳 + 长 reason 查询（Q1 43.7s）未达标。
2. 设计线 P95 ≤10s / 流式首字 ≤5s：首字被 reason 前置的 1-2 轮规划
   （每轮 TTFT ~2s + 生成）与检索占据，需①更快的小模型（等待账号扩容）②
   reason 输出长度进一步压缩（ISSUE-14 预算 4096 的实际取值调优）。
3. SSE 首字 35.3s 中约 22s 为规划+检索（前置流水线），与 reason 流式本身无关；
   若要显著压首字，需考虑规划轮也流式转发或降低规划轮数（超出本批次范围）。
