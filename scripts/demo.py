"""端到端 Demo 脚本：验证带认证与权限控制的 /v1/assistant/qa 接口。

前置条件：
  1. 已完成入库：
     uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/product product
     uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/regulation regulation
     uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/faq faq
     uv run python scripts/ingest.py data/raw/demo_knowledge_base/samples/report research_report
  2. 已启动服务：
     uv run uvicorn src.api.main:app --port 8000

用法：
  uv run python scripts/demo.py [--base-url http://127.0.0.1:8000]

Demo token:
  demo-advisor / demo-sales / demo-compliance / demo-ops / demo-tech

会调用当前环境配置的真实 LLM；OpenAI-compatible provider 可能消耗调用额度，
本地 Ollama 不消耗远端额度。本脚本不在 CI/测试中自动执行。
"""

import argparse
import json
import sys

import httpx

from src.schemas.constants import API_ROUTE_ASSISTANT_QA

_SEP = "=" * 70
DEFAULT_READ_TIMEOUT = 180.0


class DemoValidationError(RuntimeError):
    pass


def build_client_timeout(read_timeout: float) -> httpx.Timeout:
    return httpx.Timeout(connect=5.0, read=read_timeout, write=30.0, pool=5.0)


def build_client(base_url: str, read_timeout: float) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        timeout=build_client_timeout(read_timeout),
        trust_env=False,
    )


def _print_section(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def _print_assistant_response(data: dict) -> None:
    print(f"answer: {data['answer'][:200]}")
    print(f"confidence: {data['confidence']}")
    compliance = data["compliance"]
    print(f"compliance.passed: {compliance.get('passed')}")
    print(f"compliance.flags: {compliance.get('flags')}")
    print(f"citations ({len(data['citations'])} 条):")
    for c in data["citations"][:3]:
        print(
            f"  - {c.get('citation_id')} | {c.get('doc_title')} | "
            f"chunk={c.get('chunk_id')} | score={c.get('relevance_score')}"
        )
        print(f"    quote: {str(c.get('quote', ''))[:160]}")


def _validate_allowed_response(data: dict) -> None:
    answer = data.get("answer")
    citations = data.get("citations")
    compliance = data.get("compliance")
    issues = []
    if not isinstance(answer, str) or not answer.startswith("## 结论"):
        issues.append("answer 必须以 ## 结论 开头")
    if not isinstance(answer, str) or "R2" not in answer:
        issues.append("answer 必须包含已验证事实 R2")
    if not isinstance(answer, str) or "[来源1]" not in answer or "[来源N]" in answer:
        issues.append("answer 必须包含有效 [来源1]，且不得包含 [来源N]")
    if not isinstance(citations, list) or not citations:
        issues.append("citations 必须至少包含一条引用")
    elif not any(
        isinstance(citation, dict) and "R2" in str(citation.get("quote", ""))
        for citation in citations
    ):
        issues.append("至少一条 citation quote 必须直接支持 R2")
    if not isinstance(compliance, dict) or compliance.get("passed") is not True:
        issues.append("compliance.passed 必须为 true")
    if data.get("confidence") not in {"medium", "high"}:
        issues.append("confidence 必须为 medium 或 high")
    if issues:
        raise DemoValidationError("授权场景验证失败: " + "; ".join(issues))


def _validate_denied_response(data: dict) -> None:
    answer = data.get("answer")
    citations = data.get("citations")
    compliance = data.get("compliance")
    issues = []
    if not isinstance(answer, str) or "无权限" not in answer:
        issues.append("answer 必须明确说明无权限")
    if citations != []:
        issues.append("citations 必须为空")
    if not isinstance(compliance, dict) or compliance.get("passed") is not False:
        issues.append("compliance.passed 必须为 false")
    elif "permission_denied" not in compliance.get("flags", []):
        issues.append("compliance.flags 必须包含 permission_denied")
    if data.get("confidence") != "low":
        issues.append("confidence 必须为 low")
    if issues:
        raise DemoValidationError("权限拒绝场景验证失败: " + "; ".join(issues))


def demo_agent_qa_allowed(client: httpx.Client) -> None:
    """完整 Agent：服务端从 demo token 派生角色。"""
    _print_section("1. /v1/assistant/qa —— demo-advisor token 查询产品风险")
    # 演练 ISSUE-4：必须用具体产品名提问。无产品名的模糊提问会命中 Agent 的
    # 澄清追问路径（citations=[]），与下方断言的"验证通过引用"形态不匹配
    resp = client.post(
        API_ROUTE_ASSISTANT_QA,
        headers={"Authorization": "Bearer demo-advisor"},
        json={"query": "示例稳健增利理财产品的风险等级是多少？"},
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise DemoValidationError("授权场景响应必须是 JSON 对象")
    _validate_allowed_response(data)
    _print_assistant_response(data)


def demo_agent_qa_denied(client: httpx.Client) -> None:
    """完整 Agent：advisor token 查询其角色不可见的受限内容。

    演练 ISSUE-6：拒绝路由在"全部检索结果被权限过滤"时触发。知识库重建后
    report_search 混入 1,662 条 public 券商研报 chunk（对所有角色可用），
    原问题（内部研究摘要/新能源）总会带回可用结果，拒绝分支不可达。
    本问题的检索只落在内部制度文档上：advisor 允许源不含 faq_search
    （计划级拒绝），regulation 文档 allowed_roles 排除 advisor（结果级拒绝），
    两条路径都保证 usable 为空，稳定进入干净拒绝。
    """
    _print_section("2. /v1/assistant/qa —— demo-advisor token 查询角色外受限资料")
    resp = client.post(
        API_ROUTE_ASSISTANT_QA,
        headers={"Authorization": "Bearer demo-advisor"},
        json={"query": "内部制度对客户数据导出申请的操作流程有什么要求？"},
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise DemoValidationError("权限拒绝场景响应必须是 JSON 对象")
    _validate_denied_response(data)
    _print_assistant_response(data)


# 面向真实 LLM 的场景允许少量重试：planner 与答案生成存在采样波动
# （如偶发的 product_type 过滤 0 命中、答案不带字面 R2/[来源1]），
# 断言本身保持严格，不因重试放宽
MAX_SCENARIO_ATTEMPTS = 3


def _run_scenario(title: str, scenario, client: httpx.Client) -> None:
    for attempt in range(1, MAX_SCENARIO_ATTEMPTS + 1):
        try:
            scenario(client)
            return
        except DemoValidationError as exc:
            print(f"[{attempt}/{MAX_SCENARIO_ATTEMPTS}] {title} 验证未通过: {exc}")
            if attempt == MAX_SCENARIO_ATTEMPTS:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT)
    args = parser.parse_args()

    with build_client(args.base_url, args.read_timeout) as client:
        try:
            _run_scenario("授权场景", demo_agent_qa_allowed, client)
            _run_scenario("权限拒绝场景", demo_agent_qa_denied, client)
        except httpx.ConnectError:
            print(f"无法连接 {args.base_url}，请先启动服务：uv run uvicorn src.api.main:app --port 8000")
            sys.exit(1)
        except httpx.TimeoutException:
            print(f"请求在读取响应时超过 {args.read_timeout:.1f} 秒，请检查服务端链路耗时")
            sys.exit(1)
        except httpx.HTTPStatusError as exc:
            print(f"请求失败: {exc.response.status_code} {exc.response.text}")
            sys.exit(1)
        except DemoValidationError as exc:
            print(str(exc))
            sys.exit(1)

    _print_section("Demo 完成")
    print(json.dumps({"status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
