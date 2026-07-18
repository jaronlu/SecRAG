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
    resp = client.post(
        API_ROUTE_ASSISTANT_QA,
        headers={"Authorization": "Bearer demo-advisor"},
        json={"query": "这款理财产品风险等级是多少？"},
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise DemoValidationError("授权场景响应必须是 JSON 对象")
    _validate_allowed_response(data)
    _print_assistant_response(data)


def demo_agent_qa_denied(client: httpx.Client) -> None:
    """完整 Agent：demo-tech token 只能走其服务端绑定范围。"""
    _print_section("2. /v1/assistant/qa —— demo-tech token 查询受限资料")
    resp = client.post(
        API_ROUTE_ASSISTANT_QA,
        headers={"Authorization": "Bearer demo-tech"},
        json={"query": "内部研究摘要里对新能源板块怎么看？"},
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise DemoValidationError("权限拒绝场景响应必须是 JSON 对象")
    _validate_denied_response(data)
    _print_assistant_response(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT)
    args = parser.parse_args()

    with build_client(args.base_url, args.read_timeout) as client:
        try:
            demo_agent_qa_allowed(client)
            demo_agent_qa_denied(client)
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
