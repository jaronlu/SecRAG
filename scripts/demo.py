"""端到端 Demo 脚本：验证 /v1/qa 与 /v1/assistant/qa 两个接口。

前置条件：
  1. 已完成入库：
     uv run python scripts/ingest.py src/data/samples/product product
     uv run python scripts/ingest.py src/data/samples/regulation regulation
     uv run python scripts/ingest.py src/data/samples/faq faq
     uv run python scripts/ingest.py src/data/samples/report research_report
  2. 已启动服务：
     uv run uvicorn src.api.main:app --port 8000

用法：
  uv run python scripts/demo.py [--base-url http://127.0.0.1:8000]

Demo token:
  demo-advisor / demo-sales / demo-compliance / demo-ops / demo-tech

会调用 .env 中配置的真实 LLM API（消耗真实调用额度），不在 CI/测试中自动执行。
"""

import argparse
import json
import sys

import httpx

_SEP = "=" * 70


def _print_section(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def _print_qa_response(data: dict) -> None:
    print(f"answer: {data['answer'][:200]}")
    print(f"confidence: {data['confidence']}")
    print(f"retrieval_path: {data['retrieval_path']}")
    print(f"citations ({len(data['citations'])} 条):")
    for c in data["citations"][:3]:
        print(f"  - {c}")


def _print_assistant_response(data: dict) -> None:
    print(f"answer: {data['answer'][:200]}")
    print(f"confidence: {data['confidence']}")
    compliance = data["compliance"]
    print(f"compliance.passed: {compliance.get('passed')}")
    print(f"compliance.flags: {compliance.get('flags')}")
    print(f"citations ({len(data['citations'])} 条):")
    for c in data["citations"][:3]:
        print(f"  - {c}")
    audit = data["audit_trail"]
    print(f"audit_trail.request_id: {audit.get('request_id')}")
    print(f"audit_trail.retrieval.sources: {audit.get('retrieval', {}).get('sources')}")


def demo_simple_qa(client: httpx.Client) -> None:
    """单轮 RAG：/v1/qa，不涉及角色权限，仅演示检索 + 生成 + 引用。"""
    _print_section("1. /v1/qa —— 单轮 RAG（理财产品风险等级咨询）")
    resp = client.post("/v1/qa", json={
        "query": "这款理财产品风险等级是多少？适合哪类客户？",
        "top_k": 5,
    })
    resp.raise_for_status()
    _print_qa_response(resp.json())


def demo_agent_qa_allowed(client: httpx.Client) -> None:
    """完整 Agent：服务端从 demo token 派生角色。"""
    _print_section("2. /v1/assistant/qa —— demo-advisor token 查询产品风险")
    resp = client.post(
        "/v1/assistant/qa",
        headers={"Authorization": "Bearer demo-advisor"},
        json={"query": "这款理财产品风险等级是多少？"},
    )
    resp.raise_for_status()
    _print_assistant_response(resp.json())


def demo_agent_qa_denied(client: httpx.Client) -> None:
    """完整 Agent：demo-tech token 只能走其服务端绑定范围。"""
    _print_section("3. /v1/assistant/qa —— demo-tech token 查询受限资料")
    resp = client.post(
        "/v1/assistant/qa",
        headers={"Authorization": "Bearer demo-tech"},
        json={"query": "内部研究摘要里对新能源板块怎么看？"},
    )
    resp.raise_for_status()
    _print_assistant_response(resp.json())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        try:
            demo_simple_qa(client)
            demo_agent_qa_allowed(client)
            demo_agent_qa_denied(client)
        except httpx.ConnectError:
            print(f"无法连接 {args.base_url}，请先启动服务：uv run uvicorn src.api.main:app --port 8000")
            sys.exit(1)
        except httpx.HTTPStatusError as exc:
            print(f"请求失败: {exc.response.status_code} {exc.response.text}")
            sys.exit(1)

    _print_section("Demo 完成")
    print(json.dumps({"status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
