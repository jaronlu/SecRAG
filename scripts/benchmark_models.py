"""Benchmark Ark chat models for TTFT / throughput / latency (ISSUE-20).

Measures each candidate model with a fixed streaming prompt and prints a
markdown report used for the understand/plan vs reason model tiering.

Usage:
    uv run python scripts/benchmark_models.py MODEL_A MODEL_B ...

Model availability follows the account entitlements behind OPENAI_API_BASE;
candidates that fail are reported as unavailable rather than guessed.
Credentials are read from the environment / .env via src.config — never
passed on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import httpx

# 固定评测提示：短任务（模拟理解/计划 JSON 输出）+ 中长生成（模拟 reason），
# 均要求确定性较强的事实性输出，避免采样波动淹没延迟差异
BENCH_PROMPTS = {
    "small": (
        "请只返回一个 JSON 对象：{\"intent\": \"产品咨询\", "
        "\"rewritten_query\": \"货币基金的风险等级是什么\", \"ambiguity\": []}"
    ),
    "long": "用 200 字左右介绍货币基金的风险等级、流动性特征和适合的投资者类型。",
}

ROUNDS = 3
MAX_TOKENS = 256


def _run_once(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
) -> dict[str, Any] | str:
    """单次流式请求，返回 (ttft_ms, total_ms, chars) 或错误字符串。"""
    started = time.perf_counter()
    ttft: float | None = None
    chars = 0
    try:
        with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
            timeout=60.0,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")[:200]
                return f"HTTP {response.status_code}: {body}"
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content") or ""
                if text and ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                chars += len(text)
    except httpx.HTTPError as exc:
        return f"transport error: {exc!r}"[:200]
    if ttft is None:
        return "no content chunks received"
    total_ms = (time.perf_counter() - started) * 1000
    return {"ttft_ms": round(ttft, 1), "total_ms": round(total_ms, 1), "chars": chars}


def benchmark_model(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    rounds: int = ROUNDS,
) -> dict[str, Any]:
    """对单个模型跑 small/long 两档评测，各取中位数。"""
    report: dict[str, Any] = {"model": model}
    for name, prompt in BENCH_PROMPTS.items():
        results = []
        errors = []
        for _ in range(rounds):
            outcome = _run_once(client=client, base_url=base_url, api_key=api_key, model=model, prompt=prompt)
            if isinstance(outcome, dict):
                results.append(outcome)
            else:
                errors.append(outcome)
        if results:
            report[name] = {
                "ttft_ms": statistics.median(r["ttft_ms"] for r in results),
                "total_ms": statistics.median(r["total_ms"] for r in results),
                "chars": statistics.median(float(r["chars"]) for r in results),
                "ok_runs": len(results),
            }
        else:
            report[name] = None
        if errors:
            report[f"{name}_error"] = errors[0]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="候选模型名（按账号可用性自动报错）")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    # 延迟导入：复用 pydantic-settings 的 .env 装载与密钥脱敏
    from src.config import config

    api_key = config.llm.api_key.get_secret_value()
    base_url = config.llm.base_url
    if not api_key:
        raise SystemExit("OPENAI_API_KEY 未配置")
    # 编译端点与普通端点的 path 段不同，从配置读取即可，不在此处拼装

    rows = []
    with httpx.Client(trust_env=False) as client:
        for model in args.models:
            rows.append(
                benchmark_model(
                    client=client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    rounds=args.rounds,
                )
            )

    print("\n## 模型延迟基准（median of", args.rounds, "runs）\n")
    print("| model | small TTFT ms | small total ms | long TTFT ms | long total ms | long chars/s | 备注 |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        small = row.get("small") or {}
        long = row.get("long") or {}
        chars_per_s = ""
        if long.get("total_ms") and long.get("chars"):
            chars_per_s = f"{long['chars'] / (long['total_ms'] / 1000):.0f}"
        note = row.get("small_error") or row.get("long_error") or ""
        print(
            f"| {row['model']} "
            f"| {small.get('ttft_ms', '-')} | {small.get('total_ms', '-')} "
            f"| {long.get('ttft_ms', '-')} | {long.get('total_ms', '-')} "
            f"| {chars_per_s or '-'} | {note} |"
        )
    print()
    print("环境:", os.environ.get("LLM_PROVIDER", ""), base_url)


if __name__ == "__main__":
    main()
