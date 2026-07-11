"""Run a fixed-rate HTTP load test against the production assistant route."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from scripts.evaluation_common import current_commit_sha


async def run_load_test(
    *,
    url: str,
    token: str,
    qps: int,
    duration: int,
    query: str,
) -> dict[str, float]:
    latencies: list[float] = []
    errors = 0
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=15) as client:
        async def request_once() -> None:
            nonlocal errors
            started = time.perf_counter()
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    json={"query": query},
                )
                if response.status_code >= 400:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            finally:
                async with lock:
                    latencies.append(time.perf_counter() - started)

        started = time.perf_counter()
        tasks = []
        total_requests = qps * duration
        for index in range(total_requests):
            target = started + index / qps
            await asyncio.sleep(max(target - time.perf_counter(), 0))
            tasks.append(asyncio.create_task(request_once()))
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started

    ordered = sorted(latencies)
    p95_index = max(int(len(ordered) * 0.95) - 1, 0)
    return {
        "requests": float(len(latencies)),
        "achieved_qps": len(latencies) / elapsed if elapsed else 0.0,
        "error_rate": errors / len(latencies) if latencies else 1.0,
        "p95_seconds": ordered[p95_index] if ordered else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SecRAG assistant 固定速率负载测试")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/assistant/qa")
    parser.add_argument("--token", default="demo-tech")
    parser.add_argument("--qps", type=int, default=50)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--query", default="系统操作流程是什么？")
    parser.add_argument("--output-root", default="artifacts/evaluation")
    args = parser.parse_args()
    summary = asyncio.run(
        run_load_test(
            url=args.url,
            token=args.token,
            qps=args.qps,
            duration=args.duration,
            query=args.query,
        )
    )
    output_dir = Path(args.output_root) / current_commit_sha()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "load.json"
    artifact.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary)
    print(f"评估产物: {artifact}")
    if (
        summary["achieved_qps"] < args.qps * 0.99
        or summary["error_rate"] >= 0.01
        or summary["p95_seconds"] > 10
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
