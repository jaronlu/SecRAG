"""Small tracing helpers for tool-call observability."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ToolCallTrace:
    tool: str
    input: dict[str, Any]
    output: str
    duration_ms: float
    success: bool
    error: str = ""


class Tracer:
    """In-memory trace collector for tests and local debugging."""

    def __init__(self) -> None:
        self.traces: list[ToolCallTrace] = []

    def trace(self, func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                self.traces.append(
                    ToolCallTrace(
                        tool=func.__name__,
                        input={"args": args, "kwargs": kwargs},
                        output=str(result)[:500],
                        duration_ms=(time.perf_counter() - started) * 1000,
                        success=True,
                    )
                )
                return result
            except Exception as exc:
                self.traces.append(
                    ToolCallTrace(
                        tool=func.__name__,
                        input={"args": args, "kwargs": kwargs},
                        output="",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        success=False,
                        error=str(exc),
                    )
                )
                raise

        return wrapper  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        total = len(self.traces)
        return {
            "total_calls": total,
            "success_rate": (
                sum(1 for trace in self.traces if trace.success) / total if total else 0
            ),
            "avg_duration_ms": (
                sum(trace.duration_ms for trace in self.traces) / total if total else 0
            ),
            "calls": [asdict(trace) for trace in self.traces],
        }
