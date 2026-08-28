"""P2-2: 简单内存限流器——按用户 ID 滑动窗口限制请求频率。

生产环境应替换为 Redis 分布式限流；此处为单节点部署的轻量实现。
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

# 全局请求记录：key -> deque[timestamp]
_request_log: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()

# 默认配置：每分钟 30 次请求
DEFAULT_RATE_LIMIT = 30
DEFAULT_WINDOW_SECONDS = 60


def check_rate_limit(
    key: str,
    max_requests: int = DEFAULT_RATE_LIMIT,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> tuple[bool, int]:
    """检查 key 是否超过限流阈值。

    返回 (allowed, remaining)。allowed=False 时 remaining=0。
    """
    now = time.time()
    cutoff = now - window_seconds
    with _lock:
        log = _request_log[key]
        # 清理过期记录
        while log and log[0] < cutoff:
            log.popleft()
        if len(log) >= max_requests:
            return False, 0
        log.append(now)
        return True, max_requests - len(log)


def get_rate_limit_key(user_id: str | None = None, client_ip: str | None = None) -> str:
    """生成限流 key：优先 user_id，回退到 client_ip。"""
    return user_id or client_ip or "anonymous"
