"""P2-3: PII 检测与脱敏——手机号、身份证、银行卡、邮箱。

用于查询输入和回答输出的 PII 扫描，防止敏感信息进入 LLM 或泄露给用户。
"""

from __future__ import annotations

import re

# 中国大陆手机号：1[3-9]xxxxxxxxx
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 身份证：18 位（最后一位可为 X），或 15 位
ID_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")
# 银行卡：16-19 位数字（简单匹配，不校验 Luhn）
BANK_CARD_PATTERN = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
# 邮箱
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

PII_PATTERNS = [
    ("phone", PHONE_PATTERN, "***PHONE***"),
    ("id_card", ID_CARD_PATTERN, "***ID_CARD***"),
    ("bank_card", BANK_CARD_PATTERN, "***BANK_CARD***"),
    ("email", EMAIL_PATTERN, "***EMAIL***"),
]


def detect_pii(text: str) -> list[dict[str, str]]:
    """检测文本中的 PII，返回 [{type, match, position}] 列表。

    按优先级处理：身份证 > 银行卡 > 手机号 > 邮箱。
    避免 18 位身份证号被同时匹配为银行卡（16-19 位数字模式）。
    """
    findings: list[dict[str, str]] = []
    if not text:
        return findings
    # 已匹配的字符区间，用于去重（高优先级模式先匹配，低优先级跳过重叠）
    occupied_spans: list[tuple[int, int]] = []
    # 优先级顺序：身份证先于银行卡
    priority_order = ["id_card", "bank_card", "phone", "email"]
    patterns_by_type = {p[0]: (p[1], p[2]) for p in PII_PATTERNS}
    for pii_type in priority_order:
        pattern, _ = patterns_by_type[pii_type]
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            # 检查是否与已匹配区间重叠
            if any(s < end and e > start for s, e in occupied_spans):
                continue
            occupied_spans.append((start, end))
            findings.append(
                {
                    "type": pii_type,
                    "match": match.group(),
                    "start": str(start),
                    "end": str(end),
                }
            )
    return findings


def redact_pii(text: str) -> tuple[str, list[dict[str, str]]]:
    """脱敏文本中的 PII，返回 (脱敏后文本, 检测到的 PII 列表)。

    按优先级处理：先银行卡（最长数字），再身份证，再手机号，最后邮箱。
    避免银行卡号被误匹配为手机号或身份证。
    """
    if not text:
        return text, []
    findings = detect_pii(text)
    redacted = text
    # 从长到短替换，避免短模式先替换破坏长模式
    for pii_type, pattern, replacement in sorted(PII_PATTERNS, key=lambda x: -len(x[1].pattern)):
        redacted = pattern.sub(replacement, redacted)
    return redacted, findings


def has_pii(text: str) -> bool:
    """快速判断文本是否包含 PII。"""
    if not text:
        return False
    return any(pattern.search(text) for _, pattern, _ in PII_PATTERNS)
