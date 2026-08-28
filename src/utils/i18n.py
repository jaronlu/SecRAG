"""P2-10: 语言检测——判断文本主要语言（中文/英文/混合）。

用于国际化：根据查询语言选择回答语言和提示词模板。
"""

from __future__ import annotations


def detect_language(text: str) -> str:
    """检测文本主要语言，返回 'zh' | 'en' | 'mixed'。

    简单基于字符比例：中文字符占比 > 30% 视为中文，
    英文字符占比 > 50% 视为英文，否则混合。
    """
    if not text:
        return "zh"  # 默认中文
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    english = sum(1 for c in text if c.isascii() and c.isalpha())
    total = len(text.strip())
    if total == 0:
        return "zh"
    if chinese / total > 0.3:
        return "zh"
    if english / total > 0.5:
        return "en"
    return "mixed"


def get_language_instruction(lang: str) -> str:
    """根据语言返回回答语言指令。"""
    if lang == "en":
        return "Please answer in English."
    if lang == "mixed":
        return "Please answer in the same language as the user's question."
    return "请用中文回答。"
