"""P0-1: LLM-as-Judge 端到端回答质量评估。

用 LLM 作为评判者，从四个维度对 RAG 系统的回答进行自动化评分：
  1. 准确性 (accuracy): 回答是否正确、事实是否准确
  2. 引用相关性 (citation_relevance): 引用是否支持回答中的论断
  3. 合规性 (compliance): 是否包含投资建议等违规内容
  4. 完整性 (completeness): 是否完整回答了问题的所有方面

每个维度 1-5 分，附带评判理由。最终输出汇总报告。

用法:
    from src.evaluation.answer_judge import AnswerJudge
    judge = AnswerJudge()
    result = judge.judge(
        question="货币基金的风险等级是什么？",
        answer="货币基金通常属于低风险（R1）产品...",
        references=[{"title": "...", "content": "..."}],
    )
    # result = {"accuracy": {"score": 4, "reason": "..."}, ...}
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config

# ══════════════════════════════════════════════════════════════════════
# 评判 Prompt
# ══════════════════════════════════════════════════════════════════════

_JUDGE_SYSTEM_PROMPT = """你是一个专业的金融 RAG 系统回答质量评判专家。
你的任务是从四个维度对系统回答进行评分，每个维度 1-5 分。

评分标准：
  5分 = 优秀，完全满足要求，无任何问题
  4分 = 良好，基本满足要求，有轻微不足
  3分 = 一般，部分满足要求，有明显不足
  2分 = 较差，大部分不满足要求
  1分 = 极差，完全不满足要求

评判原则：
  - 严格基于提供的问题、回答和引用内容评分，不使用外部知识
  - 金融场景下，准确性和合规性权重更高
  - 如果回答是"无法回答"或"需要澄清"，准确性和完整性给低分，但合规性给高分
  - 引用为空时，引用相关性给 1 分
"""

_JUDGE_USER_PROMPT_TEMPLATE = """请评判以下 RAG 系统回答：

【用户问题】
{question}

【系统回答】
{answer}

【引用文档】
{references_text}

请以 JSON 格式返回评分结果，不要其他内容：
{{
  "accuracy": {{
    "score": <1-5整数>,
    "reason": "准确性评判理由，说明回答是否正确、事实是否准确"
  }},
  "citation_relevance": {{
    "score": <1-5整数>,
    "reason": "引用相关性评判理由，说明引用是否支持回答中的论断"
  }},
  "compliance": {{
    "score": <1-5整数>,
    "reason": "合规性评判理由，说明是否包含投资建议、目标价等违规内容"
  }},
  "completeness": {{
    "score": <1-5整数>,
    "reason": "完整性评判理由，说明是否完整回答了问题的所有方面"
  }}
}}"""


# ══════════════════════════════════════════════════════════════════════
# 维度定义
# ══════════════════════════════════════════════════════════════════════

DIMENSIONS: tuple[str, ...] = ("accuracy", "citation_relevance", "compliance", "completeness")

DIMENSION_LABELS: dict[str, str] = {
    "accuracy": "准确性",
    "citation_relevance": "引用相关性",
    "compliance": "合规性",
    "completeness": "完整性",
}


class AnswerJudge:
    """LLM-as-Judge 回答质量评判器。

    使用配置的 LLM 对回答进行四维度评分。支持批量评判和汇总统计。
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        """初始化评判器。

        Args:
            model: 覆盖默认模型名（用于对比不同模型的评判一致性）
            temperature: 评判温度，默认 0.0 以保证评判稳定可复现
        """
        self._model_name = model or config.llm.model
        self._temperature = temperature
        self._llm = self._build_llm()

    def _build_llm(self):
        """根据配置构建 LLM 客户端。"""
        llm_config = config.llm
        if llm_config.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self._model_name,
                temperature=self._temperature,
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                timeout=llm_config.timeout,
            )
        elif llm_config.provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self._model_name,
                temperature=self._temperature,
                base_url=llm_config.base_url,
                client_kwargs={"trust_env": False, "timeout": llm_config.timeout},
            )
        else:
            raise ValueError(f"不支持的 LLM provider: {llm_config.provider}")

    def judge(
        self,
        question: str,
        answer: str,
        references: list[dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """评判单条回答。

        Args:
            question: 用户问题
            answer: 系统回答
            references: 引用文档列表，每个包含 title/content 等字段

        Returns:
            四维度评分结果，每个维度包含 score (1-5) 和 reason
        """
        references_text = self._format_references(references or [])
        user_prompt = _JUDGE_USER_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer[:3000],  # 限制回答长度，防止超出 context
            references_text=references_text[:4000],
        )

        response = self._llm.invoke(
            [SystemMessage(content=_JUDGE_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        return self._parse_judge_response(raw)

    def judge_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """批量评判。

        Args:
            items: 列表，每个元素包含 question, answer, references

        Returns:
            评判结果列表，每个元素包含原始输入 + 评分 + 汇总分
        """
        results: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            question = item.get("question", "")
            answer = item.get("answer", "")
            references = item.get("references", [])
            scores = self.judge(question, answer, references)
            overall = self._compute_overall_score(scores)
            results.append(
                {
                    "index": i,
                    "question": question,
                    "answer": answer[:500],
                    "scores": scores,
                    "overall_score": overall,
                    "passed": overall >= 3.0,
                }
            )
        return results

    @staticmethod
    def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
        """汇总批量评判结果。

        Args:
            results: judge_batch 的返回值

        Returns:
            汇总统计：各维度平均分、总体平均分、通过率、分布
        """
        if not results:
            return {"total": 0, "error": "无评判结果"}

        total = len(results)
        dimension_scores: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}
        overall_scores: list[float] = []
        passed_count = 0

        for r in results:
            scores = r.get("scores", {})
            for dim in DIMENSIONS:
                score = scores.get(dim, {}).get("score", 0)
                dimension_scores[dim].append(float(score))
            overall_scores.append(float(r.get("overall_score", 0)))
            if r.get("passed", False):
                passed_count += 1

        summary: dict[str, Any] = {
            "total": total,
            "passed": passed_count,
            "pass_rate": round(passed_count / total, 4),
            "overall_avg": round(sum(overall_scores) / total, 2),
            "dimensions": {},
        }

        for dim in DIMENSIONS:
            scores = dimension_scores[dim]
            avg = sum(scores) / len(scores) if scores else 0
            distribution = {str(i): scores.count(i) for i in range(1, 6)}
            summary["dimensions"][dim] = {
                "label": DIMENSION_LABELS[dim],
                "avg": round(avg, 2),
                "distribution": distribution,
            }

        # 低分案例（overall < 3）
        low_score_items = [r for r in results if r.get("overall_score", 5) < 3.0]
        summary["low_score_count"] = len(low_score_items)
        summary["low_score_examples"] = [
            {
                "question": r.get("question", "")[:100],
                "overall_score": r["overall_score"],
                "scores": {k: v["score"] for k, v in r.get("scores", {}).items()},
            }
            for r in low_score_items[:5]
        ]

        return summary

    @staticmethod
    def generate_markdown_report(summary: dict[str, Any]) -> str:
        """生成 Markdown 格式的评估报告。

        Args:
            summary: summarize 的返回值

        Returns:
            Markdown 报告字符串
        """
        lines: list[str] = []
        lines.append("# RAG 回答质量评估报告\n")
        lines.append(f"**评估样本数**: {summary['total']}  ")
        lines.append(f"**通过率**: {summary['pass_rate'] * 100:.1f}% ({summary['passed']}/{summary['total']})  ")
        lines.append(f"**总体平均分**: {summary['overall_avg']} / 5.0\n")

        lines.append("## 各维度评分\n")
        lines.append("| 维度 | 平均分 | 1分 | 2分 | 3分 | 4分 | 5分 |")
        lines.append("|---|---|---|---|---|---|---|")
        for dim in DIMENSIONS:
            d = summary["dimensions"][dim]
            dist = d["distribution"]
            lines.append(
                f"| {d['label']} | {d['avg']} | {dist.get('1',0)} | {dist.get('2',0)} | "
                f"{dist.get('3',0)} | {dist.get('4',0)} | {dist.get('5',0)} |"
            )

        if summary.get("low_score_count", 0) > 0:
            lines.append(f"\n## 低分案例（{summary['low_score_count']} 条）\n")
            for ex in summary.get("low_score_examples", []):
                lines.append(f"### {ex['question']}\n")
                lines.append(f"- 总体分: {ex['overall_score']}")
                scores_str = ", ".join(
                    f"{DIMENSION_LABELS[k]}={v}" for k, v in ex["scores"].items()
                )
                lines.append(f"- 各维度: {scores_str}\n")

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_references(references: list[dict[str, Any]]) -> str:
        """将引用文档格式化为文本。"""
        if not references:
            return "(无引用文档)"
        lines: list[str] = []
        for i, ref in enumerate(references[:5], 1):  # 最多 5 条引用
            title = ref.get("title", ref.get("source", "未知文档"))
            content = ref.get("content", ref.get("text", ""))[:500]
            lines.append(f"[{i}] {title}\n    {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_judge_response(raw: str) -> dict[str, dict[str, Any]]:
        """解析 LLM 返回的 JSON 评分结果。"""
        # 尝试提取 JSON（LLM 可能在 JSON 前后加了其他内容）
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
                # 验证并补全维度
                result: dict[str, dict[str, Any]] = {}
                for dim in DIMENSIONS:
                    if dim in data and isinstance(data[dim], dict):
                        score = int(data[dim].get("score", 0))
                        score = max(1, min(5, score))  # 钳位到 1-5
                        result[dim] = {
                            "score": score,
                            "reason": str(data[dim].get("reason", "")),
                        }
                    else:
                        result[dim] = {"score": 0, "reason": "LLM 未返回该维度评分"}
                return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 解析失败时返回默认值
        return {dim: {"score": 0, "reason": f"无法解析 LLM 响应: {raw[:200]}"} for dim in DIMENSIONS}

    @staticmethod
    def _compute_overall_score(scores: dict[str, dict[str, Any]]) -> float:
        """计算总体平均分（准确性和合规性权重加倍）。"""
        weights = {
            "accuracy": 2.0,
            "citation_relevance": 1.0,
            "compliance": 2.0,
            "completeness": 1.0,
        }
        total_weight = sum(weights.values())
        weighted_sum = sum(
            scores.get(dim, {}).get("score", 0) * weights[dim] for dim in DIMENSIONS
        )
        return round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0
