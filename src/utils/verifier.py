"""Citation extraction and comprehensive answer verification."""

from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime, timezone

from src.schemas.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    META_CHUNK_ID,
    META_DOC_TYPE,
    META_PAGE_NUMBER,
    META_PERMISSION_LEVEL,
    META_SOURCE,
    META_TITLE,
    PERMISSION_PUBLIC,
    RR_CONTENT,
    RR_DENIED,
    RR_METADATA,
    RR_SCORE,
)
from src.schemas.models import Citation
from src.schemas.typed_dicts import CitationDict, RetrievalResult, ToolCallDict


class CitationExtractor:
    def extract(self, retrieval_results: list[RetrievalResult], query: str) -> list[CitationDict]:
        citations: list[CitationDict] = []
        eligible = [result for result in retrieval_results if not result.get(RR_DENIED)]
        seen_evidence = set()
        for result in eligible:
            metadata = result.get(RR_METADATA, {})
            evidence_key = (
                metadata.get(META_SOURCE),
                metadata.get(META_CHUNK_ID) or result.get(RR_CONTENT, ""),
            )
            if evidence_key in seen_evidence:
                continue
            seen_evidence.add(evidence_key)
            index = len(citations) + 1
            citation = Citation(
                citation_id=f"cite_{index:03d}",
                doc_title=str(metadata.get(META_TITLE, "未知文档")),
                source=str(metadata.get(META_SOURCE, "")),
                doc_type=str(metadata.get(META_DOC_TYPE, "")),
                chunk_id=str(metadata.get(META_CHUNK_ID, "")),
                quote=self._extract_quote(result.get(RR_CONTENT, ""), query),
                relevance_score=round(float(result.get(RR_SCORE, 0.0)), 4),
                permission_level=str(metadata.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC)),
                page_number=metadata.get(META_PAGE_NUMBER),
                retrieval_path=list(metadata.get("retrieval_path", ["vector_search"])),
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata=dict(metadata),
            )
            citations.append(CitationDict(**asdict(citation)))
            if len(citations) >= 5:
                break
        return citations

    def _extract_quote(self, content: str, query: str) -> str:
        sentences = [part.strip() for part in re.split(r"[。；\n]", content) if part.strip()]
        if not sentences:
            return content[:200]
        terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", query)}
        return max(
            sentences,
            key=lambda sentence: sum(term in sentence.lower() for term in terms),
        )[:200]


class SourceVerifier:
    def verify(
        self,
        answer: str,
        citations: list[CitationDict],
        retrieval_results: list[RetrievalResult],
    ) -> dict:
        issues: list[str] = []
        for cite_id in re.findall(r"\[来源(\d+)\]", answer):
            if int(cite_id) > len(citations):
                issues.append(f"引用来源 {cite_id} 不存在")
        if "[来源" in answer and not citations:
            issues.append("答案包含引用标注但无检索结果")

        evidence_keys = {
            (
                result.get(RR_METADATA, {}).get(META_SOURCE),
                result.get(RR_METADATA, {}).get(META_CHUNK_ID),
            )
            for result in retrieval_results
            if not result.get(RR_DENIED)
        }
        for citation in citations:
            key = (citation.get("source"), citation.get("chunk_id"))
            if key not in evidence_keys:
                issues.append(f"引用不属于当前轮检索结果: {key}")
        return {"passed": not issues, "issues": issues}


class NumberVerifier:
    def verify(
        self,
        answer: str,
        retrieval_results: list[RetrievalResult],
        tool_calls: list[ToolCallDict],
    ) -> dict:
        numbers = re.findall(r"\d+\.?\d*%?", answer)
        evidence = [result.get(RR_CONTENT, "") for result in retrieval_results]
        evidence.extend(
            str(call.get("output", "")) for call in tool_calls if call.get("success", False)
        )
        all_content = " ".join(evidence)
        issues = [
            f"数字 {number} 在检索或工具结果中未找到"
            for number in numbers
            if number not in all_content
        ]
        return {
            "passed": not issues,
            "issues": issues,
            "numbers_found": len(numbers) - len(issues),
            "numbers_total": len(numbers),
        }


class ConsistencyVerifier:
    def verify(self, answer: str, citations: list[CitationDict]) -> dict:
        del citations
        issues = []
        for positive, negative in (("增持", "减持"), ("买入", "卖出"), ("看多", "看空")):
            if positive in answer and negative in answer:
                issues.append(f"发现矛盾信息：'{positive}' 和 '{negative}' 同时出现")
        return {"passed": not issues, "issues": issues}


class HallucinationDetector:
    def detect(self, answer: str, retrieval_results: list[RetrievalResult]) -> dict:
        usable = [result for result in retrieval_results if not result.get(RR_DENIED)]
        if not usable:
            return {
                "passed": False,
                "issues": ["无检索结果支撑"],
                "hallucination_score": 1.0,
            }

        sentences = [part.strip() for part in re.split(r"[。；\n]", answer) if part.strip()]
        coverage = [
            any(self._similar(sentence, result.get(RR_CONTENT, "")) for result in usable)
            for sentence in sentences
            if not sentence.startswith(("【风险提示】", "【适当性提示】"))
        ]
        ratio = 1 - (sum(coverage) / len(coverage) if coverage else 1.0)
        issues = [f"幻觉比例过高：{ratio:.1%}"] if ratio > 0.3 else []
        return {"passed": not issues, "issues": issues, "hallucination_score": ratio}

    def _similar(self, text1: str, text2: str) -> bool:
        if text1 in text2 or text2 in text1:
            return True
        tokens1 = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text1.lower()))
        tokens2 = set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text2.lower()))
        if not tokens1:
            return False
        return len(tokens1 & tokens2) / len(tokens1) > 0.5


class ComprehensiveVerifier:
    def __init__(self):
        self.source_verifier = SourceVerifier()
        self.number_verifier = NumberVerifier()
        self.consistency_verifier = ConsistencyVerifier()
        self.hallucination_detector = HallucinationDetector()

    def verify(
        self,
        answer: str,
        citations: list[CitationDict],
        retrieval_results: list[RetrievalResult],
        tool_calls: list[ToolCallDict],
    ) -> dict:
        checks = {
            "source_verification": self.source_verifier.verify(
                answer, citations, retrieval_results
            ),
            "number_verification": self.number_verifier.verify(
                answer, retrieval_results, tool_calls
            ),
            "consistency_verification": self.consistency_verifier.verify(answer, citations),
            "hallucination_detection": self.hallucination_detector.detect(
                answer, retrieval_results
            ),
        }
        issues = [
            f"{name}: {issue}"
            for name, result in checks.items()
            for issue in result.get("issues", [])
        ]
        passed = all(result.get("passed", False) for result in checks.values())
        score = checks["hallucination_detection"].get("hallucination_score", 1.0)
        confidence = (
            CONFIDENCE_LOW
            if not passed
            else CONFIDENCE_MEDIUM
            if score > 0.1
            else CONFIDENCE_HIGH
        )
        return {
            "passed": passed,
            "issues": issues,
            "checks": checks,
            "confidence": confidence,
        }
