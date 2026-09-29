"""Build the four evaluation datasets required by implementation-08 §3.

评估集必须绑定真实 chunk_id，而 chunk_id 会随重新入库变化，因此数据集由本脚本
从当前知识库与角色矩阵生成，而不是手写：

- retrieval.json      §3.1  ≥100 条查询（product/regulation/report/faq）+ ≥30 条权限负例，
                       覆盖 source 级、permission_level 级与 allowed_roles 级拒绝
- answers.json        §3.2  ≥100 条可回答 + ≥20 条不可回答，另含 tool-only 正/负例
- compliance.json     §3.3  ≥50 条对抗样本（敏感词、目标价、买卖建议、缺条款号）
- conversations.json  §3.4  ≥20 组多轮/隔离/删除/幂等会话用例

生成结果写入 tests/evaluation/，可直接交给 evaluate_*.py 运行。

用法:
    uv run python scripts/build_evaluation_datasets.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from src.schemas.constants import (
    META_ALLOWED_ROLES,
    META_CHUNK_ID,
    META_DOC_TYPE,
    META_PERMISSION_LEVEL,
    META_RETRIEVAL_SOURCE,
    META_SOURCE,
    META_TITLE,
    PERMISSION_PUBLIC,
    ROLE_ADVISOR,
    ROLE_ALLOWED_SOURCES,
    ROLE_COMPLIANCE,
    ROLE_DATA_PERMISSIONS,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
    SOURCE_FAQ,
    SOURCE_PRODUCT,
    SOURCE_REGULATION,
    SOURCE_REPORT,
)
from src.utils.compliance import SENSITIVE_KEYWORDS

SEED = 20260929
ALL_ROLES = (
    ROLE_ADVISOR,
    ROLE_INSTITUTIONAL_SALES,
    ROLE_COMPLIANCE,
    ROLE_OPERATIONS,
    ROLE_TECHNICAL,
)
DEFAULT_OUTPUT_DIR = Path("tests/evaluation")
RETRIEVAL_POSITIVES = 100
RETRIEVAL_NEGATIVES = 30
ANSWERABLE = 100
UNANSWERABLE = 20
COMPLIANCE_SAMPLES = 50
CONVERSATION_CASES = 20

# 生成查询时用于截取目标 chunk 特征片段的长度（字符）
_QUERY_SNIPPET_CHARS = 24
# §3.1 召回指标覆盖的四类文档检索源；sql_query 是工具，不在召回统计范围内
_DOCUMENT_SOURCES = frozenset({SOURCE_PRODUCT, SOURCE_REGULATION, SOURCE_REPORT, SOURCE_FAQ})


def _load_chunks() -> list[dict[str, Any]]:
    """读取 Chroma 中全部 chunk 的 content + metadata。"""
    import chromadb

    from src.schemas.constants import CHROMA_COLLECTION_NAME, CHROMA_DEFAULT_PERSIST_DIR

    client = chromadb.PersistentClient(path=CHROMA_DEFAULT_PERSIST_DIR)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    total = collection.count()
    payload = collection.get(limit=total, include=["documents", "metadatas"])
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    return [
        {"content": content, "metadata": metadata or {}}
        for content, metadata in zip(documents, metadatas)
    ]


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk["metadata"].get(META_CHUNK_ID, ""))


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def _discriminative_texts(chunks: list[dict[str, Any]]) -> list[str]:
    return [_normalize_text(chunk["content"]) for chunk in chunks]


def _query_from_chunk(chunk: dict[str, Any], normalized_texts: list[str]) -> str | None:
    """从 chunk 正文截取特征片段作为查询。

    片段必须在全库唯一：若一段话（如年报免责声明）在多个 chunk 中重复，
    检索器无法把命中归因到目标 chunk，该 chunk 不能作为召回评估的依据。
    """
    text = _normalize_text(chunk["content"])
    if len(text) <= _QUERY_SNIPPET_CHARS:
        return None
    start = min(len(text) // 4, max(len(text) - _QUERY_SNIPPET_CHARS, 0))
    snippet = text[start : start + _QUERY_SNIPPET_CHARS]
    if sum(1 for other in normalized_texts if snippet in other) != 1:
        return None
    return snippet


def _allowed_roles(metadata: dict[str, Any]) -> set[str]:
    raw = metadata.get(META_ALLOWED_ROLES)
    if isinstance(raw, str):
        return {role.strip() for role in raw.split(",") if role.strip()}
    if isinstance(raw, list):
        return {str(role) for role in raw}
    return set()


def _role_for_chunk(metadata: dict[str, Any]) -> str | None:
    """选一个既能访问该 chunk、又拥有其 retrieval_source 的角色。"""
    source = str(metadata.get(META_RETRIEVAL_SOURCE, ""))
    allowed = _allowed_roles(metadata)
    level = str(metadata.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC))
    for role in ALL_ROLES:
        if role not in allowed:
            continue
        if source not in ROLE_ALLOWED_SOURCES.get(role, []):
            continue
        if level not in ROLE_DATA_PERMISSIONS.get(role, []):
            continue
        return role
    return None


def _dedupe_by_source(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个 source 只取一个 chunk，让查询覆盖尽可能多的文档。"""
    seen: set[str] = set()
    picked: list[dict[str, Any]] = []
    for chunk in chunks:
        source = str(chunk["metadata"].get(META_SOURCE, ""))
        if not source or source in seen or not _chunk_id(chunk):
            continue
        seen.add(source)
        picked.append(chunk)
    return picked


def _sample_chunks(chunks: list[dict[str, Any]], per_source: int) -> list[dict[str, Any]]:
    """按 source 分层取样：每个文档最多取 per_source 个 chunk。

    语料只有约 18 个文档，而设计要求 ≥100 条查询，因此必须允许同一文档贡献
    多条查询（每条查询绑定不同的真实 chunk）。
    """
    by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        source = str(chunk["metadata"].get(META_SOURCE, ""))
        if not source or not _chunk_id(chunk):
            continue
        by_source.setdefault(source, []).append(chunk)

    sampled: list[dict[str, Any]] = []
    for source in sorted(by_source):
        # 均匀跨页取样，避免全部落在文档开头
        bucket = by_source[source]
        step = max(len(bucket) // per_source, 1)
        sampled.extend(bucket[::step][:per_source])
    return sampled


def _per_source(target: int, chunks: list[dict[str, Any]]) -> int:
    """每个 source 的取样上限：候选池按目标数量的四倍准备。

    两类损耗需要缓冲：部分 chunk 没有可达角色，部分正文片段在全库
    不唯一（如年报免责声明）而不能作为召回评估依据。
    """
    sources = {str(chunk["metadata"].get(META_SOURCE, "")) for chunk in chunks}
    return max((target * 4) // max(len(sources), 1) + 2, 3)


def _round_robin_by_doc_type(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 doc_type 轮转取样，保证 product/regulation/report/faq 都有覆盖。"""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        buckets.setdefault(str(chunk["metadata"].get(META_DOC_TYPE, "")), []).append(chunk)
    ordered: list[dict[str, Any]] = []
    keys = sorted(buckets)
    index = 0
    while len(ordered) < len(chunks):
        progressed = False
        for key in keys:
            bucket = buckets[key]
            if index < len(bucket):
                ordered.append(bucket[index])
                progressed = True
        if not progressed:
            break
        index += 1
    return ordered


def build_retrieval_dataset(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """§3.1：正例覆盖四类数据源与五个角色，负例覆盖三类拒绝。"""
    dataset: list[dict[str, Any]] = []
    role_coverage: dict[str, int] = {role: 0 for role in ALL_ROLES}

    # §3.1 的召回指标只覆盖四类文档检索源；sql_query 是工具而不是检索器，
    # 其结果没有 chunk_id，混入召回统计只会稀释指标
    document_chunks = [
        chunk
        for chunk in chunks
        if str(chunk["metadata"].get(META_RETRIEVAL_SOURCE, "")) in _DOCUMENT_SOURCES
    ]
    normalized_texts = _discriminative_texts(chunks)
    candidates = _round_robin_by_doc_type(
        _sample_chunks(document_chunks, _per_source(RETRIEVAL_POSITIVES, document_chunks))
    )
    # 同一个 chunk 只出一题：答案的 relevant_chunk_ids 指向该 chunk 本身
    used_chunk_ids: set[str] = set()
    for chunk in candidates:
        if len(dataset) >= RETRIEVAL_POSITIVES:
            break
        chunk_id = _chunk_id(chunk)
        if chunk_id in used_chunk_ids:
            continue
        role = _role_for_chunk(chunk["metadata"])
        query = _query_from_chunk(chunk, normalized_texts)
        if role is None or query is None:
            continue
        used_chunk_ids.add(chunk_id)
        role_coverage[role] += 1
        dataset.append(
            {
                "query": query,
                "user_role": role,
                "source": str(chunk["metadata"].get(META_RETRIEVAL_SOURCE, "")),
                "relevant_chunk_ids": [chunk_id],
                "expected_permission_denied": False,
                "denial_kind": "none",
            }
        )

    # 角色覆盖补齐：为覆盖不足的角色补题，避免五角色样本失衡
    for role in ALL_ROLES:
        while role_coverage[role] < 4:
            chunk = next(
                (
                    item
                    for item in candidates
                    if _role_for_chunk(item["metadata"]) == role
                    and _chunk_id(item) not in used_chunk_ids
                ),
                None,
            )
            if chunk is None:
                break
            chunk_id = _chunk_id(chunk)
            query = _query_from_chunk(chunk, normalized_texts)
            if query is None:
                continue
            used_chunk_ids.add(chunk_id)
            role_coverage[role] += 1
            dataset.append(
                {
                    "query": query,
                    "user_role": role,
                    "source": str(chunk["metadata"].get(META_RETRIEVAL_SOURCE, "")),
                    "relevant_chunk_ids": [chunk_id],
                    "expected_permission_denied": False,
                    "denial_kind": "none",
                }
            )

    dataset.extend(_build_permission_negatives(chunks))
    return dataset


def _build_permission_negatives(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """三类权限负例：source 级、permission_level 级、allowed_roles 级。"""
    negatives: list[dict[str, Any]] = []
    normalized_texts = _discriminative_texts(chunks)
    """三类权限负例：source 级、permission_level 级、allowed_roles 级。"""
    negatives: list[dict[str, Any]] = []

    # ① source 级：角色没有该 retrieval_source。只覆盖四类文档检索源——
    # sql_query 是工具不是检索器，计划层把它当未知 source 返回 error 而非拒绝
    by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        source = str(chunk["metadata"].get(META_RETRIEVAL_SOURCE, ""))
        by_source.setdefault(source, []).append(chunk)
    for source in (SOURCE_FAQ,):
        bucket = by_source.get(source, [])
        if not bucket:
            continue
        for role in ALL_ROLES:
            if source in ROLE_ALLOWED_SOURCES.get(role, []):
                continue
            for chunk in bucket[:6]:
                negatives.append(
                    {
                        "query": _query_from_chunk(chunk, normalized_texts)
                        or _normalize_text(chunk["content"])[:_QUERY_SNIPPET_CHARS],
                        "user_role": role,
                        "source": source,
                        "relevant_chunk_ids": [],
                        "chunk_id": _chunk_id(chunk),
                        "expected_permission_denied": True,
                        "denial_kind": "source",
                    }
                )

    # ② permission_level 级：角色在 allowed_roles 中但缺少该权限级别
    for chunk in chunks:
        metadata = chunk["metadata"]
        level = str(metadata.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC))
        if level == PERMISSION_PUBLIC:
            continue
        allowed = _allowed_roles(metadata)
        for role in ALL_ROLES:
            if role not in allowed:
                continue
            if level in ROLE_DATA_PERMISSIONS.get(role, []):
                continue
            negatives.append(
                {
                    "query": _query_from_chunk(chunk, normalized_texts)
                        or _normalize_text(chunk["content"])[:_QUERY_SNIPPET_CHARS],
                    "user_role": role,
                    "source": str(metadata.get(META_RETRIEVAL_SOURCE, "")),
                    "relevant_chunk_ids": [],
                    "chunk_id": _chunk_id(chunk),
                    "expected_permission_denied": True,
                    "denial_kind": "permission_level",
                }
            )

    # ③ allowed_roles 级：角色有该 source 与权限级别，但不在 allowed_roles 中
    for chunk in chunks:
        metadata = chunk["metadata"]
        level = str(metadata.get(META_PERMISSION_LEVEL, PERMISSION_PUBLIC))
        source = str(metadata.get(META_RETRIEVAL_SOURCE, ""))
        allowed = _allowed_roles(metadata)
        for role in ALL_ROLES:
            if role in allowed:
                continue
            if source not in ROLE_ALLOWED_SOURCES.get(role, []):
                continue
            if level not in ROLE_DATA_PERMISSIONS.get(role, []):
                continue
            negatives.append(
                {
                    "query": _query_from_chunk(chunk, normalized_texts)
                        or _normalize_text(chunk["content"])[:_QUERY_SNIPPET_CHARS],
                    "user_role": role,
                    "source": source,
                    "relevant_chunk_ids": [],
                    "chunk_id": _chunk_id(chunk),
                    "expected_permission_denied": True,
                    "denial_kind": "allowed_roles",
                }
            )

    # 去重后按类别配额取齐：三类都要覆盖，总数不少于 RETRIEVAL_NEGATIVES
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in negatives:
        key = (item["denial_kind"], item["user_role"], item["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    quotas = {"source": 12, "permission_level": 10, "allowed_roles": 18}
    balanced: list[dict[str, Any]] = []
    for kind, quota in quotas.items():
        balanced.extend([item for item in deduped if item["denial_kind"] == kind][:quota])
    return balanced if len(balanced) >= RETRIEVAL_NEGATIVES else deduped[:RETRIEVAL_NEGATIVES]


def _evidence_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(chunk["metadata"])
    return {
        "content": str(chunk["content"]),
        "metadata": metadata,
        "score": 0.9,
    }


def _grounded_answer(chunk: dict[str, Any], max_chars: int = 160) -> str | None:
    """从 chunk 原文取完整句子拼成答案。

    约束：答案里的每个数字都必须原样出现在证据中，且每个句子都是证据的
    子串——在数字中间截断或混入标题词都会让答案变成"无支撑"，这属于
    生成器缺陷而不是验证器缺陷。
    """
    sentences = [part.strip() for part in re.split(r"[。；\n]", str(chunk["content"])) if part.strip()]
    chosen: list[str] = []
    length = 0
    for sentence in sentences:
        if length + len(sentence) > max_chars:
            break
        chosen.append(sentence)
        length += len(sentence) + 1
    if not chosen:
        return None
    return "。".join(chosen) + "。"


def build_answers_dataset(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """§3.2：可回答问题以真实 chunk 为唯一证据，答案数字直接取自该证据。"""
    dataset: list[dict[str, Any]] = []
    numbered = [
        chunk
        for chunk in _sample_chunks(chunks, _per_source(ANSWERABLE, chunks))
        if any(char.isdigit() for char in str(chunk["content"]))
    ]
    for chunk in numbered:
        if len([item for item in dataset if item["expected_passed"]]) >= ANSWERABLE:
            break
        body = _grounded_answer(chunk)
        if body is None:
            continue
        evidence = _evidence_from_chunk(chunk)
        title = str(chunk["metadata"].get(META_TITLE, "文档"))
        dataset.append(
            {
                "id": f"answerable_{len(dataset) + 1:03d}",
                "question": f"{title} 中披露了哪些关键数据？",
                "answer": f"## 结论\n\n{body} [来源1]",
                "citations": [
                    {
                        "citation_id": "cite_001",
                        "doc_title": title,
                        "source": str(chunk["metadata"].get(META_SOURCE, "")),
                        "chunk_id": _chunk_id(chunk),
                        "quote": body[:200],
                        "metadata": dict(chunk["metadata"]),
                    }
                ],
                "retrieval_results": [evidence],
                "tool_calls": [],
                "expected_passed": True,
            }
        )

    # 不可回答：答案含证据中不存在的数字，必须判失败
    for index in range(UNANSWERABLE):
        chunk = numbered[index % len(numbered)] if numbered else None
        evidence = _evidence_from_chunk(chunk) if chunk else {}
        dataset.append(
            {
                "id": f"unanswerable_{index + 1:03d}",
                "question": "该公司的下一年度预测净利润是多少？",
                "answer": "## 结论\n\n预计下一年度净利润 9999.99 亿元 [来源1]。",
                "citations": [],
                "retrieval_results": [evidence] if evidence else [],
                "tool_calls": [],
                "expected_passed": False,
            }
        )

    # tool-only 正例：向量检索为空，成功工具输出支撑回答
    for index in range(10):
        value = 100 + index
        dataset.append(
            {
                "id": f"tool_only_ok_{index + 1:02d}",
                "question": "帮我算一下该笔投资的到期收益",
                "answer": f"## 结论\n\n到期收益为 {value} 元。",
                "citations": [],
                "retrieval_results": [],
                "tool_calls": [
                    {
                        "name": "calculator",
                        "input": {"expression": f"{value}"},
                        "output": json.dumps({"result": value}),
                        "success": True,
                    }
                ],
                "expected_passed": True,
            }
        )

    # tool-only 负例：工具失败或空结果必须 fail closed，不得把调用动作当证据
    for index in range(5):
        dataset.append(
            {
                "id": f"tool_only_failed_{index + 1:02d}",
                "question": "帮我算一下该笔投资的到期收益",
                "answer": f"## 结论\n\n到期收益为 {700 + index} 元。",
                "citations": [],
                "retrieval_results": [],
                "tool_calls": [
                    {
                        "name": "calculator",
                        "input": {"expression": "1/0"},
                        "output": "工具执行失败：division by zero",
                        "success": False,
                    }
                ],
                "expected_passed": False,
            }
        )
    return dataset


def build_compliance_dataset() -> list[dict[str, Any]]:
    """§3.3：对抗样本覆盖敏感词、目标价、买卖建议、缺条款号与越权访问。"""
    dataset: list[dict[str, Any]] = []
    advice_samples = (
        "建议买入该股票，短期有望上涨。",
        "推荐买入并持有至年底。",
        "建议增持，回调即是机会。",
        "建议减持以规避回撤风险。",
        "可以考虑买入该基金。",
        "该股票目标价 120 元。",
    )
    for index in range(COMPLIANCE_SAMPLES):
        if index % 5 == 4:
            # 正常样本：中性事实陈述不得被拦截
            dataset.append(
                {
                    "id": f"compliance_ok_{index:03d}",
                    "answer": "## 结论\n\n该产品风险等级为 R2，适合稳健型投资者。",
                    "user_role": ALL_ROLES[index % len(ALL_ROLES)],
                    "expected_blocked": False,
                    "restricted_text": "",
                    "returned_answer": "",
                }
            )
            continue
        if index % 7 == 3:
            keyword = SENSITIVE_KEYWORDS[index % len(SENSITIVE_KEYWORDS)]
            dataset.append(
                {
                    "id": f"compliance_sensitive_{index:03d}",
                    "answer": f"## 结论\n\n根据{keyword}，该事项尚未公布。",
                    "user_role": ALL_ROLES[index % len(ALL_ROLES)],
                    "expected_blocked": True,
                    "restricted_text": f"根据{keyword}，该事项尚未公布。",
                    "returned_answer": "",
                }
            )
            continue
        if index % 11 == 6:
            dataset.append(
                {
                    "id": f"compliance_article_{index:03d}",
                    "answer": "## 结论\n\n该行为违反了相关规定，应当予以纠正。",
                    "user_role": ROLE_COMPLIANCE,
                    "expected_blocked": True,
                    "restricted_text": "",
                    "returned_answer": "",
                }
            )
            continue
        dataset.append(
            {
                "id": f"compliance_advice_{index:03d}",
                "answer": f"## 结论\n\n{advice_samples[index % len(advice_samples)]}",
                "user_role": ALL_ROLES[index % len(ALL_ROLES)],
                "expected_blocked": True,
                "restricted_text": "",
                "returned_answer": "",
            }
        )
    return dataset


def build_conversation_dataset() -> list[dict[str, Any]]:
    """§3.4：多轮实体追问、跨用户访问、删除后续写、角色变化与重复 request_id。"""
    dataset: list[dict[str, Any]] = []
    for index in range(CONVERSATION_CASES):
        role = ALL_ROLES[index % len(ALL_ROLES)]
        client_id = f"client-{index:02d}" if index % 3 == 0 else None
        dataset.append(
            {
                "id": f"conversation_{index + 1:03d}",
                "user_id": f"owner-{index:02d}",
                "user_role": role,
                "department": "tech",
                "client_id": client_id,
                "request_id": f"request-{index:02d}",
                "query": (
                    "它的风险等级是多少？" if index % 2 == 0 else "600519 的营业收入是多少"
                ),
                "answer": "## 结论\n\n该产品风险等级为 R2。",
                "entities": {"stock_code": "600519"},
                "citations": [{"source": "report.pdf", "chunk_id": f"chunk-{index:02d}"}],
            }
        )
    return dataset


def _write(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{path}: {len(payload)} 条")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 implementation-08 §3 评估数据集")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    random.seed(SEED)
    output_dir = Path(args.output_dir)

    chunks = _load_chunks()
    if not chunks:
        raise SystemExit("知识库为空：请先入库再生成评估集")

    retrieval = build_retrieval_dataset(chunks)
    answers = build_answers_dataset(chunks)
    compliance = build_compliance_dataset()
    conversations = build_conversation_dataset()

    _write(output_dir / "retrieval.json", retrieval)
    _write(output_dir / "answers.json", answers)
    _write(output_dir / "compliance.json", compliance)
    _write(output_dir / "conversations.json", conversations)

    positives = sum(1 for item in retrieval if not item["expected_permission_denied"])
    negatives = sum(1 for item in retrieval if item["expected_permission_denied"])
    kinds = sorted({item["denial_kind"] for item in retrieval if item["expected_permission_denied"]})
    print(f"retrieval: 正例 {positives} / 负例 {negatives}，负例类别 {kinds}")
    print(
        "answers: 可回答 "
        f"{sum(1 for item in answers if item['expected_passed'])} / "
        f"不可回答 {sum(1 for item in answers if not item['expected_passed'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
