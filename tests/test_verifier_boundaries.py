"""数字验证与引用编号对齐的反例测试（issues.md 一.7）。"""

from src.schemas.constants import META_CHUNK_ID, META_SOURCE, RR_CONTENT, RR_METADATA, RR_SCORE
from src.schemas.typed_dicts import RetrievalResult
from src.utils.verifier import CitationExtractor, HallucinationDetector, NumberVerifier, SourceVerifier


def _result(content: str, source: str, chunk_id: str) -> RetrievalResult:
    return {
        RR_CONTENT: content,
        RR_METADATA: {META_SOURCE: source, META_CHUNK_ID: chunk_id, "title": "报告"},
        RR_SCORE: 0.9,
    }


class TestNumberVerifier:
    def test_substring_number_is_rejected(self):
        """证据是"收入 120 亿元"，答案"收入 20 亿元"必须被拦截。"""
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="收入 20 亿元。",
            retrieval_results=[_result("本公司收入 120 亿元。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is False

    def test_sign_is_bound(self):
        """证据是 -10%，答案写 10% 必须被拦截。"""
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="同比变化 10%。",
            retrieval_results=[_result("同比变化 -10%。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is False

    def test_negative_number_matched_explicitly(self):
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="同比变化 -10%。",
            retrieval_results=[_result("同比变化 -10%。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is True

    def test_citation_markers_are_not_business_numbers(self):
        """证据无业务数字，答案只有引用编号 [来源1] 不应触发数字校验失败。"""
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="该基金为货币基金 [来源1]。",
            retrieval_results=[_result("该基金为货币基金。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is True

    def test_list_markers_are_not_business_numbers(self):
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="1. 该基金为货币基金。\n2. 风险较低。",
            retrieval_results=[_result("该基金为货币基金，风险较低。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is True

    def test_boundary_years_not_confused(self):
        """答案数字不得命中证据中更长数字的子串（如 20 命中 2024）。"""
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="规模 20 亿元。",
            retrieval_results=[_result("2024 年规模 120 亿元。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is False


class TestNumberEquivalence:
    """ISSUE-13：数值等价（小数尾零、千分位）不得词面误杀；单位与精度差异仍拦截。"""

    def test_trailing_zeros_are_equivalent(self):
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="净值 1 元。",
            retrieval_results=[_result("净值 1.00 元。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is True

    def test_percent_trailing_zero_is_equivalent(self):
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="七日年化为 3.9%。",
            retrieval_results=[_result("七日年化收益率为 3.90%。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is True

    def test_percent_requires_percent_evidence(self):
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="七日年化为 3.9%。",
            retrieval_results=[_result("收益为 3.9 元。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is False

    def test_precision_mismatch_still_rejected(self):
        """0.85 ≠ 0.8513：数值等价不得放过精度差异。"""
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="净值为 0.85 元。",
            retrieval_results=[_result("净值为 0.8513 元。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is False

    def test_thousand_separator_is_equivalent(self):
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="募集规模 1234 亿元。",
            retrieval_results=[_result("募集规模 1,234 亿元。", "s", "c1")],
            tool_calls=[],
        )
        assert result["passed"] is True


class TestCitationAlignment:
    def test_citation_numbers_follow_prompt_order(self):
        """同一来源两个 chunk 去重后，后续来源编号不得前移（issues.md 一.7）。"""
        extractor = CitationExtractor()
        results = [
            _result("A1：贵州茅台评级增持。", "s", "a1"),
            _result("A1：贵州茅台评级增持。", "s", "a2"),  # 与 a1 相同 quote，会被去重
            _result("B：某产品风险等级 R2。", "s", "b1"),
        ]
        citations = extractor.extract(results, query="评级 风险")

        # 编号应基于 prompt 序号：a1 -> 1，b1 -> 3（不再因去重移位成 2）
        ids = [c.get("citation_id") for c in citations]
        assert ids == ["cite_001", "cite_003"]

    def test_source_verifier_uses_prompt_source_count(self):
        """引用编号有效性按 prompt 来源数判定，而不是去重后的引用列表长度。"""
        verifier = SourceVerifier()
        results = [
            _result("A1 内容。", "s", "a1"),
            _result("A1 内容。", "s", "a2"),
            _result("B 内容。", "s", "b1"),
        ]
        citations = CitationExtractor().extract(results, query="内容")
        # 答案引用 [来源3]：虽然去重后只有 2 条引用，但 prompt 有 3 个来源
        answer = "结论 [来源3]。"
        result = verifier.verify(answer, citations, results)
        assert result["passed"] is True

        # 超出来源数的编号仍然非法
        bad = verifier.verify("结论 [来源4]。", citations, results)
        assert bad["passed"] is False


class TestFailedToolOutputNotEvidence:
    """P0-2: 失败工具输出（success=False）不得作为验证证据。

    工具改为抛异常后，错误文本以 status="error" 进入 tool_calls 且
    success=False；验证器必须继续排除这类文本，防止错误提示被当成数据。
    """

    def test_number_verifier_ignores_failed_tool_output(self):
        verifier = NumberVerifier()
        result = verifier.verify(
            answer="据工具查询，目标价 88.88 元。",
            retrieval_results=[],
            tool_calls=[{
                "tool": "sql_query",
                "output": "查询错误: 目标价 88.88 元",
                "success": False,
            }],
        )
        assert result["passed"] is False
        assert result["numbers_found"] == 0

    def test_hallucination_detector_ignores_failed_tool_output(self):
        detector = HallucinationDetector()
        error_text = "查询错误: 目标价为 88.88 元"

        failed = detector.detect(
            "目标价为 88.88 元",
            [],
            [{"tool": "sql_query", "output": error_text, "success": False}],
        )
        assert failed["passed"] is False

        # 对照：同一输出仅在 success=True 时才算证据
        succeeded = detector.detect(
            "目标价为 88.88 元",
            [],
            [{"tool": "sql_query", "output": error_text, "success": True}],
        )
        assert succeeded["passed"] is True


class TestHallucinationNormalization:
    """ISSUE-13：日期写法与虚词差异导致的同义改写不得判为幻觉。"""

    def test_chinese_date_matches_iso_evidence(self):
        detector = HallucinationDetector()
        result = detector.detect(
            "它的成立日期是2024年4月26日。",
            [_result("基金成立于2024-04-26，规模保持稳定。", "s", "c1")],
            [],
        )
        assert result["passed"] is True

    def test_function_word_heavy_reformulation_is_supported(self):
        detector = HallucinationDetector()
        result = detector.detect(
            "本产品的申购费率已下调至0.15%。",
            [_result("申购费率调整为0.15%（费率优惠期内有效）。", "s", "c1")],
            [],
        )
        assert result["passed"] is True

    def test_unsupported_claim_without_numbers_still_rejected(self):
        """归一化不得放过真正无证据的断言（无数字可查时的兜底线）。"""
        detector = HallucinationDetector()
        result = detector.detect(
            "该基金经理擅长量化套利策略。",
            [_result("基金成立于2024-04-26，规模保持稳定。", "s", "c1")],
            [],
        )
        assert result["passed"] is False


class TestFailureKindClassification:
    """ISSUE-13：验证失败须区分"格式不符"（可局部修复）与"事实缺失"（需重新取证）。"""

    def test_citation_format_only_failure_is_format(self):
        from src.utils.verifier import ComprehensiveVerifier

        verifier = ComprehensiveVerifier()
        results = [_result("基金规模为120亿元。", "s", "c1")]
        citations = CitationExtractor().extract(results, query="规模")

        result = verifier.verify("规模为120亿元 [来源2]。", citations, results, [])

        assert result["passed"] is False
        assert result["failure_kind"] == "format"

    def test_fabricated_number_is_facts(self):
        from src.utils.verifier import ComprehensiveVerifier

        verifier = ComprehensiveVerifier()
        results = [_result("基金规模为120亿元。", "s", "c1")]
        citations = CitationExtractor().extract(results, query="规模")

        result = verifier.verify("规模为999亿元 [来源1]。", citations, results, [])

        assert result["passed"] is False
        assert result["failure_kind"] == "facts"
