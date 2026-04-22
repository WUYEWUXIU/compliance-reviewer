"""Validation tests for the confidence calculator.

Covers the three mandated scenarios:
1. High rerank + high agreement → high confidence (>0.7)
2. Low rerank + compliance conclusion → low confidence (<0.4) + warning
3. Negation + violation conclusion → confidence attenuation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_review.confidence import (
    ConfidenceCalculator,
    get_action_advice,
    _jaccard_similarity,
    _bag_of_words_vectors,
    _cosine_distance,
)
from src.retrieval.hybrid_search import HybridSearchResult, RerankResult
from src.retrieval.query_rewriter import RewriteRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rerank_result(chunk_id: str, score: float, text: str = "") -> RerankResult:
    return RerankResult(chunk_id=chunk_id, score=score, text=text)


def make_search_result(scores: list[float], texts: list[str] | None = None) -> HybridSearchResult:
    texts = texts or [f"chunk_{i}_text" for i in range(len(scores))]
    top = [
        make_rerank_result(f"c{i}", s, t)
        for i, (s, t) in enumerate(zip(scores, texts))
    ]
    return HybridSearchResult(top_chunks=top, reference_chunks=[])


# ---------------------------------------------------------------------------
# Scenario 1: High rerank + multi-query agreement → high confidence (>0.7)
# ---------------------------------------------------------------------------

def test_high_confidence() -> bool:
    print("\n[Scenario 1] High rerank + high agreement → high confidence")
    calc = ConfidenceCalculator()

    # Simulate strong rerank scores (all > 0.8)
    scores = [0.92, 0.88, 0.85, 0.81, 0.79]
    search_result = make_search_result(scores)

    # Multiple rewrite requests to trigger agreement path
    rewrite_requests = [
        RewriteRequest(violation_type_id="V01", query_text="承诺保本保息", keywords=["保本"]),
        RewriteRequest(violation_type_id="V02", query_text="承诺确定收益", keywords=["年化"]),
        RewriteRequest(violation_type_id="V03", query_text="使用绝对化用语", keywords=["最优"]),
    ]

    marketing_text = "本产品保本保息，年化收益最优"
    result = calc.calculate(
        marketing_text=marketing_text,
        rewrite_requests=rewrite_requests,
        search_result=search_result,
        llm_compliant="no",
    )

    print(f"  Score: {result.confidence_score}")
    print(f"  Breakdown: {result.confidence_breakdown}")
    print(f"  Warning: '{result.warning}'")
    print(f"  Action: {get_action_advice(result.confidence_score)}")

    ok = result.confidence_score > 0.7
    print(f"  PASS={ok} (expected >0.7)")
    return ok


# ---------------------------------------------------------------------------
# Scenario 2: Low rerank + compliance=yes → low confidence (<=0.4) + warning
# ---------------------------------------------------------------------------

def test_low_confidence_compliance() -> bool:
    print("\n[Scenario 2] Low rerank + compliant=yes → low confidence + warning")
    calc = ConfidenceCalculator()

    # All scores below RERANK_THRESHOLD (0.3)
    scores = [0.25, 0.22, 0.20, 0.18, 0.15]
    search_result = make_search_result(scores)

    rewrite_requests = [
        RewriteRequest(violation_type_id="V00", query_text="通用合规要求", keywords=[]),
    ]

    marketing_text = "这是一段普通文案"
    result = calc.calculate(
        marketing_text=marketing_text,
        rewrite_requests=rewrite_requests,
        search_result=search_result,
        llm_compliant="yes",
    )

    print(f"  Score: {result.confidence_score}")
    print(f"  Breakdown: {result.confidence_breakdown}")
    print(f"  Warning: '{result.warning}'")
    print(f"  Action: {get_action_advice(result.confidence_score)}")

    ok_score = result.confidence_score <= 0.4
    ok_warning = result.warning != ""
    ok = ok_score and ok_warning
    print(f"  PASS={ok} (expected score ≤0.4 and warning non-empty)")
    return ok


# ---------------------------------------------------------------------------
# Scenario 3: Negation + violation conclusion → confidence attenuation
# ---------------------------------------------------------------------------

def test_negation_penalty() -> bool:
    print("\n[Scenario 3] Negation + compliant=no → confidence attenuated by 0.5")
    calc = ConfidenceCalculator()

    # Scores chosen so that raw weighted score is ~0.65.
    # After 0.5x penalty it should drop to ~0.325 (< 0.4).
    # rerank = 0.7*0.55 + 0.3*0.60 = 0.565
    # coverage = 1.0 (all > 0.3)
    # agreement = 1.0 (single query)
    # diversity = 0.8 (default for identical dummy texts)
    # raw = 0.565*0.4 + 1.0*0.2 + 1.0*0.2 + 0.8*0.2 = 0.226 + 0.2 + 0.2 + 0.16 = 0.786
    # Wait, that is too high. Let's use lower scores.
    # rerank = 0.7*0.40 + 0.3*0.45 = 0.415
    # raw = 0.415*0.4 + 1.0*0.2 + 1.0*0.2 + 0.8*0.2 = 0.166 + 0.2 + 0.2 + 0.16 = 0.726
    # penalty = 0.363 < 0.4  ✓
    scores = [0.40, 0.38, 0.36, 0.34, 0.32]
    search_result = make_search_result(scores)

    rewrite_requests = [
        RewriteRequest(violation_type_id="V01", query_text="承诺保本保息", keywords=["保本"]),
    ]

    # Text contains negation pattern "不保证"
    marketing_text = "本产品不保证收益"
    result = calc.calculate(
        marketing_text=marketing_text,
        rewrite_requests=rewrite_requests,
        search_result=search_result,
        llm_compliant="no",
    )

    print(f"  Score: {result.confidence_score}")
    print(f"  Breakdown: {result.confidence_breakdown}")
    print(f"  Warning: '{result.warning}'")
    print(f"  Action: {get_action_advice(result.confidence_score)}")

    ok = result.confidence_score < 0.4
    print(f"  PASS={ok} (expected <0.4 after 0.5x negation penalty)")
    return ok


# ---------------------------------------------------------------------------
# Unit-level helper tests
# ---------------------------------------------------------------------------

def test_jaccard() -> bool:
    print("\n[Unit] Jaccard similarity")
    ok1 = _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    ok2 = _jaccard_similarity({"a", "b"}, {"b", "c"}) == 1.0 / 3.0
    ok3 = _jaccard_similarity(set(), set()) == 1.0
    ok = ok1 and ok2 and ok3
    print(f"  PASS={ok}")
    return ok


def test_bow_vectors() -> bool:
    print("\n[Unit] Bag-of-words vectors")
    vecs = _bag_of_words_vectors(["hello world", "hello"])
    ok = vecs is not None and vecs.shape == (2, 2)
    print(f"  PASS={ok} (shape={vecs.shape if vecs is not None else None})")
    return ok


def test_cosine_distance() -> bool:
    print("\n[Unit] Cosine distance")
    import numpy as np
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    d = _cosine_distance(a, b)
    ok = abs(d - 1.0) < 1e-6
    print(f"  PASS={ok} (distance={d})")
    return ok


def test_action_advice() -> bool:
    print("\n[Unit] Action advice mapping")
    ok = True
    ok = ok and "可直接采用" in get_action_advice(0.85)
    ok = ok and "人工复核" in get_action_advice(0.65)
    ok = ok and "补充检索" in get_action_advice(0.45)
    ok = ok and "必须人工介入" in get_action_advice(0.25)
    print(f"  PASS={ok}")
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Confidence Calculator Validation")
    print("=" * 60)

    results = []
    results.append(("scenario_high_confidence", test_high_confidence()))
    results.append(("scenario_low_compliance", test_low_confidence_compliance()))
    results.append(("scenario_negation_penalty", test_negation_penalty()))
    results.append(("unit_jaccard", test_jaccard()))
    results.append(("unit_bow_vectors", test_bow_vectors()))
    results.append(("unit_cosine_distance", test_cosine_distance()))
    results.append(("unit_action_advice", test_action_advice()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
