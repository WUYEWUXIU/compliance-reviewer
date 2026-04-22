"""Compliance review pipeline — main orchestrator for the RAG system.

Integrates hybrid retrieval, confidence scoring, and LLM review into a
single end-to-end flow with graceful error handling and fallback strategies.
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, List

from src.llm_review.confidence import ConfidenceCalculator, get_action_advice
from src.llm_review.output_parser import parse_llm_output, validate_output
from src.llm_review.reviewer import LLMReviewer, ReviewResult
from src.retrieval.hybrid_search import HybridSearch, HybridSearchResult, RerankResult
from src.retrieval.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CompliancePipeline
# ---------------------------------------------------------------------------


class CompliancePipeline:
    """End-to-end compliance review pipeline.

    Orchestrates:
        1. Hybrid search (BM25 + dense + RRF + rerank)
        2. Confidence calculation (two-pass: pre-LLM and post-LLM)
        3. LLM review (with mock fallback when API is unavailable)
        4. Result assembly and formatting
    """

    def __init__(self) -> None:
        self.hybrid_search = HybridSearch()
        self.query_rewriter = QueryRewriter()
        self.llm_reviewer = LLMReviewer()
        self.confidence_calculator = ConfidenceCalculator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(self, marketing_text: str) -> Dict[str, Any]:
        """Run the full compliance review pipeline.

        Args:
            marketing_text: The marketing copy to review.

        Returns:
            A structured JSON-compatible dict.  Never raises — all errors are
            captured and returned as a structured error response.
        """
        try:
            return self._run_pipeline(marketing_text)
        except Exception as exc:
            logger.error("Pipeline uncaught exception: %s", exc)
            logger.debug(traceback.format_exc())
            return self._build_error_response(
                marketing_text,
                f"Pipeline internal error: {exc}",
            )

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _run_pipeline(self, marketing_text: str) -> Dict[str, Any]:
        """Core pipeline logic (may raise)."""
        # 1. Hybrid search
        search_result = self._safe_search(marketing_text)

        # 2. Empty retrieval guard
        if not search_result.top_chunks:
            logger.warning("No relevant chunks retrieved for: %s", marketing_text[:50])
            return {
                "compliant": "unknown",
                "violations": [],
                "positive_compliance": [],
                "confidence": 0.2,
                "confidence_breakdown": {},
                "action_advice": get_action_advice(0.2),
                "warning": "未检索到相关条文",
                "top_chunks": [],
                "reference_chunks": [],
            }

        # 3. Query rewriting (needed for confidence calculation)
        rewrite_requests = self.query_rewriter.rewrite(marketing_text)

        # 4. First-pass confidence (llm_compliant="unknown" so negation_penalty=1.0)
        pre_confidence = self.confidence_calculator.calculate(
            marketing_text=marketing_text,
            rewrite_requests=rewrite_requests,
            search_result=search_result,
            llm_compliant="unknown",
        )

        # 5. LLM review (with pre-confidence score)
        review_result = self._safe_llm_review(
            marketing_text=marketing_text,
            search_result=search_result,
            confidence_score=pre_confidence.confidence_score,
        )

        # 6. Second-pass confidence using actual LLM conclusion
        post_confidence = self.confidence_calculator.calculate(
            marketing_text=marketing_text,
            rewrite_requests=rewrite_requests,
            search_result=search_result,
            llm_compliant=review_result.compliant,
        )

        # 7. Assemble final response
        return self._assemble_response(
            search_result=search_result,
            review_result=review_result,
            confidence_result=post_confidence,
        )

    # ------------------------------------------------------------------
    # Safe wrappers for sub-module calls
    # ------------------------------------------------------------------

    def _safe_search(self, marketing_text: str) -> HybridSearchResult:
        """Run hybrid search with exception handling.

        If search fails entirely, return an empty result so the pipeline
        can fall back to the "no chunks" response.
        """
        try:
            return self.hybrid_search.search(marketing_text)
        except Exception as exc:
            logger.error("Hybrid search failed: %s", exc)
            return HybridSearchResult(top_chunks=[], reference_chunks=[])

    def _safe_llm_review(
        self,
        marketing_text: str,
        search_result: HybridSearchResult,
        confidence_score: float,
    ) -> ReviewResult:
        """Run LLM review with exception handling.

        If the LLM call fails (network, timeout, parsing, etc.), fall back
        to a mock review and mark confidence as low.
        """
        try:
            return self.llm_reviewer.review(
                marketing_text=marketing_text,
                top_chunks=search_result.top_chunks,
                reference_chunks=search_result.reference_chunks,
                confidence_score=confidence_score,
            )
        except Exception as exc:
            logger.error("LLM review failed: %s", exc)
            # Fallback to mock review with low confidence
            return self.llm_reviewer._mock_review(
                marketing_text=marketing_text,
                top_chunks=search_result.top_chunks,
                reference_chunks=search_result.reference_chunks,
                confidence_score=min(confidence_score, 0.4),
            )

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_response(
        search_result: HybridSearchResult,
        review_result: ReviewResult,
        confidence_result: Any,
    ) -> Dict[str, Any]:
        """Merge sub-module outputs into the final JSON response."""
        top_chunks = _rerank_results_to_dicts(search_result.top_chunks)
        reference_chunks = _rerank_results_to_dicts(search_result.reference_chunks)

        response: Dict[str, Any] = {
            "compliant": review_result.compliant,
            "violations": review_result.violations,
            "positive_compliance": review_result.positive_compliance,
            "confidence": confidence_result.confidence_score,
            "confidence_breakdown": confidence_result.confidence_breakdown,
            "action_advice": get_action_advice(confidence_result.confidence_score),
            "top_chunks": top_chunks,
            "reference_chunks": reference_chunks,
        }

        if confidence_result.warning:
            response["warning"] = confidence_result.warning

        if review_result.used_mock:
            response["warning"] = response.get("warning", "") + " [LLM降级为mock审核]"

        return response

    @staticmethod
    def _build_error_response(marketing_text: str, reason: str) -> Dict[str, Any]:
        """Construct a structured error response when the pipeline crashes."""
        return {
            "compliant": "unknown",
            "violations": [],
            "positive_compliance": [],
            "confidence": 0.0,
            "confidence_breakdown": {},
            "action_advice": get_action_advice(0.0),
            "warning": reason,
            "top_chunks": [],
            "reference_chunks": [],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rerank_results_to_dicts(results: List[RerankResult]) -> List[Dict[str, Any]]:
    """Convert a list of RerankResult dataclasses to plain dicts."""
    return [
        {
            "chunk_id": r.chunk_id,
            "score": r.score,
            "text": r.text,
        }
        for r in results
    ]
