"""Compliance review pipeline — main orchestrator for the RAG system."""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List

from src.llm_review.output_parser import parse_llm_output, validate_output
from src.llm_review.reviewer import LLMReviewer, ReviewResult
from src.retrieval.hybrid_search import HybridSearch, HybridSearchResult, RerankResult
from src.retrieval.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class CompliancePipeline:
    """End-to-end compliance review pipeline."""

    def __init__(self) -> None:
        self.hybrid_search = HybridSearch()
        self.query_rewriter = QueryRewriter()
        self.llm_reviewer = LLMReviewer()

    def review(self, marketing_text: str) -> Dict[str, Any]:
        """Run the full compliance review pipeline."""
        try:
            return self._run_pipeline(marketing_text)
        except Exception as exc:
            logger.error("Pipeline uncaught exception: %s", exc)
            logger.debug(traceback.format_exc())
            return self._build_error_response(f"Pipeline internal error: {exc}")

    def _run_pipeline(self, marketing_text: str) -> Dict[str, Any]:
        search_result = self._safe_search(marketing_text)

        if not search_result.top_chunks:
            logger.warning("No relevant chunks retrieved for: %s", marketing_text[:50])
            return {
                "compliant": "unknown",
                "violations": [],
                "positive_compliance": [],
                "warning": "未检索到相关条文",
                "top_chunks": [],
                "reference_chunks": [],
            }

        review_result = self._safe_llm_review(marketing_text, search_result)
        return self._assemble_response(search_result, review_result)

    def _safe_search(self, marketing_text: str) -> HybridSearchResult:
        try:
            return self.hybrid_search.search(marketing_text)
        except Exception as exc:
            logger.error("Hybrid search failed: %s", exc)
            return HybridSearchResult(top_chunks=[], reference_chunks=[])

    def _safe_llm_review(
        self,
        marketing_text: str,
        search_result: HybridSearchResult,
    ) -> ReviewResult:
        try:
            return self.llm_reviewer.review(
                marketing_text=marketing_text,
                top_chunks=search_result.top_chunks,
                reference_chunks=search_result.reference_chunks,
            )
        except Exception as exc:
            logger.error("LLM review failed: %s", exc)
            return self.llm_reviewer._mock_review(
                marketing_text=marketing_text,
                top_chunks=search_result.top_chunks,
                reference_chunks=search_result.reference_chunks,
            )

    @staticmethod
    def _assemble_response(
        search_result: HybridSearchResult,
        review_result: ReviewResult,
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "compliant": review_result.compliant,
            "violations": review_result.violations,
            "positive_compliance": review_result.positive_compliance,
            "top_chunks": _rerank_results_to_dicts(search_result.top_chunks),
            "reference_chunks": _rerank_results_to_dicts(search_result.reference_chunks),
        }

        if review_result.used_mock:
            response["warning"] = "[LLM降级为mock审核]"

        return response

    @staticmethod
    def _build_error_response(reason: str) -> Dict[str, Any]:
        return {
            "compliant": "unknown",
            "violations": [],
            "positive_compliance": [],
            "warning": reason,
            "top_chunks": [],
            "reference_chunks": [],
        }


def _rerank_results_to_dicts(results: List[RerankResult]) -> List[Dict[str, Any]]:
    return [
        {"chunk_id": r.chunk_id, "score": r.score, "text": r.text}
        for r in results
    ]
