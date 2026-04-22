"""Confidence calculator for compliance review RAG system.

Computes a composite confidence score (0-1) based on four retrieval-quality
signals, with special handling for negation and low-evidence compliance
conclusions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import jieba
import numpy as np

from src.config.settings import (
    RERANK_THRESHOLD,
    W_AGREEMENT,
    W_COVERAGE,
    W_DIVERSITY,
    W_RERANK,
)
from src.config.violation_types import NEGATION_PATTERNS
from src.indexing.dense_index import DenseIndex
from src.indexing.sparse_index import SparseIndex
from src.retrieval.hybrid_search import HybridSearchResult, RerankResult
from src.retrieval.query_rewriter import RewriteRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceResult:
    """Structured confidence output."""

    confidence_score: float
    confidence_breakdown: Dict[str, float]
    warning: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConfidenceCalculator:
    """Compute retrieval-based confidence for a compliance review decision."""

    def __init__(self) -> None:
        # Lazy-initialised indexes for multi-query agreement computation.
        self._dense_index: DenseIndex | None = None
        self._sparse_index: SparseIndex | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def calculate(
        self,
        marketing_text: str,
        rewrite_requests: List[RewriteRequest],
        search_result: HybridSearchResult,
        llm_compliant: str,
    ) -> ConfidenceResult:
        """Calculate confidence score for a compliance review.

        Args:
            marketing_text: Original marketing copy.
            rewrite_requests: Output from QueryRewriter.rewrite().
            search_result: Output from HybridSearch.search().
            llm_compliant: LLM conclusion — "yes", "no", or "unknown".

        Returns:
            ConfidenceResult with score, breakdown, and optional warning.
        """
        top5 = search_result.top_chunks[:5]

        # 1. Four component scores
        rerank_score = self._compute_rerank_score(top5)
        chunk_coverage = self._compute_chunk_coverage(top5)
        multi_query_agreement = self._compute_multi_query_agreement(
            rewrite_requests
        )
        semantic_diversity = self._compute_semantic_diversity(top5)

        # 2. Weighted combination
        raw_score = (
            W_RERANK * rerank_score
            + W_COVERAGE * chunk_coverage
            + W_AGREEMENT * multi_query_agreement
            + W_DIVERSITY * semantic_diversity
        )

        # 3. Negation penalty (§5.5.3)
        negation_penalty = self._apply_negation_penalty(
            marketing_text, llm_compliant
        )
        score = raw_score * negation_penalty

        # 4. Low-evidence compliance cap (§5.5.4)
        warning = ""
        if llm_compliant.lower() == "yes":
            all_below_threshold = all(
                r.score < RERANK_THRESHOLD for r in top5
            )
            if all_below_threshold:
                score = min(score, 0.4)
                warning = (
                    "因未检索到高置信度相关条文而判定合规，结论可信度低"
                )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        breakdown = {
            "rerank_score": round(rerank_score, 4),
            "chunk_coverage": round(chunk_coverage, 4),
            "multi_query_agreement": round(multi_query_agreement, 4),
            "semantic_diversity": round(semantic_diversity, 4),
            "negation_penalty": round(negation_penalty, 4),
        }

        return ConfidenceResult(
            confidence_score=round(score, 4),
            confidence_breakdown=breakdown,
            warning=warning,
        )

    # ------------------------------------------------------------------
    # Component: rerank_score
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rerank_score(top5: List[RerankResult]) -> float:
        """Weighted combination of mean and max rerank scores, normalised to 0-1.

        Formula: 0.7 * mean(top5_scores) + 0.3 * max(top5_scores)
        """
        if not top5:
            return 0.0

        scores = [r.score for r in top5]
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)

        raw = 0.7 * mean_score + 0.3 * max_score
        # Clamp to [0, 1] in case of out-of-range fallback scores
        return max(0.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Component: chunk_coverage
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_chunk_coverage(top5: List[RerankResult]) -> float:
        """Fraction of top-5 chunks whose rerank score exceeds threshold."""
        if not top5:
            return 0.0

        above = sum(1 for r in top5 if r.score > RERANK_THRESHOLD)
        return above / 5.0

    # ------------------------------------------------------------------
    # Component: multi_query_agreement
    # ------------------------------------------------------------------

    def _compute_multi_query_agreement(
        self, rewrite_requests: List[RewriteRequest]
    ) -> float:
        """Average Jaccard similarity across per-query dense/sparse top-5 sets.

        For each rewrite request we run a lightweight dense and sparse search
        (top-5 each) and collect the chunk_id sets.  We then compute the
        pairwise Jaccard similarity among all sets and return the mean.
        """
        if len(rewrite_requests) < 2:
            # With a single query there is nothing to agree/disagree with.
            return 1.0

        # Lazy-init indexes
        if self._dense_index is None:
            self._dense_index = DenseIndex()
            try:
                self._dense_index.load_index()
            except FileNotFoundError:
                logger.warning("Dense index not found; agreement = 1.0")
                return 1.0

        if self._sparse_index is None:
            self._sparse_index = SparseIndex()
            try:
                self._sparse_index.load_index()
            except FileNotFoundError:
                logger.warning("Sparse index not found; agreement = 1.0")
                return 1.0

        sets: List[Set[str]] = []
        for req in rewrite_requests:
            query = req.query_text
            try:
                dense_top5 = self._dense_index.search(query, top_k=5)
                sparse_top5 = self._sparse_index.search(query, top_k=5)
            except Exception as exc:
                logger.warning("Index search failed for query '%s': %s", query, exc)
                continue

            chunk_ids = {cid for cid, _ in dense_top5} | {
                cid for cid, _ in sparse_top5
            }
            if chunk_ids:
                sets.append(chunk_ids)

        if len(sets) < 2:
            return 1.0

        similarities: List[float] = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                sim = _jaccard_similarity(sets[i], sets[j])
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 1.0

    # ------------------------------------------------------------------
    # Component: semantic_diversity
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_semantic_diversity(top5: List[RerankResult]) -> float:
        """Inverse of average pairwise cosine distance among top-5 chunks.

        Uses a simple bag-of-words vector (jieba tokenisation) as a proxy
        when no embedding API is available.  Higher diversity => lower score
        because focused results should increase confidence.
        """
        if len(top5) < 2:
            return 1.0

        texts = [r.text for r in top5 if r.text]
        if len(texts) < 2:
            return 1.0

        vectors = _bag_of_words_vectors(texts)
        if vectors is None:
            return 1.0

        # Pairwise cosine distances
        distances: List[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                dist = _cosine_distance(vectors[i], vectors[j])
                distances.append(dist)

        if not distances:
            return 1.0

        avg_distance = sum(distances) / len(distances)
        if avg_distance == 0.0:
            return 1.0  # perfectly identical => zero diversity => max confidence

        # Inverse, scaled so that typical distances map into [0, 1]
        # Empirically bag-of-words cosine distance for legal chunks is ~0.3-0.8
        raw = 1.0 / (1.0 + avg_distance)
        return max(0.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Special handling: negation penalty (§5.5.3)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_negation_penalty(text: str, llm_compliant: str) -> float:
        """Return 0.5 if text contains negation and LLM says non-compliant."""
        if llm_compliant.lower() != "no":
            return 1.0

        for pat in NEGATION_PATTERNS:
            if pat in text:
                return 0.5
        return 1.0


# ---------------------------------------------------------------------------
# Action advice (§5.6.5)
# ---------------------------------------------------------------------------


def get_action_advice(confidence: float) -> str:
    """Map confidence score to recommended action."""
    if confidence >= 0.8:
        return "置信度高，可直接采用审核结论"
    if confidence >= 0.6:
        return "置信度中等，建议人工复核后采用"
    if confidence >= 0.4:
        return "置信度偏低，建议补充检索或交由专家审核"
    return "置信度低，必须人工介入审核，不可自动采信"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _bag_of_words_vectors(texts: List[str]) -> np.ndarray | None:
    """Build a simple bag-of-words matrix for a list of texts.

    Returns:
        Array of shape (n_texts, vocab_size) with L2-normalised rows,
        or None if vocabulary is empty.
    """
    # Tokenise
    tokenised = [set(jieba.cut(t)) for t in texts]
    # Build vocabulary
    vocab: Dict[str, int] = {}
    for tokens in tokenised:
        for tok in tokens:
            if tok.strip() and tok.strip() not in vocab:
                vocab[tok.strip()] = len(vocab)

    if not vocab:
        return None

    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, tokens in enumerate(tokenised):
        for tok in tokens:
            idx = vocab.get(tok.strip())
            if idx is not None:
                mat[i, idx] += 1.0

    # L2 normalise rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalised vectors."""
    # a, b are already normalised => cosine similarity = dot(a, b)
    similarity = float(np.dot(a, b))
    # Clamp to [-1, 1] to avoid numerical drift
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity
