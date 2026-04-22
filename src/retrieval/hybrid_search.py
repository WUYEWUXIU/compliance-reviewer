"""Hybrid search with RRF fusion, citation graph expansion, and reranking."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from src.config.settings import (
    BAILIAN_API_KEY,
    BAILIAN_RERANK_MODEL,
    CHUNKS_DIR,
    INDEXES_DIR,
    MAX_RETRIES,
    RERANK_THRESHOLD,
    RERANK_TIMEOUT,
    RRF_K,
    TOP_K_BM25,
    TOP_K_RERANK,
    TOP_K_VECTOR,
)
from src.indexing.dense_index import DenseIndex
from src.indexing.sparse_index import SparseIndex
from src.retrieval.query_rewriter import QueryRewriter, RewriteRequest

logger = logging.getLogger(__name__)

BAILIAN_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


# ---------------------------------------------------------------------------
# Citation graph
# ---------------------------------------------------------------------------

class CitationGraph:
    """One-hop citation graph backed by a simple adjacency dict.

    The citations file contains ``[src_chunk_id, dst_article_id]`` pairs.
    We resolve ``dst_article_id`` to chunk ids within the *same document* as
    ``src_chunk_id``.
    """

    def __init__(self, citations_path: Path | None = None) -> None:
        self.citations_path = citations_path or (CHUNKS_DIR / "citations.json")
        self._graph: Dict[str, List[str]] = {}
        self._build()

    def _build(self) -> None:
        """Load citations and build adjacency list."""
        if not self.citations_path.exists():
            logger.warning("Citation file not found: %s", self.citations_path)
            return

        with open(self.citations_path, "r", encoding="utf-8") as f:
            raw_citations = json.load(f)

        # Load chunks to resolve article_id -> chunk_id(s) per document
        chunks_path = CHUNKS_DIR / "chunks.json"
        if not chunks_path.exists():
            logger.warning("Chunks file not found: %s", chunks_path)
            return

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Group chunk_ids by (doc_name, article_id)
        doc_article_chunks: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for chunk in chunks:
            parts = chunk["chunk_id"].split("_")
            doc_name = parts[0]
            article_id = parts[1]
            doc_article_chunks[(doc_name, article_id)].append(chunk["chunk_id"])

        graph: Dict[str, List[str]] = defaultdict(list)
        for src_chunk_id, dst_article_id in raw_citations:
            parts = src_chunk_id.split("_")
            doc_name = parts[0]
            key = (doc_name, str(dst_article_id))
            if key in doc_article_chunks:
                graph[src_chunk_id].extend(doc_article_chunks[key])

        self._graph = dict(graph)
        logger.info("Citation graph built: %d edges", len(self._graph))

    def get_neighbors(self, chunk_id: str) -> List[str]:
        """Return one-hop neighbor chunk ids for *chunk_id*."""
        return self._graph.get(chunk_id, [])


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankResult:
    chunk_id: str
    score: float
    text: str


class Reranker:
    """Bailian gte-rerank wrapper with RRF fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or BAILIAN_API_KEY
        self.model = model or BAILIAN_RERANK_MODEL

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        chunk_text_map: Dict[str, str],
        top_k: int = TOP_K_RERANK,
    ) -> List[RerankResult]:
        """Rerank candidates.

        Args:
            query: Original marketing text.
            candidates: List of (chunk_id, rrf_score) sorted by RRF descending.
            chunk_text_map: Mapping from chunk_id to article_text.
            top_k: Number of results to return.

        Returns:
            Sorted list of RerankResult by rerank score descending.
        """
        if not candidates:
            return []

        if not self.api_key:
            logger.warning("BAILIAN_API_KEY not set; falling back to RRF scores.")
            return [
                RerankResult(
                    chunk_id=cid,
                    score=round(score, 6),
                    text=chunk_text_map.get(cid, ""),
                )
                for cid, score in candidates[:top_k]
            ]

        docs = []
        valid_candidates: List[Tuple[str, float]] = []
        for cid, rrf_score in candidates:
            text = chunk_text_map.get(cid, "")
            if not text:
                continue
            docs.append(text)
            valid_candidates.append((cid, rrf_score))

        if not docs:
            return []

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": docs,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    BAILIAN_RERANK_URL,
                    headers=headers,
                    json=payload,
                    timeout=RERANK_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()

                # Bailian rerank response format:
                # output: { results: [ { index: int, relevance_score: float } ] }
                results_data = result.get("output", {}).get("results", [])
                indexed_scores: Dict[int, float] = {
                    r["index"]: r["relevance_score"] for r in results_data
                }

                reranked: List[RerankResult] = []
                for idx, (cid, _) in enumerate(valid_candidates):
                    score = indexed_scores.get(idx, 0.0)
                    if score < RERANK_THRESHOLD:
                        continue
                    reranked.append(
                        RerankResult(
                            chunk_id=cid,
                            score=round(score, 6),
                            text=chunk_text_map.get(cid, ""),
                        )
                    )

                reranked.sort(key=lambda x: x.score, reverse=True)

                # If threshold filters everything, fallback to RRF ordering
                # to avoid returning empty results when retrieval is viable.
                if not reranked:
                    logger.warning(
                        "Rerank scores all below threshold %.2f; falling back to RRF.",
                        RERANK_THRESHOLD,
                    )
                    return [
                        RerankResult(
                            chunk_id=cid,
                            score=round(score, 6),
                            text=chunk_text_map.get(cid, ""),
                        )
                        for cid, score in valid_candidates[:top_k]
                    ]

                return reranked[:top_k]

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Bailian rerank API attempt %d/%d failed: %s",
                    attempt,
                    MAX_RETRIES,
                    exc,
                )

        logger.error(
            "Bailian rerank API failed after %d retries: %s. Falling back to RRF.",
            MAX_RETRIES,
            last_error,
        )
        return [
            RerankResult(
                chunk_id=cid,
                score=round(score, 6),
                text=chunk_text_map.get(cid, ""),
            )
            for cid, score in valid_candidates[:top_k]
        ]


# ---------------------------------------------------------------------------
# HybridSearch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridSearchResult:
    """Structured result from HybridSearch.search()."""

    top_chunks: List[RerankResult]
    reference_chunks: List[RerankResult]


class HybridSearch:
    """End-to-end hybrid retrieval pipeline.

    Pipeline:
        1. Query rewriting
        2. Per-query BM25 + dense retrieval
        3. Per-query RRF fusion
        4. Cross-query RRF score accumulation
        5. Global top-10 by RRF
        6. Conditional citation-graph expansion (RRF score > 0.5)
        7. Reranking (gte-rerank or fallback)
        8. Return top-5 + reference chunks
    """

    def __init__(self) -> None:
        self.dense_index = DenseIndex()
        self.sparse_index = SparseIndex()
        self.query_rewriter = QueryRewriter()
        self.citation_graph = CitationGraph()
        self.reranker = Reranker()
        self._chunk_text_map: Dict[str, str] = {}
        self._load_chunk_texts()

    def _load_chunk_texts(self) -> None:
        """Load chunk texts into memory for reranker and result assembly."""
        chunks_path = CHUNKS_DIR / "chunks.json"
        if not chunks_path.exists():
            logger.warning("Chunks file not found: %s", chunks_path)
            return
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        self._chunk_text_map = {c["chunk_id"]: c["article_text"] for c in chunks}

    @staticmethod
    def _rrf_fuse(
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        k: int = RRF_K,
    ) -> Dict[str, float]:
        """Fuse BM25 and vector results via Reciprocal Rank Fusion.

        Args:
            bm25_results: List of (chunk_id, score) from BM25.
            vector_results: List of (chunk_id, score) from dense retrieval.
            k: RRF constant (default from settings).

        Returns:
            Mapping chunk_id -> accumulated RRF score.
        """
        scores: Dict[str, float] = defaultdict(float)

        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            scores[chunk_id] += 1.0 / (k + rank)

        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            scores[chunk_id] += 1.0 / (k + rank)

        return dict(scores)

    def search(self, marketing_text: str) -> HybridSearchResult:
        """Run the full hybrid search pipeline.

        Args:
            marketing_text: Original marketing copy.

        Returns:
            HybridSearchResult with top_chunks and reference_chunks.
        """
        if not marketing_text or not marketing_text.strip():
            return HybridSearchResult(top_chunks=[], reference_chunks=[])

        # 1. Query rewriting
        rewrite_requests: List[RewriteRequest] = self.query_rewriter.rewrite(
            marketing_text
        )
        if not rewrite_requests:
            rewrite_requests = [
                RewriteRequest(
                    violation_type_id="V00",
                    query_text=marketing_text.strip(),
                    keywords=[],
                )
            ]

        # 2 & 3. Per-query retrieval + RRF fusion
        global_rrf: Dict[str, float] = defaultdict(float)
        for req in rewrite_requests:
            query = req.query_text
            bm25_results = self.sparse_index.search(query, top_k=TOP_K_BM25)
            vector_results = self.dense_index.search(query, top_k=TOP_K_VECTOR)
            per_query_rrf = self._rrf_fuse(bm25_results, vector_results, k=RRF_K)
            for chunk_id, score in per_query_rrf.items():
                global_rrf[chunk_id] += score

        # 4. Sort by accumulated RRF score descending
        sorted_rrf = sorted(global_rrf.items(), key=lambda x: x[1], reverse=True)

        # 5. Top-10 by RRF
        rrf_top10 = sorted_rrf[:10]

        # 6. Conditional citation-graph expansion (RRF score > 0.5)
        reference_chunks: List[RerankResult] = []
        seen_ref_ids: set[str] = set()
        for chunk_id, rrf_score in rrf_top10:
            if rrf_score > 0.5:
                neighbors = self.citation_graph.get_neighbors(chunk_id)
                for neighbor_id in neighbors:
                    if neighbor_id not in seen_ref_ids:
                        seen_ref_ids.add(neighbor_id)
                        text = self._chunk_text_map.get(neighbor_id, "")
                        if text:
                            reference_chunks.append(
                                RerankResult(
                                    chunk_id=neighbor_id,
                                    score=round(rrf_score, 6),
                                    text=text,
                                )
                            )

        # 7. Rerank
        top_chunks = self.reranker.rerank(
            query=marketing_text,
            candidates=rrf_top10,
            chunk_text_map=self._chunk_text_map,
            top_k=TOP_K_RERANK,
        )

        return HybridSearchResult(
            top_chunks=top_chunks,
            reference_chunks=reference_chunks,
        )
