"""BM25 sparse index with jieba Chinese tokenization."""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi

from src.config.settings import CHUNKS_DIR, INDEXES_DIR

logger = logging.getLogger(__name__)

# Custom regulatory terms to preserve as single tokens.
# Only terms that actually appear in the corpus are registered,
# otherwise query tokenization produces non-matching tokens.
_CUSTOM_TERMS_CANDIDATES = [
    "误导性陈述",
    "保本保息",
    "投保人",
    "被保险人",
    "受益人",
    "犹豫期",
    "退保",
    "转保",
    "保险代理人",
]

# Chinese stopwords for regulatory/legal text
STOPWORDS = {}


def _load_all_text() -> str:
    """Load all chunk texts concatenated for term frequency check."""
    chunks = _get_chunks()
    return " ".join(c["article_text"] for c in chunks)


def _register_custom_terms() -> List[str]:
    """Register custom terms with jieba, but only those present in the corpus.

    Returns:
        List of actually registered terms.
    """
    try:
        all_text = _load_all_text()
    except Exception:
        all_text = ""

    registered: List[str] = []
    for term in _CUSTOM_TERMS_CANDIDATES:
        if term in all_text:
            jieba.add_word(term, freq=1000)
            registered.append(term)
    return registered


# Register at module import time
_REGISTERED_TERMS = _register_custom_terms()
logger.debug("Registered %d custom jieba terms: %s",
             len(_REGISTERED_TERMS), _REGISTERED_TERMS)


def _get_chunks() -> List[dict]:
    """Load chunks from the canonical chunks file."""
    chunks_path = CHUNKS_DIR / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> List[str]:
    """Tokenize Chinese text with jieba and filter stopwords.

    Args:
        text: Raw text string.

    Returns:
        List of tokens after stopword filtering.
    """
    tokens = list(jieba.cut(text))
    filtered = [t.strip() for t in tokens if t.strip()
                and t.strip() not in STOPWORDS]
    return filtered


class SparseIndex:
    """BM25 sparse index with jieba tokenization."""

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.index_path = INDEXES_DIR / "bm25.pkl"
        self.mapping_path = INDEXES_DIR / "bm25_mapping.json"

    def build_index(self) -> None:
        """Build the BM25 index from chunks and persist to disk."""
        chunks = _get_chunks()
        if not chunks:
            raise ValueError("No chunks found to index.")

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.tokenized_docs = [_tokenize(c["article_text"]) for c in chunks]

        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Persist
        INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "tokenized_docs": self.tokenized_docs,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "Sparse index built: %d docs, saved to %s",
            len(self.chunk_ids),
            self.index_path,
        )

    def load_index(self) -> None:
        """Load an existing BM25 index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Mapping file not found: {self.mapping_path}"
            )

        with open(self.index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            self.chunk_ids = mapping["chunk_ids"]
            self.tokenized_docs = mapping["tokenized_docs"]

        logger.info(
            "Sparse index loaded: %d docs from %s",
            len(self.chunk_ids),
            self.index_path,
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the BM25 index for the most relevant chunks.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending.
        """
        if self.bm25 is None:
            self.load_index()

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices by score
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: List[Tuple[str, float]] = []
        for idx in top_indices:
            results.append((self.chunk_ids[idx], float(scores[idx])))

        return results


def build_index() -> SparseIndex:
    """Convenience function: build and return a SparseIndex."""
    idx = SparseIndex()
    idx.build_index()
    return idx


def search(query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Convenience function: load index and search."""
    idx = SparseIndex()
    idx.load_index()
    return idx.search(query, top_k=top_k)
