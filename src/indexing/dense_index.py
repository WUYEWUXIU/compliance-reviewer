"""FAISS dense vector index using Bailian text-embedding-v3."""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import requests

from src.config.settings import (
    BAILIAN_API_KEY,
    BAILIAN_EMBEDDING_MODEL,
    CHUNKS_DIR,
    EMBEDDING_TIMEOUT,
    INDEXES_DIR,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# Bailian embedding API endpoint
BAILIAN_EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

# Embedding dimension for text-embedding-v3
EMBEDDING_DIM = 1024

# Batch size for API calls to avoid rate limits / payload size issues
EMBEDDING_BATCH_SIZE = 10


def _get_chunks() -> List[dict]:
    """Load chunks from the canonical chunks file."""
    chunks_path = CHUNKS_DIR / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _call_bailian_embedding(texts: List[str]) -> np.ndarray:
    """Call Bailian embedding API for a batch of texts.

    Args:
        texts: List of input texts.

    Returns:
        Numpy array of shape (len(texts), EMBEDDING_DIM).

    Raises:
        RuntimeError: If the API call fails after retries.
    """
    headers = {
        "Authorization": f"Bearer {BAILIAN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": BAILIAN_EMBEDDING_MODEL,
        "input": {"texts": texts},
        "parameters": {"text_type": "document"},
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                BAILIAN_EMBEDDING_URL,
                headers=headers,
                json=payload,
                timeout=EMBEDDING_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()

            if "output" not in result or "embeddings" not in result["output"]:
                raise RuntimeError(f"Unexpected API response format: {result}")

            embeddings = result["output"]["embeddings"]
            vectors = np.array(
                [e["embedding"] for e in embeddings], dtype=np.float32
            )
            return vectors

        except Exception as exc:
            last_error = exc
            logger.warning(
                "Bailian embedding API attempt %d/%d failed: %s",
                attempt,
                MAX_RETRIES,
                exc,
            )

    raise RuntimeError(
        f"Bailian embedding API failed after {MAX_RETRIES} retries: {last_error}"
    )


def _generate_random_vectors(count: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Generate random normalized vectors as fallback when API key is missing.

    Args:
        count: Number of vectors to generate.
        dim: Vector dimension.

    Returns:
        Numpy array of shape (count, dim) with L2-normalized rows.
    """
    rng = np.random.default_rng(seed=42)
    vectors = rng.standard_normal(size=(count, dim)).astype(np.float32)
    # L2 normalize so that inner product equals cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    vectors = vectors / norms
    return vectors


def _compute_embeddings(chunks: List[dict]) -> Tuple[np.ndarray, List[str]]:
    """Compute or fallback-generate embeddings for all chunks.

    Args:
        chunks: List of chunk dictionaries.

    Returns:
        Tuple of (embeddings array, list of chunk_ids).
    """
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["article_text"] for c in chunks]

    if not BAILIAN_API_KEY:
        logger.warning(
            "BAILIAN_API_KEY not set. Using random vectors as fallback."
        )
        vectors = _generate_random_vectors(len(chunks))
        return vectors, chunk_ids

    all_vectors: List[np.ndarray] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        logger.info(
            "Embedding batch %d/%d (%d texts)",
            i // EMBEDDING_BATCH_SIZE + 1,
            (len(texts) - 1) // EMBEDDING_BATCH_SIZE + 1,
            len(batch),
        )
        batch_vectors = _call_bailian_embedding(batch)
        all_vectors.append(batch_vectors)

    vectors = np.vstack(all_vectors)
    return vectors, chunk_ids


class DenseIndex:
    """FAISS dense vector index with Bailian embeddings."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunk_ids: List[str] = []
        self.index_path = INDEXES_DIR / "faiss.index"
        self.mapping_path = INDEXES_DIR / "faiss_mapping.json"

    def build_index(self) -> None:
        """Build the FAISS index from chunks and persist to disk."""
        chunks = _get_chunks()
        if not chunks:
            raise ValueError("No chunks found to index.")

        vectors, chunk_ids = _compute_embeddings(chunks)

        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dim {EMBEDDING_DIM}, got {vectors.shape[1]}"
            )

        # Ensure vectors are normalized for IndexFlatIP (inner product ~ cosine)
        faiss.normalize_L2(vectors)

        # IndexFlatIP: exact search with inner product
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(vectors)
        self.chunk_ids = chunk_ids

        # Persist
        INDEXES_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f, ensure_ascii=False, indent=2)

        logger.info(
            "Dense index built: %d vectors, saved to %s",
            len(self.chunk_ids),
            self.index_path,
        )

    def load_index(self) -> None:
        """Load an existing FAISS index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Mapping file not found: {self.mapping_path}"
            )

        self.index = faiss.read_index(str(self.index_path))
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.chunk_ids = json.load(f)

        logger.info(
            "Dense index loaded: %d vectors from %s",
            len(self.chunk_ids),
            self.index_path,
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the dense index for the most similar chunks.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending.
        """
        if self.index is None:
            self.load_index()

        if not BAILIAN_API_KEY:
            # Fallback: use random query vector for consistent behavior in test mode
            query_vector = _generate_random_vectors(1)
        else:
            query_vector = _call_bailian_embedding([query])

        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.index.search(query_vector, top_k)

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append((self.chunk_ids[idx], float(score)))

        return results


def build_index() -> DenseIndex:
    """Convenience function: build and return a DenseIndex."""
    idx = DenseIndex()
    idx.build_index()
    return idx


def search(query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Convenience function: load index and search."""
    idx = DenseIndex()
    idx.load_index()
    return idx.search(query, top_k=top_k)
