import dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEXES_DIR = DATA_DIR / "indexes"
REFERENCES_DIR = DATA_DIR / "references"

# API keys from environment (do not hardcode)
dotenv.load_dotenv()  # Load from .env if present
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
BAILIAN_EMBEDDING_MODEL = os.getenv(
    "BAILIAN_EMBEDDING_MODEL", "text-embedding-v3")
BAILIAN_RERANK_MODEL = os.getenv("BAILIAN_RERANK_MODEL", "gte-rerank")
BAILIAN_LLM_MODEL = os.getenv("BAILIAN_LLM_MODEL", "qwen-max")

# Retrieval settings
RRF_K = 20
TOP_K_BM25 = 10
TOP_K_VECTOR = 10
TOP_K_RERANK = 5
RERANK_THRESHOLD = 0.3

# Confidence weights
W_RERANK = 0.4
W_COVERAGE = 0.2
W_AGREEMENT = 0.2
W_DIVERSITY = 0.2

# API timeouts & retries
EMBEDDING_TIMEOUT = 30
RERANK_TIMEOUT = 15
LLM_TIMEOUT = 30
MAX_RETRIES = 3
