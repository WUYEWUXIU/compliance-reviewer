"""Evaluation module for compliance review RAG system."""

from src.evaluation.metrics import EvaluationResult, compute_metrics
from src.evaluation.runner import GoldenSetRunner

__all__ = [
    "EvaluationResult",
    "compute_metrics",
    "GoldenSetRunner",
]
