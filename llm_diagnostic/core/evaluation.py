"""
Backwards-compatible alias module.

The canonical implementation lives in :mod:`llm_diagnostic.core.evaluator`.
This module used to be a verbatim copy; it now re-exports the public API so
existing imports (``from llm_diagnostic.core.evaluation import Evaluator``)
keep working without duplicating the source.
"""

from .evaluator import EvaluationMetrics, Evaluator

__all__ = ["Evaluator", "EvaluationMetrics"]
