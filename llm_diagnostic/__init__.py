"""
LLM Diagnostic Framework
========================

A systematic, theory-driven framework for diagnosing LLM failures
and implementing targeted improvements.

Author: Malek Senoussi, PhD
GitHub: https://github.com/MalekSnous/llm-diagnostic-framework
"""

__version__ = "0.1.0"
__author__ = "Malek Senoussi"
__email__ = "malek.senoussi@gmail.com"

# Core imports
from .core.evaluator import EvaluationMetrics, Evaluator
from .core.llm_client import BaseLLMClient, LLMResponse, get_llm_client

# Base classes (for custom implementations)
from .failure_tests.base_test import BaseFailureTest, TestCase, TestResult

# Failure tests
from .failure_tests.context_limits import ContextLimitsTest
from .failure_tests.hallucination_patterns import HallucinationPatternsTest
from .failure_tests.knowledge_boundaries import KnowledgeBoundariesTest
from .failure_tests.reasoning_depth import ReasoningDepthTest
from .failure_tests.structure_validation import StructureValidationTest
from .improvements.base_strategy import BaseImprovementStrategy, ImprovementConfig
from .improvements.fine_tuning import FineTuningStrategy

# Improvement strategies
from .improvements.prompt_engineering import PromptEngineeringStrategy
from .improvements.rag_system import RAGSystem

__all__ = [
    # Core
    "get_llm_client",
    "LLMResponse",
    "BaseLLMClient",
    "Evaluator",
    "EvaluationMetrics",
    # Tests
    "ContextLimitsTest",
    "ReasoningDepthTest",
    "KnowledgeBoundariesTest",
    "StructureValidationTest",
    "HallucinationPatternsTest",
    # Improvements
    "PromptEngineeringStrategy",
    "RAGSystem",
    "FineTuningStrategy",
    # Base classes
    "BaseFailureTest",
    "TestCase",
    "TestResult",
    "BaseImprovementStrategy",
    "ImprovementConfig",
]
