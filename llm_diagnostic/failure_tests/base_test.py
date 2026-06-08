"""
Base test class for LLM failure tests.
All specific failure tests inherit from this class.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..core.evaluator import Evaluator
from ..core.llm_client import BaseLLMClient, LLMResponse


@dataclass
class TestCase:
    """Single test case."""

    id: str
    input: str
    expected_output: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestResult:
    """Result of a single test case."""

    test_case_id: str
    prediction: str
    reference: Optional[str]
    success: bool
    metrics: Dict[str, float]
    response: LLMResponse
    error: Optional[str] = None


class BaseFailureTest(ABC):
    """
    Abstract base class for failure tests.

    Each test should:
    1. Generate test cases targeting specific LLM limitations
    2. Run tests against LLM
    3. Evaluate results with appropriate metrics
    4. Provide diagnostic insights
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []

    @abstractmethod
    def generate_test_cases(self, num_cases: int = 10, **kwargs) -> List[TestCase]:
        """
        Generate test cases for this failure type.

        Args:
            num_cases: Number of test cases to generate
            **kwargs: Additional parameters for test generation

        Returns:
            List of TestCase objects
        """
        pass

    @abstractmethod
    def evaluate_single(
        self, test_case: TestCase, prediction: str, response: LLMResponse
    ) -> TestResult:
        """
        Evaluate a single test case result.

        Args:
            test_case: The test case
            prediction: Model's prediction
            response: Full LLM response with metadata

        Returns:
            TestResult object
        """
        pass

    def run_test(
        self,
        llm_client: BaseLLMClient,
        test_cases: Optional[List[TestCase]] = None,
        verbose: bool = True,
        **llm_kwargs,
    ) -> List[TestResult]:
        """
        Run all test cases against LLM.

        Args:
            llm_client: LLM client to test
            test_cases: Optional list of test cases (uses self.test_cases if None)
            verbose: Whether to show progress bar
            **llm_kwargs: Additional arguments for LLM generation

        Returns:
            List of TestResult objects
        """
        if test_cases is None:
            test_cases = self.test_cases

        if not test_cases:
            raise ValueError("No test cases available. Call generate_test_cases() first.")

        results = []
        iterator = tqdm(test_cases, desc=f"Running {self.name}") if verbose else test_cases

        for test_case in iterator:
            try:
                # Generate prediction
                response = llm_client.generate(test_case.input, **llm_kwargs)
                prediction = response.text

                # Evaluate
                result = self.evaluate_single(test_case, prediction, response)
                results.append(result)

            except Exception as e:
                # Handle errors gracefully
                results.append(
                    TestResult(
                        test_case_id=test_case.id,
                        prediction="",
                        reference=test_case.expected_output,
                        success=False,
                        metrics={},
                        response=None,
                        error=str(e),
                    )
                )

        self.results = results
        return results

    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across all test cases.

        Returns:
            Dictionary with aggregated metrics and insights
        """
        if not self.results:
            return {"error": "No results available. Run test first."}

        # Calculate success rate
        success_count = sum(1 for r in self.results if r.success)
        success_rate = success_count / len(self.results)

        # Aggregate metrics
        all_metrics = {}
        for result in self.results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Calculate aggregated metrics (only for numeric values)
        aggregated_metrics = {}
        for name, values in all_metrics.items():
            # Filter only numeric values
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except (TypeError, ValueError):
                    # Skip non-numeric values (strings, etc.)
                    pass

            if numeric_values:
                aggregated_metrics[name] = {
                    "mean": float(np.mean(numeric_values)),
                    "std": float(np.std(numeric_values)),
                    "min": float(np.min(numeric_values)),
                    "max": float(np.max(numeric_values)),
                }
            else:
                # For non-numeric, just count occurrences
                aggregated_metrics[name] = {"values": list(set(values)), "count": len(values)}

        # Cost analysis
        responses = [r.response for r in self.results if r.response]
        if responses:
            cost_metrics = Evaluator.cost_analysis(responses)
            total_cost = cost_metrics.metrics["total_cost_usd"]
            avg_latency = cost_metrics.metrics["avg_latency_ms"]
        else:
            total_cost = 0.0
            avg_latency = 0.0

        return {
            "test_name": self.name,
            "description": self.description,
            "num_test_cases": len(self.results),
            "success_rate": success_rate,
            "success_count": success_count,
            "failure_count": len(self.results) - success_count,
            "aggregated_metrics": aggregated_metrics,
            "total_cost_usd": total_cost,
            "avg_latency_ms": avg_latency,
            "errors": [r.error for r in self.results if r.error],
        }

    @abstractmethod
    def get_diagnostic_insights(self) -> Dict[str, Any]:
        """
        Provide diagnostic insights based on test results.

        Returns:
            Dictionary with diagnostic information and recommendations
        """
        pass

    def save_results(self, filepath: str):
        # """Save results to JSON file."""

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        # Helper function to convert numpy types to Python types

        # Build a mapping of test_case_id to input
        test_case_inputs = {tc.id: tc.input for tc in self.test_cases}

        data = {
            "test_name": self.name,
            "description": self.description,
            "timestamp": datetime.now().isoformat(),
            "aggregated_results": convert_numpy(self.aggregate_results()),
            "diagnostic_insights": convert_numpy(self.get_diagnostic_insights()),
            "detailed_results": [
                {
                    "test_case_id": r.test_case_id,
                    "input": test_case_inputs.get(r.test_case_id, ""),  # Add input field
                    "prediction": r.prediction,
                    "reference": r.reference,
                    "success": r.success,
                    "metrics": convert_numpy(r.metrics),
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', num_cases={len(self.test_cases)})"
