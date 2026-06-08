"""
Context window limitation tests.
Tests LLM performance as context length increases to identify attention degradation.
"""

import random
from typing import Any, Dict, List

from ..core.llm_client import LLMResponse
from .base_test import BaseFailureTest, TestCase, TestResult


class ContextLimitsTest(BaseFailureTest):
    """
    Test LLM behavior with increasing context lengths.

    Theory: Transformer attention degrades with distance. Models lose track of
    information at the beginning or middle of long contexts.
    """

    def __init__(self):
        super().__init__(
            name="Context Window Limits", description="Tests attention degradation in long contexts"
        )

    def generate_test_cases(self, num_cases: int = 10, **kwargs) -> List[TestCase]:
        """
        Generate test cases with increasing context lengths.

        Strategy: Hide key information at different positions in the context
        and ask questions about it.
        """
        test_cases = []

        # Context lengths to test (in tokens, approximately)
        context_lengths = [500, 1000, 2000, 4000, 8000]

        for context_length in context_lengths:
            # Generate filler text
            filler_text = self._generate_filler_text(context_length)

            # Test 1: Key info at beginning
            key_info_start = "The secret code is ALPHA-7239."
            context_start = f"{key_info_start}\n\n{filler_text}"
            test_cases.append(
                TestCase(
                    id=f"context_{context_length}_start",
                    input=f"{context_start}\n\nQuestion: What is the secret code?",
                    expected_output="ALPHA-7239",
                    metadata={
                        "context_length": context_length,
                        "info_position": "start",
                        "position_ratio": 0.0,
                    },
                )
            )

            # Test 2: Key info in middle
            half_filler = self._generate_filler_text(context_length // 2)
            key_info_middle = "The secret code is BETA-8451."
            context_middle = f"{half_filler}\n\n{key_info_middle}\n\n{half_filler}"
            test_cases.append(
                TestCase(
                    id=f"context_{context_length}_middle",
                    input=f"{context_middle}\n\nQuestion: What is the secret code?",
                    expected_output="BETA-8451",
                    metadata={
                        "context_length": context_length,
                        "info_position": "middle",
                        "position_ratio": 0.5,
                    },
                )
            )

            # Test 3: Key info at end
            key_info_end = "The secret code is GAMMA-9362."
            context_end = f"{filler_text}\n\n{key_info_end}"
            test_cases.append(
                TestCase(
                    id=f"context_{context_length}_end",
                    input=f"{context_end}\n\nQuestion: What is the secret code?",
                    expected_output="GAMMA-9362",
                    metadata={
                        "context_length": context_length,
                        "info_position": "end",
                        "position_ratio": 1.0,
                    },
                )
            )

        self.test_cases = test_cases
        return test_cases

    def _generate_filler_text(self, target_length: int) -> str:
        """Generate filler text of approximately target_length tokens."""
        # Rough estimate: 1 token ≈ 4 characters
        target_chars = target_length * 4

        paragraphs = [
            "The development of artificial intelligence has transformed modern technology. "
            "Machine learning models process vast amounts of data to identify patterns. "
            "Neural networks consist of interconnected layers that transform input data. "
            "Training requires careful tuning of hyperparameters and optimization strategies.",
            "Natural language processing enables computers to understand human language. "
            "Transformer architectures use attention mechanisms to capture dependencies. "
            "Large language models demonstrate impressive capabilities across various tasks. "
            "Fine-tuning adapts pre-trained models to specific domains and applications.",
            "Computer vision systems analyze and interpret visual information from images. "
            "Convolutional neural networks excel at recognizing patterns in image data. "
            "Object detection identifies and localizes multiple objects within scenes. "
            "Image segmentation assigns labels to individual pixels in images.",
            "Deep reinforcement learning combines neural networks with decision making. "
            "Agents learn optimal policies through interaction with environments. "
            "Reward signals guide the learning process toward desired behaviors. "
            "Applications range from game playing to robotic control systems.",
        ]

        text = ""
        while len(text) < target_chars:
            text += random.choice(paragraphs) + "\n\n"

        return text[:target_chars]

    def evaluate_single(
        self, test_case: TestCase, prediction: str, response: LLMResponse
    ) -> TestResult:
        """Evaluate if model correctly retrieved information from context."""
        # Check if expected code is in prediction
        success = test_case.expected_output.lower() in prediction.lower()

        # Calculate metrics
        metrics = {
            "exact_match": 1.0 if success else 0.0,
            "context_length": test_case.metadata["context_length"],
            "position_ratio": test_case.metadata["position_ratio"],
            "latency_ms": response.latency_ms if response else 0,
            "cost_usd": response.cost_usd if response else 0,
        }

        return TestResult(
            test_case_id=test_case.id,
            prediction=prediction,
            reference=test_case.expected_output,
            success=success,
            metrics=metrics,
            response=response,
        )

    def get_diagnostic_insights(self) -> Dict[str, Any]:
        """Analyze results to provide insights on context limitations."""
        if not self.results:
            return {"error": "No results available"}

        # Group by context length and position
        by_length = {}
        by_position = {"start": [], "middle": [], "end": []}

        for result in self.results:
            length = result.metrics.get("context_length", 0)
            position = result.test_case_id.split("_")[-1]

            if length not in by_length:
                by_length[length] = []
            by_length[length].append(result.metrics.get("exact_match", 0.0))
            by_position[position].append(result.metrics.get("exact_match", 0.0))

        # Calculate success rates
        length_analysis = {
            length: sum(scores) / len(scores) for length, scores in by_length.items()
        }

        position_analysis = {
            position: sum(scores) / len(scores) for position, scores in by_position.items()
        }

        # Identify critical length
        critical_length = None
        for length in sorted(by_length.keys()):
            if length_analysis[length] < 0.5:  # <50% success
                critical_length = length
                break

        # Recommendations
        recommendations = []

        if critical_length and critical_length < 4000:
            recommendations.append(
                f"Context window becomes unreliable beyond {critical_length} tokens. "
                "Consider using RAG with chunking for longer documents."
            )

        if position_analysis["middle"] < position_analysis["start"]:
            recommendations.append(
                "Model struggles with information in the middle of context. "
                "Place critical information at beginning or end of prompts."
            )

        if position_analysis["start"] < 0.8:
            recommendations.append(
                "Even early context information is being lost. "
                "This model may not be suitable for this task without architectural changes."
            )

        return {
            "bottleneck_identified": critical_length is not None,
            "critical_context_length": critical_length,
            "success_by_length": length_analysis,
            "success_by_position": position_analysis,
            "recommendations": recommendations,
            "theoretical_explanation": (
                "Transformer attention weights decay with distance. As context grows, "
                "the model allocates less attention to tokens far from the current position, "
                "leading to information loss."
            ),
        }


# Example usage
if __name__ == "__main__":
    from ..core.llm_client import get_llm_client

    # Initialize test
    test = ContextLimitsTest()
    test.generate_test_cases(num_cases=15)

    # Run with GPT-4
    client = get_llm_client("gpt-4-turbo-preview")
    results = test.run_test(client, max_tokens=100)

    # Get diagnostics
    insights = test.get_diagnostic_insights()
    print(insights)

    # Save results
    test.save_results("context_limits_results.json")
