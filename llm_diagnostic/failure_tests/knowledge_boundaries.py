"""
Knowledge boundary tests.
Tests LLM knowledge cutoff, rare facts, and specialized domain knowledge.
"""

from typing import Any, Dict, List

from ..core.llm_client import LLMResponse
from .base_test import BaseFailureTest, TestCase, TestResult


class KnowledgeBoundariesTest(BaseFailureTest):
    """
    Test LLM knowledge boundaries and limitations.

    Theory: LLMs are limited by training data cutoff, frequency bias,
    and domain coverage.
    """

    def __init__(self, knowledge_cutoff: str = "2024-01"):
        super().__init__(
            name="Knowledge Boundaries",
            description="Tests knowledge cutoff, rare facts, and domain expertise",
        )
        self.knowledge_cutoff = knowledge_cutoff

    def generate_test_cases(self, num_cases: int = 10, **kwargs) -> List[TestCase]:
        """Generate test cases targeting knowledge boundaries."""
        test_cases = []

        # Test 1: Temporal knowledge (before and after cutoff)
        temporal_cases = self._generate_temporal_tests()
        test_cases.extend(temporal_cases)

        # Test 2: Rare vs common facts
        frequency_cases = self._generate_frequency_tests()
        test_cases.extend(frequency_cases)

        # Test 3: Specialized domains
        domain_cases = self._generate_domain_tests()
        test_cases.extend(domain_cases)

        self.test_cases = test_cases
        return test_cases

    def _generate_temporal_tests(self) -> List[TestCase]:
        """Generate questions about events before/after knowledge cutoff."""
        cases = []

        # Events clearly before cutoff (should know)
        cases.append(
            TestCase(
                id="temporal_before_1",
                input="Who won the FIFA World Cup in 2022?",
                expected_output="Argentina",
                metadata={
                    "test_type": "temporal",
                    "temporal_category": "before_cutoff",
                    "expected_knowledge": True,
                },
            )
        )

        cases.append(
            TestCase(
                id="temporal_before_2",
                input="What company did Elon Musk acquire in 2022?",
                expected_output="Twitter",
                metadata={
                    "test_type": "temporal",
                    "temporal_category": "before_cutoff",
                    "expected_knowledge": True,
                },
            )
        )

        # Events potentially after cutoff (might not know)
        cases.append(
            TestCase(
                id="temporal_after_1",
                input="What major AI model was released in November 2024?",
                expected_output="Unknown",  # Placeholder
                metadata={
                    "test_type": "temporal",
                    "temporal_category": "after_cutoff",
                    "expected_knowledge": False,
                },
            )
        )

        cases.append(
            TestCase(
                id="temporal_after_2",
                input="Who won the 2024 US Presidential election?",
                expected_output="Donald Trump",
                metadata={
                    "test_type": "temporal",
                    "temporal_category": "near_cutoff",
                    "expected_knowledge": False,  # Depends on exact cutoff
                },
            )
        )

        return cases

    def _generate_frequency_tests(self) -> List[TestCase]:
        """Test common vs rare facts (frequency bias)."""
        cases = []

        # Common facts (high frequency in training data)
        cases.append(
            TestCase(
                id="frequency_common_1",
                input="What is the capital of France?",
                expected_output="Paris",
                metadata={
                    "test_type": "frequency",
                    "frequency_category": "very_common",
                    "expected_knowledge": True,
                },
            )
        )

        # Moderately rare facts
        cases.append(
            TestCase(
                id="frequency_rare_1",
                input="What is the capital of Burkina Faso?",
                expected_output="Ouagadougou",
                metadata={
                    "test_type": "frequency",
                    "frequency_category": "rare",
                    "expected_knowledge": True,
                },
            )
        )

        # Very rare facts (obscure)
        cases.append(
            TestCase(
                id="frequency_obscure_1",
                input="Who was the 3rd emperor of the Mali Empire?",
                expected_output="Mansa Uli",  # Very obscure
                metadata={
                    "test_type": "frequency",
                    "frequency_category": "very_rare",
                    "expected_knowledge": False,  # Likely to fail
                },
            )
        )

        return cases

    def _generate_domain_tests(self) -> List[TestCase]:
        """Test specialized domain knowledge."""
        cases = []

        # General domain (should know)
        cases.append(
            TestCase(
                id="domain_general_1",
                input="What is photosynthesis?",
                expected_output="process",  # Should contain explanation
                metadata={
                    "test_type": "domain",
                    "domain_category": "general_science",
                    "expected_knowledge": True,
                },
            )
        )

        # Specialized domain (medical)
        cases.append(
            TestCase(
                id="domain_medical_1",
                input="What is the mechanism of action of metformin?",
                expected_output="AMPK",  # Key mechanism
                metadata={
                    "test_type": "domain",
                    "domain_category": "medical",
                    "expected_knowledge": True,  # Basic medical knowledge
                },
            )
        )

        # Highly specialized (likely to fail)
        cases.append(
            TestCase(
                id="domain_specialized_1",
                input="What is the binding affinity of compound XJ-472 to the GPCR target?",
                expected_output="Unknown",  # Fake compound
                metadata={
                    "test_type": "domain",
                    "domain_category": "highly_specialized",
                    "expected_knowledge": False,
                },
            )
        )

        return cases

    def evaluate_single(
        self, test_case: TestCase, prediction: str, response: LLMResponse
    ) -> TestResult:
        """Evaluate knowledge accuracy."""
        prediction_lower = prediction.lower()

        # Check for explicit "I don't know" statements
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "i don't have",
            "not available",
            "cannot provide",
            "after my knowledge cutoff",
            "beyond my training",
            "i cannot answer",
        ]
        expressed_uncertainty = any(phrase in prediction_lower for phrase in uncertainty_phrases)

        # Check if answer contains expected information
        if test_case.expected_output != "Unknown":
            contains_answer = test_case.expected_output.lower() in prediction_lower
        else:
            contains_answer = False

        # Determine success based on whether model should know
        expected_knowledge = test_case.metadata.get("expected_knowledge", True)

        if expected_knowledge:
            # Should know: success if answer present and no uncertainty
            success = contains_answer and not expressed_uncertainty
        else:
            # Shouldn't know: success if expresses uncertainty OR doesn't hallucinate
            success = expressed_uncertainty or not contains_answer

        metrics = {
            "contains_expected_answer": 1.0 if contains_answer else 0.0,
            "expressed_uncertainty": 1.0 if expressed_uncertainty else 0.0,
            "appropriate_response": 1.0 if success else 0.0,
            "test_type": test_case.metadata["test_type"],
            "expected_to_know": 1.0 if expected_knowledge else 0.0,
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
        """Analyze knowledge boundary patterns."""
        if not self.results:
            return {"error": "No results available"}

        # Group by test type
        by_test_type = {}
        by_category = {}

        for result in self.results:
            test_type = result.metrics.get("test_type", "unknown")

            # Extract category from test_case_id
            parts = result.test_case_id.split("_")
            category = "_".join(parts[1:-1]) if len(parts) > 2 else "unknown"

            if test_type not in by_test_type:
                by_test_type[test_type] = []
            by_test_type[test_type].append(result.metrics.get("appropriate_response", 0))

            if category not in by_category:
                by_category[category] = {"correct": [], "uncertain": [], "expected_to_know": []}

            by_category[category]["correct"].append(result.metrics.get("appropriate_response", 0))
            by_category[category]["uncertain"].append(
                result.metrics.get("expressed_uncertainty", 0)
            )
            by_category[category]["expected_to_know"].append(
                result.metrics.get("expected_to_know", 0)
            )

        # Calculate averages
        type_analysis = {t: sum(scores) / len(scores) for t, scores in by_test_type.items()}

        category_analysis = {
            cat: {
                "success_rate": sum(data["correct"]) / len(data["correct"]),
                "uncertainty_rate": sum(data["uncertain"]) / len(data["uncertain"]),
                "was_expected_to_know": sum(data["expected_to_know"])
                / len(data["expected_to_know"]),
            }
            for cat, data in by_category.items()
        }

        # Recommendations
        recommendations = []

        if type_analysis.get("temporal", 1.0) < 0.8:
            recommendations.append(
                f"Model knowledge is limited by training cutoff ({self.knowledge_cutoff}). "
                "For recent events, use RAG with up-to-date sources or web search."
            )

        if type_analysis.get("frequency", 1.0) < 0.7:
            recommendations.append(
                "Model struggles with rare/obscure facts due to frequency bias. "
                "For specialized knowledge, consider domain-specific fine-tuning or retrieval."
            )

        if type_analysis.get("domain", 1.0) < 0.7:
            recommendations.append(
                "Specialized domain knowledge is unreliable. "
                "Use domain-specific models or retrieval from expert sources."
            )

        # Check hallucination risk
        hallucination_cases = [
            r
            for r in self.results
            if r.metrics.get("expected_to_know", 0.0) == 0.0
            and r.metrics.get("expressed_uncertainty", 0.0) == 0.0
        ]
        hallucination_rate = len(hallucination_cases) / len(self.results)

        if hallucination_rate > 0.3:
            recommendations.append(
                f"High hallucination risk ({hallucination_rate:.1%}): Model fabricates answers "
                "for unknown facts instead of expressing uncertainty. Add explicit instructions "
                "to say 'I don't know' when uncertain."
            )

        return {
            "knowledge_cutoff": self.knowledge_cutoff,
            "success_by_type": type_analysis,
            "success_by_category": category_analysis,
            "hallucination_rate": hallucination_rate,
            "recommendations": recommendations,
            "theoretical_explanation": (
                "LLM knowledge is constrained by: (1) training data cutoff, "
                "(2) frequency bias (rare facts underrepresented), and "
                "(3) domain coverage (specialized knowledge often missing)."
            ),
        }


# Example usage
if __name__ == "__main__":
    from ..core.llm_client import get_llm_client

    test = KnowledgeBoundariesTest(knowledge_cutoff="2024-01")
    test.generate_test_cases()

    client = get_llm_client("gpt-4-turbo-preview")
    results = test.run_test(client, max_tokens=200)

    insights = test.get_diagnostic_insights()
    print(insights)

    test.save_results("knowledge_boundaries_results.json")
