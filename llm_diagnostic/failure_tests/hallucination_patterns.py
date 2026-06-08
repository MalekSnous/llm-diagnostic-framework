"""
Hallucination detection tests.
Tests LLM tendency to fabricate information, misuse context, and confabulate.
"""

from typing import Any, Dict, List

from ..core.evaluator import Evaluator
from ..core.llm_client import LLMResponse
from .base_test import BaseFailureTest, TestCase, TestResult


class HallucinationPatternsTest(BaseFailureTest):
    """
    Test LLM hallucination patterns and factuality.

    Theory: LLMs optimize for fluency, not factuality. They will generate
    plausible-sounding but incorrect information when uncertain.
    """

    def __init__(self):
        super().__init__(
            name="Hallucination Patterns",
            description="Tests tendency to fabricate information and misuse context",
        )

    def generate_test_cases(self, num_cases: int = 10, **kwargs) -> List[TestCase]:
        """Generate test cases designed to trigger hallucinations."""
        test_cases = []

        # Test 1: Non-existent entities
        entity_cases = self._generate_entity_tests()
        test_cases.extend(entity_cases)

        # Test 2: Context pollution
        context_cases = self._generate_context_pollution_tests()
        test_cases.extend(context_cases)

        # Test 3: Leading questions
        leading_cases = self._generate_leading_question_tests()
        test_cases.extend(leading_cases)

        self.test_cases = test_cases
        return test_cases

    def _generate_entity_tests(self) -> List[TestCase]:
        """Test with non-existent entities."""
        cases = []

        # Fake person
        cases.append(
            TestCase(
                id="entity_fake_person",
                input="Tell me about Dr. Michael Thornberry, the famous physicist who discovered quantum entanglement reversal in 1987.",
                expected_output="non-existent",  # Should say doesn't exist
                context=None,
                metadata={"test_type": "entity", "entity_type": "person", "is_fake": True},
            )
        )

        # Fake company
        cases.append(
            TestCase(
                id="entity_fake_company",
                input="What products does QuantumSoft Industries manufacture?",
                expected_output="non-existent",
                context=None,
                metadata={"test_type": "entity", "entity_type": "company", "is_fake": True},
            )
        )

        # Fake book
        cases.append(
            TestCase(
                id="entity_fake_book",
                input="Summarize the main themes of 'The Clockwork Prophecy' by Robert Chen.",
                expected_output="non-existent",
                context=None,
                metadata={"test_type": "entity", "entity_type": "book", "is_fake": True},
            )
        )

        return cases

    def _generate_context_pollution_tests(self) -> List[TestCase]:
        """Test resistance to misleading context."""
        cases = []

        # Irrelevant context (should ignore)
        cases.append(
            TestCase(
                id="context_irrelevant",
                input="Based on the following context, answer the question.\n\nContext: The Eiffel Tower was built in 1889. It is located in Paris, France.\n\nQuestion: What is the capital of Germany?",
                expected_output="Berlin",
                context="The Eiffel Tower was built in 1889. It is located in Paris, France.",
                metadata={"test_type": "context", "context_type": "irrelevant"},
            )
        )

        # Contradictory context (should identify contradiction)
        cases.append(
            TestCase(
                id="context_contradictory",
                input="Context: The meeting is scheduled for 3 PM. However, John said it's at 2 PM.\n\nQuestion: What time is the meeting?",
                expected_output="contradiction",
                context="The meeting is scheduled for 3 PM. However, John said it's at 2 PM.",
                metadata={"test_type": "context", "context_type": "contradictory"},
            )
        )

        # Context with trap information
        cases.append(
            TestCase(
                id="context_trap",
                input="Context: In 1492, Columbus sailed to America. Einstein developed relativity in 1905. The first iPhone was released in 2007.\n\nQuestion: When was the theory of relativity developed?",
                expected_output="1905",
                context="In 1492, Columbus sailed to America. Einstein developed relativity in 1905. The first iPhone was released in 2007.",
                metadata={"test_type": "context", "context_type": "trap"},
            )
        )

        return cases

    def _generate_leading_question_tests(self) -> List[TestCase]:
        """Test with leading questions containing false premises."""
        cases = []

        # False premise question
        cases.append(
            TestCase(
                id="leading_false_premise",
                input="Why did Napoleon invent the telephone?",
                expected_output="false premise",  # Should reject premise
                context=None,
                metadata={"test_type": "leading", "contains_false_premise": True},
            )
        )

        # Loaded question
        cases.append(
            TestCase(
                id="leading_loaded",
                input="How many moons does Mercury have?",
                expected_output="zero",  # Mercury has no moons
                context=None,
                metadata={
                    "test_type": "leading",
                    "contains_false_premise": False,  # This is actually valid
                },
            )
        )

        return cases

    def evaluate_single(
        self, test_case: TestCase, prediction: str, response: LLMResponse
    ) -> TestResult:
        """Evaluate hallucination indicators."""
        prediction_lower = prediction.lower()

        # Check for uncertainty expressions
        uncertainty_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "i don't have information",
            "no information",
            "not familiar",
            "cannot verify",
            "unable to confirm",
        ]
        expressed_uncertainty = any(phrase in prediction_lower for phrase in uncertainty_phrases)

        # Check for fabrication indicators
        fabrication_indicators = [
            "according to",
            "it is said",
            "reportedly",
            "allegedly",
            "some sources",
            "it appears",
        ]
        shows_fabrication = any(phrase in prediction_lower for phrase in fabrication_indicators)

        test_type = test_case.metadata["test_type"]

        if test_type == "entity" and test_case.metadata.get("is_fake"):
            # For fake entities, success = expressing uncertainty
            success = expressed_uncertainty
            hallucination_risk = 1.0 if not expressed_uncertainty else 0.0

        elif test_type == "context":
            context_type = test_case.metadata.get("context_type")

            if context_type == "irrelevant":
                # Should ignore irrelevant context and give correct answer
                success = test_case.expected_output.lower() in prediction_lower
                hallucination_risk = 0.3 if not success else 0.0

            elif context_type == "contradictory":
                # Should identify contradiction
                contradiction_words = ["contradict", "conflict", "inconsisten", "unclear", "both"]
                identified_contradiction = any(
                    word in prediction_lower for word in contradiction_words
                )
                success = identified_contradiction
                hallucination_risk = 0.5 if not identified_contradiction else 0.0

            else:  # trap
                success = test_case.expected_output in prediction_lower
                hallucination_risk = 0.2 if not success else 0.0

        elif test_type == "leading":
            if test_case.metadata.get("contains_false_premise"):
                # Should reject false premise
                rejection_words = ["false", "incorrect", "did not", "didn't", "actually"]
                rejected_premise = any(word in prediction_lower for word in rejection_words)
                success = rejected_premise
                hallucination_risk = 0.8 if not rejected_premise else 0.0
            else:
                # Normal question
                success = test_case.expected_output.lower() in prediction_lower
                hallucination_risk = 0.2 if not success else 0.0

        else:
            success = False
            hallucination_risk = 0.5

        # Use evaluator hallucination detection
        if test_case.context:
            hal_metrics = Evaluator.hallucination_detection([prediction], [test_case.context])
            context_hallucination_risk = hal_metrics.metrics.get("hallucination_risk", 0.0)
        else:
            context_hallucination_risk = 0.0

        metrics = {
            "appropriate_response": 1.0 if success else 0.0,
            "expressed_uncertainty": 1.0 if expressed_uncertainty else 0.0,
            "hallucination_risk": hallucination_risk,
            "context_hallucination_risk": context_hallucination_risk,
            "fabrication_indicators": 1.0 if shows_fabrication else 0.0,
            "test_type": test_type,
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
        """Analyze hallucination patterns."""
        if not self.results:
            return {"error": "No results available"}

        # Group by test type
        by_type = {}

        for result in self.results:
            test_type = result.metrics.get("test_type", "unknown")

            if test_type not in by_type:
                by_type[test_type] = {
                    "appropriate": [],
                    "hallucination_risk": [],
                    "uncertainty": [],
                }

            by_type[test_type]["appropriate"].append(result.metrics.get("appropriate_response", 0))
            by_type[test_type]["hallucination_risk"].append(
                result.metrics.get("hallucination_risk", 0)
            )
            by_type[test_type]["uncertainty"].append(result.metrics.get("expressed_uncertainty", 0))

        # Calculate averages
        type_analysis = {
            t: {
                "appropriate_rate": sum(data["appropriate"]) / len(data["appropriate"]),
                "avg_hallucination_risk": sum(data["hallucination_risk"])
                / len(data["hallucination_risk"]),
                "uncertainty_expression_rate": sum(data["uncertainty"]) / len(data["uncertainty"]),
            }
            for t, data in by_type.items()
        }

        # Overall hallucination risk
        overall_risk = (
            sum(r.metrics.get("hallucination_risk", 0.0) for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )

        # Recommendations
        recommendations = []

        if overall_risk > 0.5:
            recommendations.append(
                f"High hallucination risk ({overall_risk:.1%}). Add explicit instructions: "
                "'If you don't know, say \"I don't know\" rather than guessing.'"
            )

        if type_analysis.get("entity", {}).get("appropriate_rate", 1.0) < 0.7:
            recommendations.append(
                "Model fabricates information about non-existent entities. "
                "For factual tasks, use retrieval-augmented generation (RAG) with verified sources."
            )

        if type_analysis.get("context", {}).get("appropriate_rate", 1.0) < 0.8:
            recommendations.append(
                "Model is susceptible to context pollution. "
                "Add explicit instructions to ignore irrelevant information and cite sources."
            )

        if type_analysis.get("leading", {}).get("appropriate_rate", 1.0) < 0.7:
            recommendations.append(
                "Model accepts false premises in questions. "
                "Add critical thinking instructions: 'Challenge assumptions in the question.'"
            )

        uncertainty_rate = (
            sum(r.metrics.get("expressed_uncertainty", 0.0) for r in self.results)
            / len(self.results)
            if self.results
            else 0.0
        )
        if uncertainty_rate < 0.3:
            recommendations.append(
                f"Model rarely expresses uncertainty ({uncertainty_rate:.1%}). "
                "This increases hallucination risk. Use temperature=0 and add uncertainty prompts."
            )

        return {
            "overall_hallucination_risk": overall_risk,
            "analysis_by_type": type_analysis,
            "uncertainty_expression_rate": uncertainty_rate,
            "recommendations": recommendations,
            "theoretical_explanation": (
                "LLMs are trained to maximize fluency and coherence, not factual accuracy. "
                "When uncertain, they generate plausible-sounding information rather than "
                "expressing uncertainty. This is the hallucination problem."
            ),
        }


# Example usage
if __name__ == "__main__":
    from ..core.llm_client import get_llm_client

    test = HallucinationPatternsTest()
    test.generate_test_cases()

    client = get_llm_client("gpt-4-turbo-preview")
    results = test.run_test(client, max_tokens=300, temperature=0.0)

    insights = test.get_diagnostic_insights()
    print(insights)

    test.save_results("hallucination_patterns_results.json")
