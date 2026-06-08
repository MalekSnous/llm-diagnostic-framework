"""
Prompt engineering improvement strategy.
Tests various prompting techniques: few-shot, chain-of-thought, structured prompts.
"""

import time
from typing import Any, Dict, List

from ..core.evaluator import EvaluationMetrics
from ..core.llm_client import BaseLLMClient
from ..failure_tests.base_test import TestCase, TestResult
from .base_strategy import BaseImprovementStrategy, ImprovementConfig, ImprovementResult


class PromptEngineeringStrategy(BaseImprovementStrategy):
    """
    Improve LLM performance through advanced prompting techniques.

    Techniques:
    - Zero-shot → Few-shot (with examples)
    - Basic prompt → Chain-of-thought
    - Unstructured → Structured output instructions
    """

    def __init__(self):
        super().__init__(
            name="Prompt Engineering", description="Improve performance through better prompts"
        )

    def configure(self, technique: str = "few-shot", **kwargs) -> ImprovementConfig:
        """
        Configure prompting technique.

        Args:
            technique: "few-shot", "chain-of-thought", "structured", or "combined"
            **kwargs: Additional parameters
        """
        return ImprovementConfig(
            strategy_name=self.name,
            parameters={
                "technique": technique,
                "num_examples": kwargs.get("num_examples", 3),
                "use_cot": kwargs.get("use_cot", False),
                "use_structure": kwargs.get("use_structure", False),
            },
            estimated_cost=0.0,  # Prompt engineering is free
            estimated_time="Minutes to hours",
        )

    def apply(
        self,
        test_cases: List[TestCase],
        baseline_results: List[TestResult],
        config: ImprovementConfig,
        llm_client: BaseLLMClient,
        **kwargs,
    ) -> List[TestResult]:
        """Apply prompt engineering improvements."""
        technique = config.parameters["technique"]

        improved_results = []
        start_time = time.time()

        for test_case in test_cases:
            # Build improved prompt based on technique
            if technique == "few-shot":
                improved_prompt = self._add_few_shot_examples(
                    test_case.input, num_examples=config.parameters["num_examples"]
                )

            elif technique == "chain-of-thought":
                improved_prompt = self._add_chain_of_thought(test_case.input)

            elif technique == "structured":
                improved_prompt = self._add_structure_instructions(test_case.input)

            elif technique == "combined":
                improved_prompt = self._add_few_shot_examples(test_case.input)
                improved_prompt = self._add_chain_of_thought(improved_prompt)
                improved_prompt = self._add_structure_instructions(improved_prompt)

            else:
                improved_prompt = test_case.input

            # Generate with improved prompt
            response = llm_client.generate(
                improved_prompt, max_tokens=kwargs.get("max_tokens", 500)
            )

            # Evaluate (simplified - would use actual evaluation logic)
            success = (
                test_case.expected_output.lower() in response.text.lower()
                if test_case.expected_output
                else True
            )

            improved_results.append(
                TestResult(
                    test_case_id=test_case.id,
                    prediction=response.text,
                    reference=test_case.expected_output,
                    success=success,
                    metrics={"accuracy": 1.0 if success else 0.0},
                    response=response,
                )
            )

        elapsed = time.time() - start_time

        # Calculate metrics
        baseline_metrics = self._calculate_metrics(baseline_results)
        improved_metrics = self._calculate_metrics(improved_results)

        # Store result
        result = ImprovementResult(
            strategy_name=self.name,
            config=config,
            baseline_metrics=baseline_metrics,
            improved_metrics=improved_metrics,
            improvement_delta=self.evaluate_improvement(baseline_metrics, improved_metrics),
            total_cost=sum(
                r.response.cost_usd for r in improved_results if r.response and r.response.cost_usd
            ),
            total_time_seconds=elapsed,
        )

        self.results.append(result)
        return improved_results

    #     def _add_few_shot_examples(self, prompt: str, num_examples: int = 3) -> str:
    #         """Add few-shot examples to prompt."""
    #         # Generic examples (in practice, should be task-specific)
    #         examples = """Here are some examples:

    # Example 1:
    # Input: What is 2+2?
    # Output: 4

    # Example 2:
    # Input: What is the capital of France?
    # Output: Paris

    # Example 3:
    # Input: Who wrote Romeo and Juliet?
    # Output: William Shakespeare

    # Now, please answer this:
    # """
    #        return examples + prompt

    def _add_few_shot_examples(self, prompt, num_examples=3):
        """Add few-shot examples for medical entity extraction."""

        # Détecter si c'est une tâche médicale
        if "medical entities" in prompt.lower() or "extract" in prompt.lower():
            examples = [
                "Example: From 'Patient with diabetes on insulin', extract: diabetes, insulin",
                "Example: From 'Diagnosed with pneumonia, prescribed antibiotics', extract: pneumonia, antibiotics",
                "Example: From 'Underwent cardiac catheterization', extract: cardiac catheterization",
                "Example: From 'Pt c/o SOB, h/o COPD. Rx: Albuterol PRN', extract: shortness of breath, COPD, albuterol",
                "Example: From '45M PMH: HTN, DM2 on metformin 500mg BID', extract: hypertension, diabetes type 2, metformin",
                "Example: From 'CXR shows infiltrate. Dx: pneumonia. Started levofloxacin', extract: chest x-ray, infiltrate, pneumonia, levofloxacin",
            ]
        else:
            # Exemples génériques par défaut
            examples = [
                """Here are some examples:

                                Example 1:
                                Input: What is 2+2?
                                Output: 4""",
                """  Example 2:
                                Input: What is the capital of France?
                                Output: Paris""",
                """ Example 3:
                                Input: Who wrote Romeo and Juliet?
                                Output: William Shakespeare

                                Now, please answer this:
                                """,
            ]

        # Ajouter les exemples au prompt
        examples_text = "\n".join(examples[:num_examples])
        return f"{examples_text}\n\nNow:\n{prompt}"

    def _add_chain_of_thought(self, prompt: str) -> str:
        """Add chain-of-thought instructions."""
        cot_instruction = """Let's think step by step:

1. First, understand the question
2. Break down the problem
3. Reason through each step
4. Provide the final answer

"""
        return cot_instruction + prompt + "\n\nPlease show your reasoning step by step."

    def _add_structure_instructions(self, prompt: str) -> str:
        """Add structured output instructions."""
        structure_instruction = """Please format your response as follows:

**Analysis:**
[Your reasoning here]

**Answer:**
[Your final answer here]

**Confidence:**
[High/Medium/Low]

"""
        return structure_instruction + prompt

    def _calculate_metrics(self, results: List[TestResult]) -> EvaluationMetrics:
        """Calculate aggregate metrics from results."""
        metrics = EvaluationMetrics()

        if not results:
            return metrics

        # Calculate accuracy
        accuracy = sum(r.metrics.get("accuracy", 0) for r in results) / len(results)
        metrics.add_metric("accuracy", accuracy)

        # Calculate cost
        total_cost = sum(r.response.cost_usd for r in results if r.response and r.response.cost_usd)
        metrics.add_metric("total_cost", total_cost)

        return metrics

    def get_implementation_guide(self) -> Dict[str, Any]:
        """Implementation guide for prompt engineering."""
        return {
            "overview": "Prompt engineering improves performance through better instructions",
            "techniques": {
                "few-shot": {
                    "description": "Provide examples before the actual task",
                    "when_to_use": "When task format is unclear or model needs guidance",
                    "cost": "Free (but increases tokens)",
                    "expected_improvement": "5-15%",
                },
                "chain-of-thought": {
                    "description": "Ask model to show reasoning steps",
                    "when_to_use": "Complex reasoning tasks, math, logic",
                    "cost": "Free (but increases tokens)",
                    "expected_improvement": "10-30% for reasoning tasks",
                },
                "structured": {
                    "description": "Specify exact output format",
                    "when_to_use": "Need consistent, parseable outputs",
                    "cost": "Free",
                    "expected_improvement": "20-40% for format compliance",
                },
            },
            "best_practices": [
                "Be explicit about what you want",
                "Provide examples for complex tasks",
                "Use temperature=0 for consistency",
                "Add 'If unsure, say I don't know' to reduce hallucinations",
                "Iterate on prompts based on failure patterns",
            ],
            "limitations": [
                "Only works within model's capabilities",
                "Diminishing returns after 3-5 examples",
                "Cannot fix fundamental knowledge gaps",
                "Increases cost due to longer prompts",
            ],
            "code_example": """
from llm_diagnostic.improvements import PromptEngineeringStrategy

# Initialize strategy
strategy = PromptEngineeringStrategy()

# Configure
config = strategy.configure(technique="few-shot", num_examples=3)

# Apply
improved_results = strategy.apply(
    test_cases=test_cases,
    baseline_results=baseline_results,
    config=config,
    llm_client=client
)

# Analyze
analysis = strategy.cost_benefit_analysis(
    improvement_delta=improved_results.improvement_delta,
    total_cost=improved_results.total_cost
)
print(analysis)
""",
        }


# Example usage
if __name__ == "__main__":
    from ..core.llm_client import get_llm_client
    from ..failure_tests.base_test import TestCase

    # Create test cases
    test_cases = [
        TestCase(id="test1", input="What is 5+7?", expected_output="12"),
        TestCase(id="test2", input="What is the capital of Italy?", expected_output="Rome"),
    ]

    # Run baseline (pretend we have results)
    client = get_llm_client("gpt-4-turbo-preview")
    baseline_results = []  # Would run actual baseline

    # Apply prompt engineering
    strategy = PromptEngineeringStrategy()
    config = strategy.configure(technique="few-shot")

    improved_results = strategy.apply(
        test_cases=test_cases, baseline_results=baseline_results, config=config, llm_client=client
    )

    print("Baseline: N/A")
    print(f"Improved: {len(improved_results)} results")
