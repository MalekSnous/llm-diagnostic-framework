"""
Reasoning depth limitation tests.
Tests LLM performance on multi-hop reasoning tasks to identify depth limits.
"""

import random
from typing import Any, Dict, List, Tuple

from .base_test import BaseFailureTest, TestCase, TestResult
from ..core.llm_client import LLMResponse


class ReasoningDepthTest(BaseFailureTest):
    """
    Test LLM reasoning capabilities with increasing complexity.
    
    Theory: Transformers are sequence models, not symbolic reasoners. 
    They struggle with tasks requiring multiple reasoning steps.
    """
    
    def __init__(self):
        super().__init__(
            name="Reasoning Depth Limits",
            description="Tests multi-hop reasoning and logical depth limits"
        )
    
    def generate_test_cases(self, num_cases: int = 10, **kwargs) -> List[TestCase]:
        """
        Generate reasoning tasks with varying depth.
        
        Tasks:
        - Transitive relations (if A>B and B>C, then A>C)
        - Chain inference (requires N steps to reach conclusion)
        - Math word problems (varying complexity)
        """
        test_cases = []
        
        # Test 1: Transitive relations with increasing depth
        for depth in range(2, 8):  # 2 to 7 hops
            chain, question, answer = self._generate_transitive_chain(depth)
            test_cases.append(TestCase(
                id=f"transitive_depth_{depth}",
                input=f"{chain}\n\nQuestion: {question}",
                expected_output=answer,
                metadata={"reasoning_depth": depth, "task_type": "transitive"}
            ))
        
        # Test 2: Multi-step arithmetic
        for depth in range(2, 6):  # 2 to 5 operations
            problem, answer = self._generate_arithmetic_chain(depth)
            test_cases.append(TestCase(
                id=f"arithmetic_depth_{depth}",
                input=f"Solve step by step: {problem}",
                expected_output=str(answer),
                metadata={"reasoning_depth": depth, "task_type": "arithmetic"}
            ))
        
        # Test 3: Logical deduction
        for depth in range(2, 5):
            premises, conclusion = self._generate_logical_deduction(depth)
            test_cases.append(TestCase(
                id=f"logical_depth_{depth}",
                input=f"{premises}\n\nWhat can we conclude?",
                expected_output=conclusion,
                metadata={"reasoning_depth": depth, "task_type": "logical"}
            ))
        
        self.test_cases = test_cases
        return test_cases
    
    def _generate_transitive_chain(self, depth: int) -> Tuple[str, str, str]:
        """Generate transitive relation chain (A > B > C > ...)"""
        # Generate random names
        names = random.sample([
            "Alice", "Bob", "Carol", "David", "Eve", "Frank", 
            "Grace", "Henry", "Iris", "Jack"
        ], depth + 1)
        
        # Build chain
        relations = []
        for i in range(depth):
            relations.append(f"{names[i]} is taller than {names[i+1]}")
        
        chain = ". ".join(relations) + "."
        question = f"Who is taller, {names[0]} or {names[-1]}?"
        answer = names[0]
        
        return chain, question, answer
    
    def _generate_arithmetic_chain(self, depth: int) -> Tuple[str, int]:
        """Generate multi-step arithmetic problem."""
        start = random.randint(10, 50)
        operations = []
        current = start
        
        for _ in range(depth):
            op = random.choice(["+", "-", "*"])
            operand = random.randint(2, 10)
            
            if op == "+":
                operations.append(f"add {operand}")
                current += operand
            elif op == "-":
                operations.append(f"subtract {operand}")
                current -= operand
            else:  # multiplication
                operations.append(f"multiply by {operand}")
                current *= operand
        
        problem = f"Start with {start}. "
        problem += ", then ".join(operations)
        problem += ". What is the final result?"
        
        return problem, current
    
    def _generate_logical_deduction(self, depth: int) -> Tuple[str, str]:
        """Generate logical deduction task."""
        # Simple pattern: A→B, B→C, C→D, ... therefore A→D
        statements = []
        items = random.sample([
            "dogs", "mammals", "animals", "living things",
            "cats", "pets", "creatures", "beings"
        ], depth + 1)
        
        for i in range(depth):
            statements.append(f"All {items[i]} are {items[i+1]}")
        
        premises = ". ".join(statements) + "."
        conclusion = f"All {items[0]} are {items[-1]}"
        
        return premises, conclusion
    
    def evaluate_single(
        self,
        test_case: TestCase,
        prediction: str,
        response: LLMResponse
    ) -> TestResult:
        """Evaluate reasoning correctness."""
        # Extract answer from prediction (might contain explanation)
        prediction_clean = prediction.strip().lower()
        expected_clean = test_case.expected_output.strip().lower()
        
        # Check if expected answer is in prediction
        success = expected_clean in prediction_clean
        
        # For arithmetic, be more strict
        if test_case.metadata["task_type"] == "arithmetic":
            # Extract numbers from prediction
            import re
            numbers = re.findall(r'\b\d+\b', prediction)
            success = test_case.expected_output in numbers
        
        metrics = {
            "correct": 1.0 if success else 0.0,
            "reasoning_depth": test_case.metadata["reasoning_depth"],
            "task_type": test_case.metadata["task_type"],
            "latency_ms": response.latency_ms if response else 0
        }
        
        return TestResult(
            test_case_id=test_case.id,
            prediction=prediction,
            reference=test_case.expected_output,
            success=success,
            metrics=metrics,
            response=response
        )
    
    def get_diagnostic_insights(self) -> Dict[str, Any]:
        """Analyze reasoning depth limitations."""
        if not self.results:
            return {"error": "No results available"}
        
        # Group by depth
        by_depth = {}
        by_task_type = {}
        
        for result in self.results:
            depth = result.metrics.get("reasoning_depth", 0)
            task_type = result.metrics.get("task_type", "unknown")
            
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(result.metrics.get("correct", 0.0))
            
            if task_type not in by_task_type:
                by_task_type[task_type] = []
            by_task_type[task_type].append(result.metrics.get("correct", 0.0))
        
        # Calculate success rates
        depth_analysis = {
            depth: sum(scores) / len(scores)
            for depth, scores in by_depth.items()
        }
        
        task_analysis = {
            task: sum(scores) / len(scores)
            for task, scores in by_task_type.items()
        }
        
        # Find critical depth (where success drops below 70%)
        critical_depth = None
        for depth in sorted(by_depth.keys()):
            if depth_analysis[depth] < 0.7:
                critical_depth = depth
                break
        
        # Recommendations
        recommendations = []
        
        if critical_depth and critical_depth <= 3:
            recommendations.append(
                f"Model struggles with reasoning beyond {critical_depth} steps. "
                "Consider breaking complex tasks into simpler sub-tasks or using "
                "chain-of-thought prompting."
            )
        
        if task_analysis.get("arithmetic", 1.0) < 0.8:
            recommendations.append(
                "Arithmetic reasoning is weak. Consider using external calculators "
                "or specialized math tools via function calling."
            )
        
        if task_analysis.get("logical", 1.0) < 0.8:
            recommendations.append(
                "Logical deduction is unreliable. For critical logic tasks, "
                "consider using symbolic reasoning systems or formal verification."
            )
        
        return {
            "bottleneck_identified": critical_depth is not None,
            "critical_reasoning_depth": critical_depth,
            "success_by_depth": depth_analysis,
            "success_by_task_type": task_analysis,
            "recommendations": recommendations,
            "theoretical_explanation": (
                "Transformers process sequences token by token without explicit "
                "symbolic reasoning mechanisms. Multi-hop reasoning requires "
                "maintaining and manipulating abstract representations, which "
                "becomes unreliable beyond a certain depth."
            )
        }


# Example usage
if __name__ == "__main__":
    from ..core.llm_client import get_llm_client
    
    # Initialize test
    test = ReasoningDepthTest()
    test.generate_test_cases()
    
    # Run with GPT-4
    client = get_llm_client("gpt-4-turbo-preview")
    results = test.run_test(client, max_tokens=200, temperature=0.0)
    
    # Get diagnostics
    insights = test.get_diagnostic_insights()
    print(insights)
    
    # Save results
    test.save_results("reasoning_depth_results.json")