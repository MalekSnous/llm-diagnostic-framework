"""
Structured output validation tests.
Tests LLM ability to generate valid JSON, code, and constrained formats.
"""

import json
import re
from typing import Any, Dict, List

from .base_test import BaseFailureTest, TestCase, TestResult
from ..core.llm_client import LLMResponse


class StructureValidationTest(BaseFailureTest):
    """
    Test LLM's ability to produce structured, valid outputs.
    
    Theory: LLMs generate tokens sequentially without built-in structure
    validation, leading to malformed JSON, invalid code, etc.
    """
    
    def __init__(self):
        super().__init__(
            name="Structure Validation",
            description="Tests ability to generate valid structured outputs"
        )
    
    def generate_test_cases(self, num_cases: int = 10, **kwargs) -> List[TestCase]:
        """Generate test cases for structured output."""
        test_cases = []
        
        # Test 1: JSON generation
        json_cases = self._generate_json_tests()
        test_cases.extend(json_cases)
        
        # Test 2: Code generation
        code_cases = self._generate_code_tests()
        test_cases.extend(code_cases)
        
        # Test 3: Constrained formats
        format_cases = self._generate_format_tests()
        test_cases.extend(format_cases)
        
        self.test_cases = test_cases
        return test_cases
    
    def _generate_json_tests(self) -> List[TestCase]:
        """Generate JSON generation tasks."""
        cases = []
        
        # Simple JSON
        cases.append(TestCase(
            id="json_simple",
            input=(
                "Generate a JSON object with fields: name (string), age (number), "
                "email (string). Use example data."
            ),
            expected_output=None,  # Will validate structure, not content
            metadata={
                "output_type": "json",
                "complexity": "simple",
                "required_fields": ["name", "age", "email"]
            }
        ))
        
        # Nested JSON
        cases.append(TestCase(
            id="json_nested",
            input=(
                "Generate a JSON object representing a person with nested address "
                "(street, city, country) and hobbies (array of strings)."
            ),
            expected_output=None,
            metadata={
                "output_type": "json",
                "complexity": "nested",
                "required_fields": ["address", "hobbies"]
            }
        ))
        
        # Complex JSON array
        cases.append(TestCase(
            id="json_array",
            input=(
                "Generate a JSON array of 3 product objects. Each product should have: "
                "id (number), name (string), price (number), inStock (boolean), "
                "tags (array of strings)."
            ),
            expected_output=None,
            metadata={
                "output_type": "json",
                "complexity": "array",
                "required_fields": ["id", "name", "price", "inStock", "tags"],
                "expected_count": 3
            }
        ))
        
        return cases
    
    def _generate_code_tests(self) -> List[TestCase]:
        """Generate code generation tasks."""
        cases = []
        
        # Simple function
        cases.append(TestCase(
            id="code_function",
            input="Write a Python function that takes a list and returns the sum of even numbers.",
            expected_output=None,
            metadata={
                "output_type": "code",
                "language": "python",
                "complexity": "simple"
            }
        ))
        
        # Class with methods
        cases.append(TestCase(
            id="code_class",
            input=(
                "Write a Python class 'Calculator' with methods: add, subtract, "
                "multiply, divide. Include error handling for division by zero."
            ),
            expected_output=None,
            metadata={
                "output_type": "code",
                "language": "python",
                "complexity": "moderate"
            }
        ))
        
        return cases
    
    def _generate_format_tests(self) -> List[TestCase]:
        """Generate constrained format tasks."""
        cases = []
        
        # Email format
        cases.append(TestCase(
            id="format_email",
            input="Generate 3 valid email addresses in the format: name@company.com",
            expected_output=None,
            metadata={
                "output_type": "format",
                "format_type": "email",
                "expected_count": 3
            }
        ))
        
        # Date format
        cases.append(TestCase(
            id="format_date",
            input="Generate 5 dates in ISO 8601 format (YYYY-MM-DD)",
            expected_output=None,
            metadata={
                "output_type": "format",
                "format_type": "date",
                "expected_count": 5
            }
        ))
        
        # Phone number
        cases.append(TestCase(
            id="format_phone",
            input="Generate 3 US phone numbers in format: (XXX) XXX-XXXX",
            expected_output=None,
            metadata={
                "output_type": "format",
                "format_type": "phone",
                "expected_count": 3
            }
        ))
        
        return cases
    
    def evaluate_single(
        self,
        test_case: TestCase,
        prediction: str,
        response: LLMResponse
    ) -> TestResult:
        """Evaluate structural validity."""
        output_type = test_case.metadata["output_type"]
        
        if output_type == "json":
            return self._evaluate_json(test_case, prediction, response)
        elif output_type == "code":
            return self._evaluate_code(test_case, prediction, response)
        elif output_type == "format":
            return self._evaluate_format(test_case, prediction, response)
        else:
            return TestResult(
                test_case_id=test_case.id,
                prediction=prediction,
                reference=None,
                success=False,
                metrics={},
                response=response,
                error="Unknown output type"
            )
    
    def _evaluate_json(
        self,
        test_case: TestCase,
        prediction: str,
        response: LLMResponse
    ) -> TestResult:
        """Evaluate JSON validity."""
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', prediction, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = prediction
        
        metrics = {
            "output_type": "json",
            "complexity": test_case.metadata["complexity"]
        }
        
        # Try to parse JSON
        try:
            parsed = json.loads(json_str)
            metrics["valid_json"] = 1.0
            
            # Check required fields
            required_fields = test_case.metadata.get("required_fields", [])
            if required_fields:
                # Handle both dict and list
                if isinstance(parsed, list) and len(parsed) > 0:
                    obj = parsed[0]
                else:
                    obj = parsed
                
                if isinstance(obj, dict):
                    fields_present = sum(1 for field in required_fields if field in obj)
                    metrics["field_completeness"] = fields_present / len(required_fields)
                else:
                    metrics["field_completeness"] = 0.0
            else:
                metrics["field_completeness"] = 1.0
            
            # Check expected count (for arrays)
            expected_count = test_case.metadata.get("expected_count")
            if expected_count and isinstance(parsed, list):
                metrics["count_match"] = 1.0 if len(parsed) == expected_count else 0.0
            
            success = metrics["valid_json"] == 1.0 and metrics.get("field_completeness", 1.0) >= 0.8
            
        except json.JSONDecodeError as e:
            metrics["valid_json"] = 0.0
            metrics["parse_error"] = str(e)
            success = False
        
        return TestResult(
            test_case_id=test_case.id,
            prediction=prediction,
            reference=None,
            success=success,
            metrics=metrics,
            response=response
        )
    
    def _evaluate_code(
        self,
        test_case: TestCase,
        prediction: str,
        response: LLMResponse
    ) -> TestResult:
        """Evaluate code validity."""
        # Extract code from markdown
        code_match = re.search(r'```python\s*(.*?)\s*```', prediction, re.DOTALL)
        if code_match:
            code_str = code_match.group(1)
        else:
            code_str = prediction
        
        metrics = {
            "output_type": "code",
            "language": test_case.metadata["language"]
        }
        
        # Try to compile Python code
        try:
            compile(code_str, '<string>', 'exec')
            metrics["valid_syntax"] = 1.0
            success = True
        except SyntaxError as e:
            metrics["valid_syntax"] = 0.0
            metrics["syntax_error"] = str(e)
            success = False
        
        return TestResult(
            test_case_id=test_case.id,
            prediction=prediction,
            reference=None,
            success=success,
            metrics=metrics,
            response=response
        )
    
    def _evaluate_format(
        self,
        test_case: TestCase,
        prediction: str,
        response: LLMResponse
    ) -> TestResult:
        """Evaluate format compliance."""
        format_type = test_case.metadata["format_type"]
        expected_count = test_case.metadata.get("expected_count", 1)
        
        metrics = {
            "output_type": "format",
            "format_type": format_type
        }
        
        # Define regex patterns
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "date": r'\b\d{4}-\d{2}-\d{2}\b',
            "phone": r'\(\d{3}\)\s*\d{3}-\d{4}'
        }
        
        pattern = patterns.get(format_type)
        if not pattern:
            return TestResult(
                test_case_id=test_case.id,
                prediction=prediction,
                reference=None,
                success=False,
                metrics=metrics,
                response=response,
                error=f"Unknown format type: {format_type}"
            )
        
        # Find all matches
        matches = re.findall(pattern, prediction)
        found_count = len(matches)
        
        metrics["found_count"] = found_count
        metrics["expected_count"] = expected_count
        metrics["format_compliance"] = min(found_count / expected_count, 1.0)
        
        success = found_count >= expected_count
        
        return TestResult(
            test_case_id=test_case.id,
            prediction=prediction,
            reference=None,
            success=success,
            metrics=metrics,
            response=response
        )
    
    def get_diagnostic_insights(self) -> Dict[str, Any]:
        """Analyze structure generation patterns."""
        if not self.results:
            return {"error": "No results available"}
        
        # Group by output type
        by_type = {}
        
        for result in self.results:
            output_type = result.metrics.get("output_type", "unknown")
            
            if output_type not in by_type:
                by_type[output_type] = {
                    "success": [],
                    "details": []
                }
            
            by_type[output_type]["success"].append(1.0 if result.success else 0.0)
            by_type[output_type]["details"].append(result.metrics)
        
        # Calculate success rates
        type_analysis = {
            t: sum(data["success"]) / len(data["success"])
            for t, data in by_type.items()
        }
        
        # JSON-specific analysis
        json_results = [r for r in self.results if r.metrics.get("output_type") == "json"]
        if json_results:
            json_valid_rate = sum(r.metrics.get("valid_json", 0) for r in json_results) / len(json_results)
            json_completeness = sum(r.metrics.get("field_completeness", 0) for r in json_results) / len(json_results)
        else:
            json_valid_rate = 0.0
            json_completeness = 0.0
        
        # Recommendations
        recommendations = []
        
        if type_analysis.get("json", 1.0) < 0.8:
            recommendations.append(
                "JSON generation is unreliable. Add explicit formatting instructions, "
                "provide examples, or use output validation + retry logic."
            )
        
        if json_valid_rate < 0.9 and json_valid_rate > 0:
            recommendations.append(
                f"JSON syntax errors in {(1-json_valid_rate)*100:.0f}% of outputs. "
                "Consider using function calling / structured outputs API if available."
            )
        
        if type_analysis.get("code", 1.0) < 0.8:
            recommendations.append(
                "Code generation has syntax errors. Use specialized code models "
                "(e.g., CodeLlama) or add unit test validation."
            )
        
        if type_analysis.get("format", 1.0) < 0.8:
            recommendations.append(
                "Format constraints are not reliably followed. Use regex validation "
                "+ regeneration, or post-process outputs programmatically."
            )
        
        return {
            "success_by_type": type_analysis,
            "json_analysis": {
                "valid_syntax_rate": json_valid_rate,
                "field_completeness": json_completeness
            },
            "recommendations": recommendations,
            "theoretical_explanation": (
                "LLMs generate tokens sequentially without built-in structure validation. "
                "They learn formatting patterns from training data but cannot guarantee "
                "syntactic correctness. Structured output APIs (when available) solve this "
                "by constraining the generation space."
            )
        }


# Example usage
if __name__ == "__main__":
    from ..core.llm_client import get_llm_client
    
    test = StructureValidationTest()
    test.generate_test_cases()
    
    client = get_llm_client("gpt-4-turbo-preview")
    results = test.run_test(client, max_tokens=500)
    
    insights = test.get_diagnostic_insights()
    print(insights)
    
    test.save_results("structure_validation_results.json")