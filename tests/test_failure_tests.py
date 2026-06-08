"""Smoke tests for the diagnostic failure tests.

Each test class should be able to generate cases, run against a mock client,
and produce diagnostic insights -- all offline.
"""

import pytest

from llm_diagnostic.failure_tests.context_limits import ContextLimitsTest
from llm_diagnostic.failure_tests.hallucination_patterns import HallucinationPatternsTest
from llm_diagnostic.failure_tests.knowledge_boundaries import KnowledgeBoundariesTest
from llm_diagnostic.failure_tests.reasoning_depth import ReasoningDepthTest
from llm_diagnostic.failure_tests.structure_validation import StructureValidationTest

TEST_CLASSES = [
    ContextLimitsTest,
    ReasoningDepthTest,
    KnowledgeBoundariesTest,
    StructureValidationTest,
    HallucinationPatternsTest,
]


@pytest.mark.parametrize("test_cls", TEST_CLASSES)
def test_generate_test_cases(test_cls):
    test = test_cls()
    cases = test.generate_test_cases(num_cases=6)
    assert len(cases) > 0
    assert all(c.input for c in cases)


@pytest.mark.parametrize("test_cls", TEST_CLASSES)
def test_run_and_aggregate(test_cls, mock_llm_client):
    test = test_cls()
    test.generate_test_cases(num_cases=6)
    results = test.run_test(mock_llm_client, verbose=False)
    assert len(results) == len(test.test_cases)

    aggregated = test.aggregate_results()
    assert "error" not in aggregated

    insights = test.get_diagnostic_insights()
    assert isinstance(insights, dict)
