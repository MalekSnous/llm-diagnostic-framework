"""
Evaluation metrics for LLM diagnostics.
Provides both standard metrics (accuracy, F1) and LLM-specific metrics (hallucination, faithfulness).
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class EvaluationMetrics:
    """Container for evaluation metrics."""

    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.details: Dict[str, Any] = {}

    def add_metric(self, name: str, value: float, details: Optional[Any] = None):
        """Add a metric."""
        self.metrics[name] = value
        if details is not None:
            self.details[name] = details

    def add_details(self, name: str, details: Any) -> None:
        """Attach auxiliary details that are not a scalar metric."""
        self.details[name] = details

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"metrics": self.metrics, "details": self.details}

    def __repr__(self) -> str:
        lines = ["Evaluation Metrics:"]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


class Evaluator:
    """Main evaluator class with various metric calculations."""

    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy."""
        matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        return matches / len(predictions)

    @staticmethod
    def classification_metrics(
        predictions: List[str], references: List[str], labels: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """Calculate classification metrics (accuracy, precision, recall, F1)."""
        metrics = EvaluationMetrics()

        # Exact match accuracy
        accuracy = Evaluator.exact_match(predictions, references)
        metrics.add_metric("accuracy", accuracy)

        # Convert to numeric labels if needed
        if labels is None:
            unique_labels = sorted(set(predictions + references))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            pred_numeric = [label_to_idx.get(p, -1) for p in predictions]
            ref_numeric = [label_to_idx[r] for r in references]
        else:
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            pred_numeric = [label_to_idx.get(p, -1) for p in predictions]
            ref_numeric = [label_to_idx[r] for r in references]

        # Calculate metrics
        metrics.add_metric(
            "precision",
            precision_score(ref_numeric, pred_numeric, average="weighted", zero_division=0),
        )
        metrics.add_metric(
            "recall", recall_score(ref_numeric, pred_numeric, average="weighted", zero_division=0)
        )
        metrics.add_metric(
            "f1", f1_score(ref_numeric, pred_numeric, average="weighted", zero_division=0)
        )

        return metrics

    @staticmethod
    def entity_extraction_metrics(
        predictions: List[List[str]], references: List[List[str]]
    ) -> EvaluationMetrics:
        """Calculate entity extraction metrics (precision, recall, F1)."""
        metrics = EvaluationMetrics()

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_entities, ref_entities in zip(predictions, references):
            pred_set = set(pred_entities)
            ref_set = set(ref_entities)

            true_positives += len(pred_set & ref_set)
            false_positives += len(pred_set - ref_set)
            false_negatives += len(ref_set - pred_set)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics.add_metric("precision", precision)
        metrics.add_metric("recall", recall)
        metrics.add_metric("f1", f1)
        metrics.add_metric("true_positives", true_positives)
        metrics.add_metric("false_positives", false_positives)
        metrics.add_metric("false_negatives", false_negatives)

        return metrics

    @staticmethod
    def structure_validation_metrics(
        predictions: List[str], expected_format: str = "json"
    ) -> EvaluationMetrics:
        """Validate structured outputs (JSON, XML, etc.)."""
        metrics = EvaluationMetrics()

        valid_count = 0
        parse_errors = []

        for i, pred in enumerate(predictions):
            try:
                if expected_format == "json":
                    json.loads(pred)
                    valid_count += 1
                # Add other formats as needed
            except Exception as e:
                parse_errors.append(
                    {"index": i, "error": str(e), "output": pred[:200]}  # First 200 chars
                )

        metrics.add_metric("parse_success_rate", valid_count / len(predictions))
        metrics.add_metric("valid_outputs", valid_count)
        metrics.add_metric("invalid_outputs", len(predictions) - valid_count)
        metrics.add_details("parse_errors", parse_errors[:10])  # Store first 10 errors

        return metrics

    @staticmethod
    def hallucination_detection(
        predictions: List[str], contexts: List[str], use_heuristics: bool = True
    ) -> EvaluationMetrics:
        """Detect hallucinations in model outputs."""
        metrics = EvaluationMetrics()

        hallucination_indicators = []

        for pred, context in zip(predictions, contexts):
            indicators = {
                "hedging_phrases": Evaluator._detect_hedging(pred),
                "unsupported_facts": Evaluator._check_facts_in_context(pred, context),
                "contradiction": Evaluator._detect_contradictions(pred, context),
            }
            hallucination_indicators.append(indicators)

        # Calculate aggregated metrics
        avg_hedging = np.mean([ind["hedging_phrases"] for ind in hallucination_indicators])
        avg_unsupported = np.mean([ind["unsupported_facts"] for ind in hallucination_indicators])
        avg_contradiction = np.mean([ind["contradiction"] for ind in hallucination_indicators])

        metrics.add_metric("hedging_rate", avg_hedging)
        metrics.add_metric("unsupported_facts_rate", avg_unsupported)
        metrics.add_metric("contradiction_rate", avg_contradiction)
        metrics.add_metric("hallucination_risk", (avg_unsupported + avg_contradiction) / 2)

        return metrics

    @staticmethod
    def _detect_hedging(text: str) -> float:
        """Detect hedging phrases that might indicate uncertainty."""
        hedging_phrases = [
            "might",
            "could",
            "possibly",
            "perhaps",
            "maybe",
            "i think",
            "i believe",
            "it seems",
            "appears to",
        ]
        text_lower = text.lower()
        count = sum(phrase in text_lower for phrase in hedging_phrases)
        return min(count / 5, 1.0)  # Normalize to [0, 1]

    @staticmethod
    def _check_facts_in_context(prediction: str, context: str) -> float:
        """Check if facts in prediction are supported by context."""
        # Simple heuristic: extract entities and check if they appear in context
        # This is a simplified version; production would use NER
        pred_words = set(prediction.lower().split())
        context_words = set(context.lower().split())

        # Filter common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        pred_words -= stopwords
        context_words -= stopwords

        if not pred_words:
            return 0.0

        unsupported_ratio = len(pred_words - context_words) / len(pred_words)
        return min(unsupported_ratio, 1.0)

    @staticmethod
    def _detect_contradictions(prediction: str, context: str) -> float:
        """Detect potential contradictions between prediction and context."""
        # Simple heuristic: look for negation patterns
        # Production version would use NLI models

        negation_words = ["not", "no", "never", "none", "neither", "nor"]

        pred_negations = sum(word in prediction.lower().split() for word in negation_words)
        context_negations = sum(word in context.lower().split() for word in negation_words)

        # If one has significantly more negations, flag potential contradiction
        if pred_negations + context_negations == 0:
            return 0.0

        neg_diff = abs(pred_negations - context_negations)
        return min(neg_diff / 5, 1.0)

    @staticmethod
    def cost_analysis(
        responses: List[Any],  # List of LLMResponse objects
    ) -> EvaluationMetrics:
        """Analyze cost metrics."""
        metrics = EvaluationMetrics()

        total_cost = sum(r.cost_usd for r in responses if r.cost_usd)
        total_tokens = sum(r.tokens_used for r in responses)
        avg_latency = np.mean([r.latency_ms for r in responses])

        metrics.add_metric("total_cost_usd", total_cost)
        metrics.add_metric("total_tokens", total_tokens)
        metrics.add_metric("avg_latency_ms", avg_latency)
        metrics.add_metric("cost_per_request", total_cost / len(responses))

        return metrics


# Example usage
if __name__ == "__main__":
    # Test classification metrics
    predictions = ["positive", "negative", "positive", "neutral"]
    references = ["positive", "positive", "positive", "neutral"]

    metrics = Evaluator.classification_metrics(predictions, references)
    print(metrics)

    # Test entity extraction
    pred_entities = [["Apple", "Google"], ["Microsoft"], ["Tesla", "SpaceX"]]
    ref_entities = [["Apple", "Microsoft"], ["Microsoft"], ["Tesla", "SpaceX", "Neuralink"]]

    metrics = Evaluator.entity_extraction_metrics(pred_entities, ref_entities)
    print(metrics)
