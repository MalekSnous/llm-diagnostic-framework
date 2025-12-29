"""
Base class for improvement strategies.
All improvement strategies (prompting, fine-tuning, RAG, etc.) inherit from this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.evaluator import EvaluationMetrics


@dataclass
class ImprovementConfig:
    """Configuration for an improvement strategy."""
    strategy_name: str
    parameters: Dict[str, Any]
    estimated_cost: Optional[float] = None
    estimated_time: Optional[str] = None


@dataclass
class ImprovementResult:
    """Result of applying an improvement strategy."""
    strategy_name: str
    config: ImprovementConfig
    baseline_metrics: EvaluationMetrics
    improved_metrics: EvaluationMetrics
    improvement_delta: Dict[str, float]
    total_cost: float
    total_time_seconds: float
    metadata: Optional[Dict[str, Any]] = None


class BaseImprovementStrategy(ABC):
    """
    Abstract base class for improvement strategies.
    
    Each strategy implements:
    1. Configuration and setup
    2. Application to test cases
    3. Evaluation and comparison
    4. Cost-benefit analysis
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results: List[ImprovementResult] = []
    
    @abstractmethod
    def configure(self, **kwargs) -> ImprovementConfig:
        """
        Configure the improvement strategy.
        
        Returns:
            ImprovementConfig with strategy parameters
        """
        pass
    
    @abstractmethod
    def apply(
        self,
        test_cases: List[Any],
        baseline_results: List[Any],
        config: ImprovementConfig,
        **kwargs
    ) -> List[Any]:
        """
        Apply the improvement strategy to test cases.
        
        Args:
            test_cases: List of test cases to improve
            baseline_results: Baseline results for comparison
            config: Strategy configuration
            **kwargs: Additional parameters
            
        Returns:
            List of improved results
        """
        pass
    
    def evaluate_improvement(
        self,
        baseline_metrics: EvaluationMetrics,
        improved_metrics: EvaluationMetrics
    ) -> Dict[str, float]:
        """
        Calculate improvement delta.
        
        Args:
            baseline_metrics: Metrics before improvement
            improved_metrics: Metrics after improvement
            
        Returns:
            Dictionary of metric improvements (positive = better)
        """
        delta = {}
        
        for metric_name in baseline_metrics.metrics:
            if metric_name in improved_metrics.metrics:
                baseline_value = baseline_metrics.metrics[metric_name]
                improved_value = improved_metrics.metrics[metric_name]
                delta[metric_name] = improved_value - baseline_value
        
        return delta
    
    def cost_benefit_analysis(
        self,
        improvement_delta: Dict[str, float],
        total_cost: float,
        key_metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Analyze cost-benefit trade-offs.
        
        Args:
            improvement_delta: Improvement in metrics
            total_cost: Total cost in USD
            key_metric: Primary metric to optimize
            
        Returns:
            Cost-benefit analysis results
        """
        key_improvement = improvement_delta.get(key_metric, 0.0)
        
        # Cost per 1% improvement
        if key_improvement > 0:
            cost_per_point = total_cost / (key_improvement * 100)
        else:
            cost_per_point = float('inf')
        
        # ROI calculation (simplified)
        # Assume 1% accuracy improvement = $1000 value (arbitrary, adjust per use case)
        value_per_point = 1000
        estimated_value = key_improvement * 100 * value_per_point
        roi = (estimated_value - total_cost) / total_cost if total_cost > 0 else float('inf')
        
        return {
            "key_metric": key_metric,
            "improvement_percentage": key_improvement * 100,
            "total_cost_usd": total_cost,
            "cost_per_point_improvement": cost_per_point,
            "estimated_value_usd": estimated_value,
            "roi": roi,
            "recommendation": self._generate_recommendation(cost_per_point, roi)
        }
    
    def _generate_recommendation(self, cost_per_point: float, roi: float) -> str:
        """Generate recommendation based on cost-benefit."""
        if roi > 5:
            return "Highly recommended: Excellent ROI"
        elif roi > 1:
            return "Recommended: Positive ROI"
        elif roi > 0:
            return "Marginal: Slightly positive ROI"
        elif cost_per_point < 100:
            return "Consider: Low cost, acceptable ROI"
        else:
            return "Not recommended: Poor cost-benefit ratio"
    
    def compare_strategies(
        self,
        other_strategies: List['BaseImprovementStrategy']
    ) -> Dict[str, Any]:
        """
        Compare this strategy with others.
        
        Args:
            other_strategies: List of other strategies to compare
            
        Returns:
            Comparative analysis
        """
        all_strategies = [self] + other_strategies
        
        comparison = {
            "strategies": [],
            "ranking": []
        }
        
        for strategy in all_strategies:
            if not strategy.results:
                continue
            
            # Aggregate results
            avg_improvement = sum(
                r.improvement_delta.get("accuracy", 0) 
                for r in strategy.results
            ) / len(strategy.results)
            
            avg_cost = sum(r.total_cost for r in strategy.results) / len(strategy.results)
            
            comparison["strategies"].append({
                "name": strategy.name,
                "avg_improvement": avg_improvement,
                "avg_cost": avg_cost,
                "cost_per_point": avg_cost / (avg_improvement * 100) if avg_improvement > 0 else float('inf')
            })
        
        # Rank by cost-effectiveness
        comparison["ranking"] = sorted(
            comparison["strategies"],
            key=lambda x: x["cost_per_point"]
        )
        
        return comparison
    
    @abstractmethod
    def get_implementation_guide(self) -> Dict[str, Any]:
        """
        Provide implementation guide for this strategy.
        
        Returns:
            Dictionary with implementation details, code examples, etc.
        """
        pass
    
    def save_results(self, filepath: str):
        """Save improvement results to JSON."""
        import json
        from datetime import datetime
        
        data = {
            "strategy_name": self.name,
            "description": self.description,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "config": {
                        "strategy_name": r.config.strategy_name,
                        "parameters": r.config.parameters,
                        "estimated_cost": r.config.estimated_cost
                    },
                    "baseline_metrics": r.baseline_metrics.to_dict(),
                    "improved_metrics": r.improved_metrics.to_dict(),
                    "improvement_delta": r.improvement_delta,
                    "total_cost": r.total_cost,
                    "total_time_seconds": r.total_time_seconds
                }
                for r in self.results
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', num_results={len(self.results)})"