#!/usr/bin/env python3
"""
CLI script to run improvement strategies and compare with baseline.
Usage: python scripts/run_improvements.py --strategy prompt --baseline-file results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_diagnostic.core.llm_client import get_llm_client
from llm_diagnostic.improvements.prompt_engineering import PromptEngineeringStrategy
from llm_diagnostic.improvements.rag_system import RAGSystem
from llm_diagnostic.failure_tests.base_test import TestCase, TestResult
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_test_cases_from_diagnostic(diagnostic_file: str) -> tuple:
    """Load test cases and baseline results from diagnostic output."""
    with open(diagnostic_file, 'r') as f:
        data = json.load(f)
    
    # Extract test cases from diagnostic results
    test_cases = []
    baseline_results = []
    
    for detail in data.get("detailed_results", []):
        # Reconstruct TestCase
        test_case = TestCase(
            id=detail["test_case_id"],
            input=detail.get("input", ""),  # May not be saved
            expected_output=detail["reference"]
        )
        test_cases.append(test_case)
        
        # Reconstruct TestResult (simplified)
        baseline_result = TestResult(
            test_case_id=detail["test_case_id"],
            prediction=detail["prediction"],
            reference=detail["reference"],
            success=detail["success"],
            metrics=detail["metrics"],
            response=None  # Can't reconstruct full response
        )
        baseline_results.append(baseline_result)
    
    return test_cases, baseline_results


def run_prompt_engineering(test_cases, baseline_results, llm_client, technique="few-shot"):
    """Run prompt engineering improvement."""
    console.print(f"\n[bold cyan]Running Prompt Engineering ({technique})...[/bold cyan]")
    
    strategy = PromptEngineeringStrategy()
    config = strategy.configure(technique=technique)
    
    improved_results = strategy.apply(
        test_cases=test_cases,
        baseline_results=baseline_results,
        config=config,
        llm_client=llm_client
    )

    # # 🔍 DEBUG : Voir ce qui a été généré
    # console.print("\n[yellow]DEBUG: Checking improved_results...[/yellow]")
    # for i, result in enumerate(improved_results[:2]):  # Juste les 2 premiers
    #     console.print(f"[yellow]  Result {i}:[/yellow]")
    #     console.print(f"    Prediction: {result.prediction[:100]}...")
    #     console.print(f"    Expected: {EXPECTED_ENTITIES[i]}")
    #     entities_found = sum(1 for e in EXPECTED_ENTITIES[i] if e.lower() in result.prediction.lower())
    #     console.print(f"    Entities found: {entities_found}/{len(EXPECTED_ENTITIES[i])}")
    
    
    return strategy.results[0] if strategy.results else None


def run_rag_system(test_cases, baseline_results, llm_client, knowledge_base):
    """Run RAG system improvement."""
    console.print(f"\n[bold cyan]Running RAG System...[/bold cyan]")
    
    strategy = RAGSystem()
    config = strategy.configure(top_k=3)
    
    improved_results = strategy.apply(
        test_cases=test_cases,
        baseline_results=baseline_results,
        config=config,
        llm_client=llm_client,
        knowledge_base=knowledge_base
    )
    
    return strategy.results[0] if strategy.results else None


def display_improvement_results(result, strategy_name):
    """Display improvement results."""
    console.print(f"\n[bold green]✓ {strategy_name} Complete![/bold green]")
    
    # Create comparison table
    table = Table(title="Baseline vs Improved", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="yellow")
    table.add_column("Improved", style="green")
    table.add_column("Delta", style="magenta")
    
    # Compare metrics
    for metric_name in result.baseline_metrics.metrics:
        baseline_val = result.baseline_metrics.metrics[metric_name]
        improved_val = result.improved_metrics.metrics.get(metric_name, 0)
        delta = result.improvement_delta.get(metric_name, 0)
        
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        
        table.add_row(
            metric_name.capitalize(),
            f"{baseline_val:.3f}",
            f"{improved_val:.3f}",
            delta_str
        )
    
    console.print(table)
    
    # Cost-benefit analysis
    console.print("\n[bold]Cost-Benefit Analysis:[/bold]")
    cost_benefit = {
        "Total Cost": f"${result.total_cost:.4f}",
        "Time": f"{result.total_time_seconds:.1f}s",
        "Accuracy Improvement": f"{result.improvement_delta.get('accuracy', 0) * 100:.1f}%"
    }
    
    for key, value in cost_benefit.items():
        console.print(f"  • {key}: {value}")


def compare_strategies(results_dict):
    """Compare multiple improvement strategies."""
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]       STRATEGY COMPARISON           [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    table = Table(show_header=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("Accuracy Δ", style="green")
    table.add_column("Cost", style="yellow")
    table.add_column("Time", style="magenta")
    table.add_column("Cost/Point", style="red")
    
    for strategy_name, result in results_dict.items():
        acc_delta = result.improvement_delta.get("accuracy", 0) * 100
        cost = result.total_cost
        time = result.total_time_seconds
        
        cost_per_point = cost / acc_delta if acc_delta > 0 else float('inf')
        cost_per_point_str = f"${cost_per_point:.2f}" if cost_per_point != float('inf') else "N/A"
        
        table.add_row(
            strategy_name,
            f"+{acc_delta:.1f}%",
            f"${cost:.4f}",
            f"{time:.1f}s",
            cost_per_point_str
        )
    
    console.print(table)
    
    # Recommendation
    if results_dict:
        best_strategy = min(
            results_dict.items(),
            key=lambda x: x[1].total_cost / (x[1].improvement_delta.get("accuracy", 0.001) * 100)
        )
        
        console.print(f"\n[bold green]Recommended Strategy: {best_strategy[0]}[/bold green]")
        console.print(f"  Best cost-effectiveness for this task")


def main():
    parser = argparse.ArgumentParser(
        description="Run improvement strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run prompt engineering
  python scripts/run_improvements.py --strategy prompt --baseline diagnostic_results.json
  
  # Run RAG system
  python scripts/run_improvements.py --strategy rag --baseline diagnostic_results.json --kb knowledge_base.txt
  
  # Compare all strategies
  python scripts/run_improvements.py --strategy all --baseline diagnostic_results.json
        """
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["prompt", "rag", "fine-tune", "all"],
        help="Improvement strategy to use"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline diagnostic results JSON file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo-preview",
        help="LLM model to use (default: gpt-4-turbo-preview)"
    )
    
    parser.add_argument(
        "--kb",
        type=str,
        help="Knowledge base file (for RAG strategy)"
    )
    
    parser.add_argument(
        "--technique",
        type=str,
        default="few-shot",
        choices=["few-shot", "chain-of-thought", "structured", "combined"],
        help="Prompting technique (for prompt strategy)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/improvements",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Print header
    console.print(Panel.fit(
        "[bold cyan]LLM Improvement Framework[/bold cyan]\n"
        f"Strategy: {args.strategy}\n"
        f"Model: {args.model}\n"
        f"Baseline: {args.baseline}",
        border_style="cyan"
    ))
    
    # Load baseline data
    console.print(f"\n[bold]Loading baseline data from {args.baseline}[/bold]")
    try:
        test_cases, baseline_results = load_test_cases_from_diagnostic(args.baseline)
        console.print(f"[green]Loaded {len(test_cases)} test cases[/green]")
    except Exception as e:
        console.print(f"[bold red]Error loading baseline: {e}[/bold red]")
        sys.exit(1)
    
    # Initialize LLM client
    console.print(f"\n[bold]Initializing LLM client: {args.model}[/bold]")
    try:
        llm_client = get_llm_client(args.model)
    except Exception as e:
        console.print(f"[bold red]Error initializing LLM client: {e}[/bold red]")
        sys.exit(1)
    
    # Run improvement strategies
    results_dict = {}
    
    if args.strategy == "prompt" or args.strategy == "all":
        result = run_prompt_engineering(
            test_cases, baseline_results, llm_client, technique=args.technique
        )
        if result:
            results_dict["Prompt Engineering"] = result
            display_improvement_results(result, "Prompt Engineering")
    
    if args.strategy == "rag" or args.strategy == "all":
        if args.kb:
            # Load knowledge base
            with open(args.kb, 'r') as f:
                knowledge_base = [line.strip() for line in f if line.strip()]
            
            result = run_rag_system(
                test_cases, baseline_results, llm_client, knowledge_base
            )
            if result:
                results_dict["RAG System"] = result
                display_improvement_results(result, "RAG System")
        else:
            console.print("[yellow]Warning: --kb required for RAG strategy. Skipping.[/yellow]")
    
    if args.strategy == "fine-tune":
        console.print("[yellow]Fine-tuning requires separate setup. Use fine_tuning.py directly.[/yellow]")
    
    # Compare strategies if multiple were run
    if len(results_dict) > 1:
        compare_strategies(results_dict)
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  


    if results_dict:
        # Sauvegarder chaque résultat de stratégie
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for strategy_name, result in results_dict.items():
            # Créer un nom de fichier propre
            safe_strategy_name = strategy_name.lower().replace(" ", "_")
            filename = output_path / f"{safe_strategy_name}_{timestamp}.json"
            
            # Préparer les données à sauvegarder
            result_data = {
                "strategy_name": result.strategy_name,
                "model": args.model,
                "baseline_file": args.baseline,
                "timestamp": timestamp,
                "config": {
                    "parameters": result.config.parameters,
                    "estimated_cost": result.config.estimated_cost,
                    "estimated_time": result.config.estimated_time
                },
                "baseline_metrics": result.baseline_metrics.metrics,
                "improved_metrics": result.improved_metrics.metrics,
                "improvement_delta": result.improvement_delta,
                "total_cost": result.total_cost,
                "total_time_seconds": result.total_time_seconds,
                "cost_benefit_analysis": {
                    "accuracy_improvement": result.improvement_delta.get("accuracy", 0) * 100,
                    "cost_per_point": result.total_cost / (result.improvement_delta.get("accuracy", 0.001) * 100)
                }
            }
            
            # Sauvegarder en JSON
            import json
            with open(filename, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            console.print(f"[green]✓ Saved: {filename}[/green]")
    else:
        console.print("[yellow]⚠ No results to save[/yellow]")
    
    console.print("[bold]✓ Improvement analysis complete![/bold]")

    
    # console.print(f"\n[green]Results saved to {output_path}/[/green]")
    # console.print("[bold]✓ Improvement analysis complete![/bold]")


if __name__ == "__main__":
    main()