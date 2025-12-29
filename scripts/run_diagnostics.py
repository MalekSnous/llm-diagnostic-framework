#!/usr/bin/env python3
"""
CLI script to run LLM diagnostic tests.
Usage: python scripts/run_diagnostics.py --model gpt-4-turbo-preview --test all
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_diagnostic.core.llm_client import get_llm_client
from llm_diagnostic.failure_tests.context_limits import ContextLimitsTest
from llm_diagnostic.failure_tests.reasoning_depth import ReasoningDepthTest
from llm_diagnostic.failure_tests.knowledge_boundaries import KnowledgeBoundariesTest
from llm_diagnostic.failure_tests.structure_validation import StructureValidationTest
from llm_diagnostic.failure_tests.hallucination_patterns import HallucinationPatternsTest
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def run_single_test(test_class, test_name, llm_client, num_cases=10, output_dir="./results/diagnostics"):
    """Run a single diagnostic test."""
    console.print(f"\n[bold cyan]Running {test_name}...[/bold cyan]")
    
    # Initialize test
    test = test_class()
    
    # Generate test cases
    console.print(f"Generating {num_cases} test cases...")
    test.generate_test_cases(num_cases=num_cases)
    
    # Run test
    results = test.run_test(llm_client, verbose=True)
    
    # Get insights
    insights = test.get_diagnostic_insights()
    aggregated = test.aggregate_results()
    
    # Display results
    display_test_results(test_name, aggregated, insights)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{test_name.lower().replace(' ', '_')}_{timestamp}.json"
    
    test.save_results(str(filename))
    console.print(f"[green]Results saved to {filename}[/green]")
    
    return aggregated, insights


def display_test_results(test_name, aggregated, insights):
    """Display test results in formatted table."""
    # Create results table
    table = Table(title=f"{test_name} Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add aggregated metrics
    table.add_row("Success Rate", f"{aggregated['success_rate']:.1%}")
    table.add_row("Success Count", str(aggregated['success_count']))
    table.add_row("Failure Count", str(aggregated['failure_count']))
    table.add_row("Total Cost", f"${aggregated['total_cost_usd']:.4f}")
    table.add_row("Avg Latency", f"{aggregated['avg_latency_ms']:.0f}ms")
    
    console.print(table)
    
    # Display insights
    if "recommendations" in insights and insights["recommendations"]:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for i, rec in enumerate(insights["recommendations"], 1):
            console.print(f"  {i}. {rec}")
    
    # Display theoretical explanation
    if "theoretical_explanation" in insights:
        panel = Panel(
            insights["theoretical_explanation"],
            title="[bold]Theoretical Explanation[/bold]",
            border_style="blue"
        )
        console.print(panel)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM diagnostic tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with GPT-4
  python scripts/run_diagnostics.py --model gpt-4-turbo-preview --test all
  
  # Run specific test
  python scripts/run_diagnostics.py --model gpt-4-turbo-preview --test context
  
  # Run with custom number of cases
  python scripts/run_diagnostics.py --model gpt-4-turbo-preview --test all --num-cases 20
  
  # Run with Claude
  python scripts/run_diagnostics.py --model claude-3-sonnet-20240229 --test all
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model to test (e.g., gpt-4-turbo-preview, claude-3-sonnet-20240229)"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "context", "reasoning", "knowledge", "structure", "hallucination"],
        help="Which test(s) to run"
    )
    
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of test cases per test (default: 10)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/diagnostics",
        help="Output directory for results (default: ./results/diagnostics)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max tokens for LLM responses (default: 500)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # Print header
    console.print(Panel.fit(
        "[bold cyan]LLM Diagnostic Framework[/bold cyan]\n"
        f"Model: {args.model}\n"
        f"Test(s): {args.test}\n"
        f"Cases per test: {args.num_cases}",
        border_style="cyan"
    ))
    
    # Initialize LLM client
    console.print(f"\n[bold]Initializing LLM client: {args.model}[/bold]")
    try:
        llm_client = get_llm_client(args.model)
    except Exception as e:
        console.print(f"[bold red]Error initializing LLM client: {e}[/bold red]")
        console.print("[yellow]Make sure your API keys are set in .env file[/yellow]")
        sys.exit(1)
    
    # Define test mapping
    tests = {
        "context": (ContextLimitsTest, "Context Window Limits"),
        "reasoning": (ReasoningDepthTest, "Reasoning Depth Limits"),
        "knowledge": (KnowledgeBoundariesTest, "Knowledge Boundaries"),
        "structure": (StructureValidationTest, "Structure Validation"),
        "hallucination": (HallucinationPatternsTest, "Hallucination Patterns")
    }
    
    # Determine which tests to run
    if args.test == "all":
        tests_to_run = tests.items()
    else:
        tests_to_run = [(args.test, tests[args.test])]
    
    # Run tests
    all_results = {}
    
    for test_key, (test_class, test_name) in tests_to_run:
        try:
            aggregated, insights = run_single_test(
                test_class=test_class,
                test_name=test_name,
                llm_client=llm_client,
                num_cases=args.num_cases,
                output_dir=args.output_dir
            )
            
            all_results[test_name] = {
                "aggregated": aggregated,
                "insights": insights
            }
            
        except Exception as e:
            console.print(f"[bold red]Error running {test_name}: {e}[/bold red]")
            import traceback
            traceback.print_exc()
    
    # Summary
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]           DIAGNOSTIC SUMMARY          [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    summary_table = Table(show_header=True)
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Success Rate", style="green")
    summary_table.add_column("Cost", style="yellow")
    summary_table.add_column("Bottleneck?", style="red")
    
    for test_name, result in all_results.items():
        agg = result["aggregated"]
        insights = result["insights"]
        
        bottleneck = "Yes" if insights.get("bottleneck_identified", False) else "No"
        
        summary_table.add_row(
            test_name,
            f"{agg['success_rate']:.1%}",
            f"${agg['total_cost_usd']:.4f}",
            bottleneck
        )
    
    console.print(summary_table)
    
    # Overall recommendations
    console.print("\n[bold yellow]Overall Recommendations:[/bold yellow]")
    all_recommendations = []
    for result in all_results.values():
        all_recommendations.extend(result["insights"].get("recommendations", []))
    
    if all_recommendations:
        for i, rec in enumerate(all_recommendations[:5], 1):  # Top 5
            console.print(f"  {i}. {rec}")
    else:
        console.print("  No major issues detected. Model performs well on these tests.")
    
    console.print(f"\n[green]All results saved to {args.output_dir}/[/green]")
    console.print("[bold]✓ Diagnostic complete![/bold]")


if __name__ == "__main__":
    main()