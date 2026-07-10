#!/usr/bin/env python3
"""
Case study: Text-to-SQL with execution accuracy.

Natural-language questions over a small e-commerce SQLite schema. The model's
SQL is *executed* against the same seeded database as the gold query and the
result sets are compared — so the headline metric (execution accuracy) cannot
be inflated by verbosity, formatting, or partial matches. The query either
computes the right answer or it doesn't.

RAG is deliberately not tested here: the full schema fits in the prompt, so
there is nothing to retrieve — the interesting comparison is zero-shot vs
few-shot prompting (which teaches SQL patterns like anti-joins and
"only the requested columns").
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table

from llm_diagnostic.core.evaluator import EvaluationMetrics, Evaluator
from llm_diagnostic.core.llm_client import get_llm_client
from llm_diagnostic.failure_tests.base_test import TestCase, TestResult
from llm_diagnostic.improvements.base_strategy import ImprovementResult
from llm_diagnostic.improvements.prompt_engineering import PromptEngineeringStrategy
from llm_diagnostic.utils.case_study_reporter import CaseStudyReporter

# Load the local dataset by explicit path under a unique module name: several
# case studies ship a `dataset.py`, so a bare `from dataset import ...` would
# collide in sys.modules when more than one study is loaded (e.g. in tests).
_spec = importlib.util.spec_from_file_location(
    "text_to_sql_dataset", Path(__file__).parent / "dataset.py"
)
_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dataset)
CASES, SCHEMA_SQL, build_db = _dataset.CASES, _dataset.SCHEMA_SQL, _dataset.build_db

console = Console()

QUESTIONS = [c["question"] for c in CASES]
GOLD_SQL = [c["gold_sql"] for c in CASES]
DIFFICULTIES = [c["difficulty"] for c in CASES]

PROMPT_TEMPLATE = (
    "You are given this SQLite database schema:\n"
    "{schema}\n"
    "Write a single SQLite SELECT query that answers the question below.\n"
    "Return only the SQL query — no explanation, no markdown. Select only the "
    "columns the question asks for.\n\n"
    "Question: {question}"
)

# Task-specific few-shot examples for the prompt-engineering strategy: they
# teach the output contract (bare SQL, only requested columns) and the
# patterns models most often get wrong (anti-join, aggregate-over-join).
FEW_SHOT_EXAMPLES = [
    "Example question: How many orders are there?\n" "Example answer: SELECT COUNT(*) FROM orders",
    "Example question: List the names of products in the 'home' category.\n"
    "Example answer: SELECT name FROM products WHERE category = 'home'",
    "Example question: What is the total value of each order? Return the order id and the total.\n"
    "Example answer: SELECT order_id, SUM(quantity * unit_price) FROM order_items GROUP BY order_id",
    "Example question: List the names of customers who never placed an order.\n"
    "Example answer: SELECT name FROM customers "
    "WHERE id NOT IN (SELECT DISTINCT customer_id FROM orders)",
]


def execute_sql(sql):
    """Run one SELECT/WITH statement on a fresh seeded DB.

    Returns (rows, error). Non-SELECT statements are rejected up front; the
    database is in-memory and rebuilt per call, so a rogue query cannot
    corrupt anything anyway.
    """
    if not sql or not sql.lstrip().lower().startswith(("select", "with")):
        return None, "not a SELECT query"
    conn = build_db()
    try:
        return conn.execute(sql).fetchall(), None
    except Exception as exc:  # sqlite3.Error, sqlite3.Warning (multi-statement), ...
        return None, str(exc)
    finally:
        conn.close()


def normalize_rows(rows):
    """Order-insensitive comparison form: sorted rows, floats rounded to 2dp."""
    normalized = [tuple(round(v, 2) if isinstance(v, float) else v for v in row) for row in rows]
    return sorted(normalized, key=repr)


def score(prediction_text, gold_sql):
    """Score one prediction by execution accuracy against the gold query."""
    sql = Evaluator.parse_sql_query(prediction_text)
    predicted_rows, error = execute_sql(sql)
    if error is not None:
        return {"accuracy": 0.0, "valid_sql": 0.0}
    gold_rows, gold_error = execute_sql(gold_sql)
    assert gold_error is None, f"gold query failed: {gold_sql!r} -> {gold_error}"
    match = normalize_rows(predicted_rows) == normalize_rows(gold_rows)
    return {"accuracy": 1.0 if match else 0.0, "valid_sql": 1.0}


def _mean(results, key):
    return sum(r.metrics[key] for r in results) / len(results) if results else 0.0


def create_test_cases():
    """Create test cases for text-to-SQL."""
    schema = SCHEMA_SQL.strip()
    return [
        TestCase(
            id=f"sql_{i}",
            input=PROMPT_TEMPLATE.format(schema=schema, question=question),
            expected_output=gold_sql,
            context=question,
            metadata={"task": "text_to_sql", "domain": "e-commerce", "difficulty": difficulty},
        )
        for i, (question, gold_sql, difficulty) in enumerate(zip(QUESTIONS, GOLD_SQL, DIFFICULTIES))
    ]


def difficulty_breakdown(results):
    """Mean execution accuracy per difficulty tier."""
    tiers = {}
    for r in results:
        idx = int(r.test_case_id.split("_")[1])
        tiers.setdefault(DIFFICULTIES[idx], []).append(r.metrics["accuracy"])
    order = ["easy", "medium", "hard", "expert"]
    return {t: sum(v) / len(v) for t in order if (v := tiers.get(t))}


def run_baseline(test_cases, llm_client):
    """Run baseline: zero-shot with the schema in the prompt."""
    console.print("\n[bold cyan]Step 1: Running Baseline (Zero-shot)[/bold cyan]")

    baseline_results = []
    for test_case in test_cases:
        response = llm_client.generate(test_case.input, max_tokens=300)
        s = score(response.text, GOLD_SQL[int(test_case.id.split("_")[1])])
        baseline_results.append(
            TestResult(
                test_case_id=test_case.id,
                prediction=response.text,
                reference=test_case.expected_output,
                success=s["accuracy"] > 0.5,
                metrics=s,
                response=response,
            )
        )

    avg_acc = _mean(baseline_results, "accuracy")
    valid = _mean(baseline_results, "valid_sql")
    total_cost = sum(r.response.cost_usd for r in baseline_results if r.response.cost_usd)

    console.print(
        f"[yellow]Baseline execution accuracy: {avg_acc:.1%} (valid SQL: {valid:.1%})[/yellow]"
    )
    console.print(f"[yellow]Baseline Cost: ${total_cost:.4f}[/yellow]")

    by_diff = difficulty_breakdown(baseline_results)
    if by_diff:
        breakdown = "  ".join(f"{tier}={acc:.0%}" for tier, acc in by_diff.items())
        console.print(f"[dim]Baseline accuracy by difficulty: {breakdown}[/dim]")

    return baseline_results, avg_acc, total_cost


def run_prompt_engineering(test_cases, baseline_results, llm_client):
    """Run with few-shot prompt engineering (task-specific SQL examples)."""
    console.print("\n[bold cyan]Step 2: Applying Prompt Engineering (few-shot)[/bold cyan]")

    strategy = PromptEngineeringStrategy()
    config = strategy.configure(
        technique="few-shot",
        examples=FEW_SHOT_EXAMPLES,
        num_examples=len(FEW_SHOT_EXAMPLES),
    )

    improved_results = strategy.apply(
        test_cases=test_cases,
        baseline_results=baseline_results,
        config=config,
        llm_client=llm_client,
        max_tokens=300,
    )

    # Re-score with the same execution-accuracy metric used for the baseline.
    for result in improved_results:
        result.metrics = score(result.prediction, GOLD_SQL[int(result.test_case_id.split("_")[1])])

    improved_accuracy = _mean(improved_results, "accuracy")
    baseline_accuracy = _mean(baseline_results, "accuracy")
    improvement = (improved_accuracy - baseline_accuracy) * 100

    total_cost = sum(
        r.response.cost_usd for r in improved_results if r.response and r.response.cost_usd
    )

    console.print(
        f"[green]Improved execution accuracy: {improved_accuracy:.1%} ({improvement:+.1f}%) "
        f"(valid SQL: {_mean(improved_results, 'valid_sql'):.1%})[/green]"
    )
    console.print(f"[green]Cost: ${total_cost:.4f}[/green]")

    baseline_metrics = EvaluationMetrics()
    baseline_metrics.add_metric("accuracy", baseline_accuracy)

    improved_metrics = EvaluationMetrics()
    improved_metrics.add_metric("accuracy", improved_accuracy)
    improved_metrics.add_metric("valid_sql", _mean(improved_results, "valid_sql"))
    improved_metrics.add_details("by_difficulty", difficulty_breakdown(improved_results))

    return ImprovementResult(
        strategy_name="Prompt Engineering",
        config=config,
        baseline_metrics=baseline_metrics,
        improved_metrics=improved_metrics,
        improvement_delta={"accuracy": improved_accuracy - baseline_accuracy},
        total_cost=total_cost,
        total_time_seconds=0.0,
    )


def generate_summary(baseline_acc, baseline_cost, prompt_result):
    """Print the final summary table."""
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]     CASE STUDY: FINAL SUMMARY       [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

    table = Table(show_header=True, title="Text-to-SQL Results (execution accuracy)")
    table.add_column("Strategy", style="cyan")
    table.add_column("Exec. accuracy", style="green")
    table.add_column("Improvement", style="magenta")
    table.add_column("Cost", style="yellow")

    table.add_row("Baseline (Zero-shot)", f"{baseline_acc:.1%}", "-", f"${baseline_cost:.4f}")

    prompt_acc = prompt_result.improved_metrics.metrics.get("accuracy", 0)
    table.add_row(
        "Prompt Engineering",
        f"{prompt_acc:.1%}",
        f"{(prompt_acc - baseline_acc) * 100:+.1f}%",
        f"${prompt_result.total_cost:.4f}",
    )
    console.print(table)


def main():
    console.print("[bold cyan]╔════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   Text-to-SQL Case Study (execution)      ║[/bold cyan]")
    console.print("[bold cyan]╚════════════════════════════════════════════╝[/bold cyan]")

    import argparse
    import os

    parser = argparse.ArgumentParser(description="Text-to-SQL case study")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to benchmark")
    args = parser.parse_args()
    model_name = args.model

    # Hosted models need an API key; local HF models don't.
    key_var = None
    if model_name.startswith("gpt-"):
        key_var = "OPENAI_API_KEY"
    elif model_name.startswith("claude-"):
        key_var = "ANTHROPIC_API_KEY"
    elif model_name.startswith("groq/"):
        key_var = "GROQ_API_KEY"
    if key_var and not os.getenv(key_var):
        console.print(f"[red]✗ {key_var} not set (required for {model_name}).[/red]")
        return

    console.print("\n[bold]Initializing...[/bold]")
    test_cases = create_test_cases()
    console.print(f"[green]Created {len(test_cases)} test cases[/green]")

    llm_client = get_llm_client(model_name)
    console.print(f"[green]Initialized LLM client: {model_name}[/green]")

    baseline_results, baseline_acc, baseline_cost = run_baseline(test_cases, llm_client)
    prompt_result = run_prompt_engineering(test_cases, baseline_results, llm_client)

    generate_summary(baseline_acc, baseline_cost, prompt_result)

    console.print("\n[bold cyan]Saving results...[/bold cyan]")
    reporter = CaseStudyReporter("text_to_sql")
    baseline = {
        "accuracy": baseline_acc,
        "valid_sql": _mean(baseline_results, "valid_sql"),
        "cost": baseline_cost,
        "by_difficulty": difficulty_breakdown(baseline_results),
    }
    improvements = {
        "Prompt Engineering": {
            "accuracy": prompt_result.improved_metrics.metrics.get("accuracy", 0),
            "valid_sql": prompt_result.improved_metrics.metrics.get("valid_sql", 0),
            "cost": prompt_result.total_cost,
            "by_difficulty": prompt_result.improved_metrics.details.get("by_difficulty", {}),
        }
    }
    json_path = reporter.save_results(model_name, baseline, improvements)
    html_path = reporter.generate_html_report(model_name, baseline, improvements)

    console.print(f"[green]✓ JSON saved: {json_path}[/green]")
    console.print(f"[green]✓ HTML report: {html_path}[/green]")

    console.print("\n[bold green]✓ Case study complete![/bold green]")
    console.print("\n[yellow]📊 Open the HTML report to view full results:[/yellow]")
    console.print(f"[yellow]   file://{html_path.absolute()}[/yellow]")


if __name__ == "__main__":
    main()
