#!/usr/bin/env python3
"""
Case study: Document Question Answering with RAG.

Contrast with the medical case study (where generic RAG *hurt* a strong model):
here the questions are about a private product document (AcmeCloud) whose facts
are NOT in any model's training data. A baseline (closed-book) model therefore
has to guess, while a RAG-augmented model can retrieve the answer. This is the
canonical case where retrieval pays off, and the study quantifies the gain in
accuracy and cost.

Usage:
    python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini
    python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini --no-report
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table

from llm_diagnostic.core.llm_client import get_llm_client
from llm_diagnostic.failure_tests.base_test import TestCase, TestResult
from llm_diagnostic.improvements.rag_system import RAGSystem
from llm_diagnostic.utils.case_study_reporter import CaseStudyReporter

console = Console()

DOCS_PATH = Path(__file__).parent.parent.parent / "data" / "product_docs.txt"

# (question, list of acceptable answer substrings). A prediction is correct if it
# contains any of the accepted strings (case-insensitive).
QA_PAIRS = [
    ("How many concurrent pipelines does the AcmeCloud Growth plan allow?", ["50"]),
    ("After how many days does a default AcmeCloud AKT token expire?", ["14"]),
    (
        "What is the default region for new AcmeCloud accounts created after January 2025?",
        ["eu-paris-1"],
    ),
    ("How much is each additional ACU billed at on the Growth plan?", ["0.08", "$0.08"]),
    ("What is the Enterprise severity-1 support response time?", ["1 hour", "1-hour", "one hour"]),
    ("How long are pipeline logs retained on the Enterprise plan?", ["400"]),
    ("What minimum Python version does the acmecloud-sdk require?", ["3.10"]),
    ("What is the maximum webhook payload size in AcmeCloud?", ["256"]),
]


def load_documents() -> list:
    """Load the product documentation as a list of paragraph chunks."""
    text = DOCS_PATH.read_text(encoding="utf-8")
    # Split on blank lines into reasonably sized chunks.
    chunks = [block.strip() for block in text.split("\n\n") if block.strip()]
    return chunks


def create_test_cases() -> list:
    """Build QA test cases. expected_output is the primary accepted answer."""
    return [
        TestCase(
            id=f"qa_{i}",
            input=f"Answer the question concisely. Question: {question}",
            expected_output=answers[0],
            metadata={"accepted": answers, "task": "document_qa"},
        )
        for i, (question, answers) in enumerate(QA_PAIRS)
    ]


def _is_correct(prediction: str, test_case: TestCase) -> bool:
    pred = prediction.lower()
    return any(ans.lower() in pred for ans in test_case.metadata["accepted"])


def run_baseline(test_cases, llm_client):
    """Closed-book baseline: ask the question with no document context."""
    console.print("\n[bold cyan]Step 1: Baseline (closed-book, no retrieval)[/bold cyan]")

    results = []
    for tc in test_cases:
        response = llm_client.generate(tc.input, max_tokens=100)
        correct = _is_correct(response.text, tc)
        results.append(
            TestResult(
                test_case_id=tc.id,
                prediction=response.text,
                reference=tc.expected_output,
                success=correct,
                metrics={"accuracy": 1.0 if correct else 0.0},
                response=response,
            )
        )

    accuracy = sum(r.metrics["accuracy"] for r in results) / len(results)
    cost = sum(r.response.cost_usd for r in results if r.response and r.response.cost_usd)
    console.print(f"[yellow]Baseline accuracy: {accuracy:.1%} | cost: ${cost:.4f}[/yellow]")
    return results, accuracy, cost


def run_rag(test_cases, baseline_results, llm_client, documents):
    """RAG: retrieve from the product docs before answering."""
    console.print("\n[bold cyan]Step 2: RAG (retrieval over product docs)[/bold cyan]")

    strategy = RAGSystem()
    config = strategy.configure(top_k=3, chunk_size=300)
    rag_results = strategy.apply(
        test_cases=test_cases,
        baseline_results=baseline_results,
        config=config,
        llm_client=llm_client,
        knowledge_base=documents,
    )

    # Re-score with the substring-match criterion used for the baseline.
    correct = sum(
        1 for r in rag_results if _is_correct(r.prediction, _by_id(test_cases, r.test_case_id))
    )
    accuracy = correct / len(rag_results) if rag_results else 0.0
    cost = sum(r.response.cost_usd for r in rag_results if r.response and r.response.cost_usd)
    console.print(f"[green]RAG accuracy: {accuracy:.1%} | cost: ${cost:.4f}[/green]")
    return rag_results, accuracy, cost


def _by_id(test_cases, tc_id):
    return next(tc for tc in test_cases if tc.id == tc_id)


def generate_summary(baseline_acc, baseline_cost, rag_acc, rag_cost):
    console.print("\n[bold cyan]══════════ FINAL SUMMARY ══════════[/bold cyan]\n")
    table = Table(title="Document QA: Baseline vs RAG", show_header=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Δ vs baseline", style="magenta")
    table.add_column("Cost", style="yellow")

    table.add_row("Baseline (closed-book)", f"{baseline_acc:.1%}", "-", f"${baseline_cost:.4f}")
    delta = (rag_acc - baseline_acc) * 100
    table.add_row("RAG", f"{rag_acc:.1%}", f"{delta:+.1f}%", f"${rag_cost:.4f}")
    console.print(table)

    if rag_acc > baseline_acc:
        console.print(
            "\n[bold green]✓ RAG wins:[/bold green] the answers live in a private document "
            "the model was never trained on, so retrieval is what unlocks accuracy."
        )
    else:
        console.print(
            "\n[bold yellow]RAG did not help here — inspect retrieval quality.[/bold yellow]"
        )


def main():
    parser = argparse.ArgumentParser(description="RAG document-QA case study")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to benchmark")
    parser.add_argument("--no-report", action="store_true", help="Skip JSON/HTML report")
    args = parser.parse_args()

    console.print("[bold]Document QA Case Study — Baseline vs RAG[/bold]")
    console.print(f"Model: {args.model}\n")

    documents = load_documents()
    test_cases = create_test_cases()
    llm_client = get_llm_client(args.model)

    baseline_results, baseline_acc, baseline_cost = run_baseline(test_cases, llm_client)
    _, rag_acc, rag_cost = run_rag(test_cases, baseline_results, llm_client, documents)
    generate_summary(baseline_acc, baseline_cost, rag_acc, rag_cost)

    if not args.no_report:
        reporter = CaseStudyReporter("rag_document_qa")
        baseline = {"accuracy": baseline_acc, "cost": baseline_cost}
        improvements = {"RAG System": {"accuracy": rag_acc, "cost": rag_cost}}
        json_path = reporter.save_results(args.model, baseline, improvements)
        html_path = reporter.generate_html_report(args.model, baseline, improvements)
        console.print(f"\n[green]Saved:[/green] {json_path}\n[green]Report:[/green] {html_path}")


if __name__ == "__main__":
    main()
