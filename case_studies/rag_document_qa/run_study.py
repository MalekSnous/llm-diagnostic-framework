#!/usr/bin/env python3
"""
Case study: Document QA with a full RAG pipeline.

100 questions about AcmeCloud, a fictional internal platform whose facts live
only in data/acmecloud_kb/ — no model can know them from training. Three arms:

1. Baseline (closed-book) — the model answers from parametric memory alone;
2. RAG — dense (or TF-IDF) retrieval over the indexed knowledge base;
3. RAG + rerank — wider first-stage retrieval, cross-encoder reranking.

Metrics: answer accuracy (boundary-aware matching against accepted answers,
so "50" can't be satisfied by "500"), per-difficulty breakdown, retrieval
recall@k against gold source documents, and real token cost. The expert tier
includes unanswerable questions where the correct behaviour is to abstain.

Usage:
    python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini
    python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini --embedder tfidf --limit 20
"""

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table

from llm_diagnostic.core.llm_client import get_llm_client
from llm_diagnostic.failure_tests.base_test import TestCase, TestResult
from llm_diagnostic.rag import RAGPipeline, get_embedder, load_directory
from llm_diagnostic.utils.case_study_reporter import CaseStudyReporter

# Unique module name: several case studies ship a `dataset.py` (see the
# text_to_sql study for why a bare import would collide in sys.modules).
_spec = importlib.util.spec_from_file_location(
    "rag_qa_dataset", Path(__file__).parent / "dataset.py"
)
_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dataset)
CASES, TIERS = _dataset.CASES, _dataset.TIERS

console = Console()

KB_PATH = Path(__file__).parent.parent.parent / "data" / "acmecloud_kb"

BASELINE_PROMPT = (
    "Answer this question about the AcmeCloud platform concisely (one short sentence). "
    "If you do not know the answer, reply exactly: I don't know.\n\n"
    "Question: {question}\n\nAnswer:"
)

TOP_K = 4
FETCH_K = 12  # first-stage candidates handed to the reranker

# An accepted answer that is purely numeric (optionally $/€-prefixed) is matched
# with digit boundaries, so "50" is not satisfied by "500", "3.50" or "50,000".
_NUMERIC = re.compile(r"^[$€]?\d[\d.,]*$")


def _matches(prediction_lower: str, accepted: str) -> bool:
    accepted = accepted.lower()
    if _NUMERIC.match(accepted):
        number = re.escape(accepted.lstrip("$€"))
        return (
            re.search(rf"(?<!\d)(?<!\d[.,]){number}(?!\d)(?![.,]\d)", prediction_lower) is not None
        )
    return accepted in prediction_lower


def is_correct(prediction: str, case: dict) -> bool:
    """A prediction is correct if any accepted answer matches."""
    prediction_lower = prediction.lower()
    return any(_matches(prediction_lower, ans) for ans in case["accepted"])


def create_test_cases(cases) -> list:
    return [
        TestCase(
            id=f"qa_{i}",
            input=case["question"],
            expected_output=case["accepted"][0],
            metadata={"task": "document_qa", **case},
        )
        for i, case in enumerate(cases)
    ]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def difficulty_breakdown(results, cases) -> dict:
    tiers = {}
    for r in results:
        case = cases[int(r.test_case_id.split("_")[1])]
        tiers.setdefault(case["difficulty"], []).append(r.metrics["accuracy"])
    return {t: _mean(v) for t in TIERS if (v := tiers.get(t))}


def _cost(results) -> float:
    return sum(r.response.cost_usd for r in results if r.response and r.response.cost_usd)


def _summarise(label, results, cases, extra="") -> None:
    acc = _mean(r.metrics["accuracy"] for r in results)
    breakdown = "  ".join(f"{t}={a:.0%}" for t, a in difficulty_breakdown(results, cases).items())
    console.print(f"[green]{label}: accuracy {acc:.1%} | cost ${_cost(results):.4f}{extra}[/green]")
    console.print(f"[dim]  by difficulty: {breakdown}[/dim]")


def run_baseline(test_cases, cases, llm_client):
    """Arm 1 — closed-book: no retrieval, the model answers from memory."""
    console.print("\n[bold cyan]Step 1: Baseline (closed-book, no retrieval)[/bold cyan]")
    results = []
    for tc in test_cases:
        response = llm_client.generate(BASELINE_PROMPT.format(question=tc.input), max_tokens=100)
        case = cases[int(tc.id.split("_")[1])]
        correct = is_correct(response.text, case)
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
    _summarise("Baseline", results, cases)
    return results


def run_rag_arm(label, pipeline, test_cases, cases, llm_client):
    """One retrieval arm: answer every question through the RAG pipeline."""
    results = []
    recalls = []
    for tc in test_cases:
        case = cases[int(tc.id.split("_")[1])]
        answer = pipeline.answer(tc.input, llm_client, max_tokens=100)
        correct = is_correct(answer.text, case)

        metrics = {"accuracy": 1.0 if correct else 0.0}
        if case["sources"]:  # retrieval recall only defined for answerable cases
            hit = any(doc_id in case["sources"] for doc_id in answer.sources)
            metrics["retrieval_recall"] = 1.0 if hit else 0.0
            recalls.append(metrics["retrieval_recall"])

        results.append(
            TestResult(
                test_case_id=tc.id,
                prediction=answer.text,
                reference=tc.expected_output,
                success=correct,
                metrics=metrics,
                response=answer.response,
            )
        )
    recall = _mean(recalls)
    _summarise(label, results, cases, extra=f" | retrieval recall@{pipeline.top_k} {recall:.1%}")
    return results, recall


def build_pipelines(embedder_kind, with_rerank):
    """Index the knowledge base once; return (rag_pipeline, rerank_pipeline|None)."""
    console.print("\n[bold cyan]Indexing knowledge base[/bold cyan]")
    documents = load_directory(KB_PATH)
    embedder = get_embedder(embedder_kind)
    pipeline = RAGPipeline(embedder=embedder, top_k=TOP_K)
    n_chunks = pipeline.index(documents)
    console.print(
        f"[green]Indexed {len(documents)} documents → {n_chunks} chunks "
        f"(embedder: {embedder.name})[/green]"
    )

    rerank_pipeline = None
    if with_rerank:
        try:
            from llm_diagnostic.rag import CrossEncoderReranker

            rerank_pipeline = RAGPipeline(
                embedder=embedder,
                store=pipeline.store,
                reranker=CrossEncoderReranker(),
                top_k=TOP_K,
                fetch_k=FETCH_K,
            )
        except ImportError:
            console.print(
                "[yellow]sentence-transformers not installed — skipping the rerank arm "
                '(pip install -e ".[rag]").[/yellow]'
            )
    return pipeline, rerank_pipeline


def generate_summary(arms):
    console.print("\n[bold cyan]══════════ FINAL SUMMARY ══════════[/bold cyan]\n")
    table = Table(title="Document QA — closed-book vs RAG", show_header=True)
    table.add_column("Arm", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Δ vs baseline", style="magenta")
    table.add_column("Recall@k", style="blue")
    table.add_column("Cost", style="yellow")

    baseline_acc = arms[0][1]["accuracy"]
    for label, stats in arms:
        delta = (
            "-"
            if label == "Baseline (closed-book)"
            else (f"{(stats['accuracy'] - baseline_acc) * 100:+.1f}%")
        )
        recall = f"{stats['retrieval_recall']:.1%}" if "retrieval_recall" in stats else "—"
        table.add_row(label, f"{stats['accuracy']:.1%}", delta, recall, f"${stats['cost']:.4f}")
    console.print(table)


def _sample(cases, limit):
    """Evenly sample `limit` cases so every difficulty tier stays represented."""
    if not limit or limit >= len(cases):
        return cases
    step = len(cases) / limit
    return [cases[int(i * step)] for i in range(limit)]


def main():
    parser = argparse.ArgumentParser(description="RAG document-QA case study")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to benchmark")
    parser.add_argument(
        "--embedder",
        default="auto",
        choices=["auto", "dense", "tfidf"],
        help="Embedding backend (auto = dense if installed, else TF-IDF)",
    )
    parser.add_argument("--no-rerank", action="store_true", help="Skip the reranking arm")
    parser.add_argument("--limit", type=int, default=0, help="Run only N cases (smoke test)")
    parser.add_argument("--no-report", action="store_true", help="Skip JSON/HTML report")
    args = parser.parse_args()

    # Hosted models need an API key; fail fast with a clear message.
    key_var = None
    if args.model.startswith("gpt-"):
        key_var = "OPENAI_API_KEY"
    elif args.model.startswith("claude-"):
        key_var = "ANTHROPIC_API_KEY"
    elif args.model.startswith("groq/"):
        key_var = "GROQ_API_KEY"
    if key_var and not os.getenv(key_var):
        console.print(f"[red]✗ {key_var} not set (required for {args.model}).[/red]")
        return

    console.print("[bold]Document QA Case Study — closed-book vs RAG vs RAG+rerank[/bold]")
    console.print(f"Model: {args.model}\n")

    cases = _sample(CASES, args.limit)
    test_cases = create_test_cases(cases)
    console.print(f"[green]{len(test_cases)} questions ({', '.join(TIERS)} tiers)[/green]")
    llm_client = get_llm_client(args.model)

    pipeline, rerank_pipeline = build_pipelines(args.embedder, with_rerank=not args.no_rerank)

    baseline_results = run_baseline(test_cases, cases, llm_client)

    console.print("\n[bold cyan]Step 2: RAG (retrieval, no reranking)[/bold cyan]")
    rag_results, rag_recall = run_rag_arm(
        f"RAG (top-{TOP_K})", pipeline, test_cases, cases, llm_client
    )

    rerank_stats = None
    if rerank_pipeline is not None:
        console.print("\n[bold cyan]Step 3: RAG + cross-encoder reranking[/bold cyan]")
        rerank_results, rerank_recall = run_rag_arm(
            "RAG + rerank", rerank_pipeline, test_cases, cases, llm_client
        )
        rerank_stats = {
            "accuracy": _mean(r.metrics["accuracy"] for r in rerank_results),
            "cost": _cost(rerank_results),
            "by_difficulty": difficulty_breakdown(rerank_results, cases),
            "retrieval_recall": rerank_recall,
        }

    baseline = {
        "accuracy": _mean(r.metrics["accuracy"] for r in baseline_results),
        "cost": _cost(baseline_results),
        "by_difficulty": difficulty_breakdown(baseline_results, cases),
    }
    rag_stats = {
        "accuracy": _mean(r.metrics["accuracy"] for r in rag_results),
        "cost": _cost(rag_results),
        "by_difficulty": difficulty_breakdown(rag_results, cases),
        "retrieval_recall": rag_recall,
        "embedder": pipeline.embedder.name,
    }

    arms = [("Baseline (closed-book)", baseline), (f"RAG (top-{TOP_K})", rag_stats)]
    if rerank_stats:
        arms.append(("RAG + rerank", rerank_stats))
    generate_summary(arms)

    if not args.no_report:
        improvements = {f"RAG (top-{TOP_K})": rag_stats}
        if rerank_stats:
            improvements["RAG + rerank"] = rerank_stats
        reporter = CaseStudyReporter("rag_document_qa")
        json_path = reporter.save_results(args.model, baseline, improvements)
        html_path = reporter.generate_html_report(args.model, baseline, improvements)
        console.print(f"\n[green]✓ JSON saved: {json_path}[/green]")
        console.print(f"[green]✓ HTML report: {html_path}[/green]")


if __name__ == "__main__":
    main()
