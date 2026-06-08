#!/usr/bin/env python3
"""
Complete case study: Medical Entity Extraction
Demonstrates full diagnostic → improvement → deployment pipeline
WITH AUTOMATIC REPORTING
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table

from llm_diagnostic.core.evaluator import EvaluationMetrics
from llm_diagnostic.core.llm_client import get_llm_client
from llm_diagnostic.failure_tests.base_test import TestCase, TestResult
from llm_diagnostic.improvements.base_strategy import ImprovementResult
from llm_diagnostic.improvements.prompt_engineering import PromptEngineeringStrategy
from llm_diagnostic.improvements.rag_system import RAGSystem
from llm_diagnostic.utils.case_study_reporter import save_case_study_results

console = Console()


# Sample medical data
MEDICAL_TEXTS = [
    "Patient John Doe, 45 years old, presents with hypertension and type 2 diabetes. Prescribed metformin 500mg twice daily.",
    "Jane Smith, female, 32, diagnosed with asthma. Treatment includes albuterol inhaler as needed.",
    "Robert Johnson, 67, admitted with acute myocardial infarction. Emergency angioplasty performed successfully.",
    "Mary Brown, 28, pregnant, gestational diabetes detected. Insulin therapy initiated.",
    "Patient presents with severe migraine headaches. Prescribed sumatriptan 100mg as needed.",
    "Pt c/o CP, h/o MI 2y ago. PMH: HTN, DM2, HLD. Meds: ASA, metoprolol, lisinopril, atorvastatin.",
    "36F G2P1 at 38w presents c/ ROM x 2h. FHR 140s, contractions q3min. Cx 4cm/80%/-1. Plan: EFM, await spontaneous labor.",
    "CXR shows R lower lobe infiltrate. Dx: CAP. Started on Levaquin 750mg qd x 5d. F/u in 1wk or sooner if worsening.",
]

# Expected entities (simplified)
EXPECTED_ENTITIES = [
    ["John Doe", "hypertension", "type 2 diabetes", "metformin"],
    ["Jane Smith", "asthma", "albuterol"],
    ["Robert Johnson", "acute myocardial infarction", "angioplasty"],
    ["Mary Brown", "pregnant", "gestational diabetes", "insulin"],
    ["migraine", "sumatriptan"],
    [
        "chest pain",
        "myocardial infarction",
        "hypertension",
        "diabetes",
        "hyperlipidemia",
        "aspirin",
        "metoprolol",
        "lisinopril",
        "atorvastatin",
    ],
    [
        "gravida 2 para 1",
        "38 weeks",
        "rupture of membranes",
        "fetal heart rate",
        "cervix",
        "electronic fetal monitoring",
    ],
    ["chest x-ray", "right lower lobe", "infiltrate", "community-acquired pneumonia", "Levaquin"],
]

# Medical knowledge base for RAG
MEDICAL_KB = [
    "Metformin is a first-line medication for type 2 diabetes that helps control blood sugar levels.",
    "Hypertension (high blood pressure) is a common condition that increases risk of heart disease.",
    "Albuterol is a bronchodilator used to treat asthma by relaxing airway muscles.",
    "Acute myocardial infarction (heart attack) requires emergency intervention like angioplasty.",
    "Gestational diabetes occurs during pregnancy and requires careful monitoring.",
    "Sumatriptan is a medication used to treat migraine headaches by narrowing blood vessels.",
    "Insulin therapy helps regulate blood sugar in diabetic patients.",
    "Angioplasty is a procedure to open blocked or narrowed blood vessels in the heart.",
    "CP = chest pain",
    "h/o = history of",
    "MI = myocardial infarction",
    "HTN = hypertension",
    "DM2 = diabetes mellitus type 2",
    "HLD = hyperlipidemia",
    "ASA = aspirin",
    "Pt = patient",
    "c/o = complains of",
    "PMH = past medical history",
    "Rx = prescription",
    "Dx = diagnosis",
]


def create_test_cases():
    """Create test cases for medical entity extraction."""
    test_cases = []

    for i, (text, entities) in enumerate(zip(MEDICAL_TEXTS, EXPECTED_ENTITIES)):
        test_cases.append(
            TestCase(
                id=f"medical_{i}",
                input=f"Extract medical entities (conditions, medications, procedures) from: {text}",
                expected_output=", ".join(entities),
                context=text,
                metadata={"task": "entity_extraction", "domain": "medical"},
            )
        )

    return test_cases


def run_baseline(test_cases, llm_client):
    """Run baseline without any improvements."""
    console.print("\n[bold cyan]Step 1: Running Baseline (Zero-shot)[/bold cyan]")

    baseline_results = []

    for test_case in test_cases:
        response = llm_client.generate(test_case.input, max_tokens=200)

        # Simple evaluation: check if expected entities are in response
        prediction = response.text

        # Count how many expected entities were found
        entities_found = sum(
            1
            for entity in EXPECTED_ENTITIES[int(test_case.id.split("_")[1])]
            if entity.lower() in prediction.lower()
        )
        total_entities = len(EXPECTED_ENTITIES[int(test_case.id.split("_")[1])])

        accuracy = entities_found / total_entities if total_entities > 0 else 0

        baseline_results.append(
            TestResult(
                test_case_id=test_case.id,
                prediction=prediction,
                reference=test_case.expected_output,
                success=accuracy > 0.5,
                metrics={
                    "accuracy": accuracy,
                    "entities_found": entities_found,
                    "total_entities": total_entities,
                },
                response=response,
            )
        )

    # Calculate aggregate
    avg_accuracy = sum(r.metrics["accuracy"] for r in baseline_results) / len(baseline_results)
    total_cost = sum(r.response.cost_usd for r in baseline_results if r.response.cost_usd)

    console.print(f"[yellow]Baseline Accuracy: {avg_accuracy:.1%}[/yellow]")
    console.print(f"[yellow]Baseline Cost: ${total_cost:.4f}[/yellow]")

    return baseline_results, avg_accuracy, total_cost


def run_prompt_engineering(test_cases, baseline_results, llm_client):
    """Run with prompt engineering improvements."""
    console.print("\n[bold cyan]Step 2: Applying Prompt Engineering[/bold cyan]")

    strategy = PromptEngineeringStrategy()
    config = strategy.configure(technique="few-shot")

    improved_results = strategy.apply(
        test_cases=test_cases,
        baseline_results=baseline_results,
        config=config,
        llm_client=llm_client,
    )

    # Recompute accuracy from the entity-overlap metric used for the baseline
    total_accuracy = 0
    for i, result in enumerate(improved_results):
        prediction = result.prediction.lower()
        expected_entities = EXPECTED_ENTITIES[i]
        entities_found = sum(1 for entity in expected_entities if entity.lower() in prediction)
        accuracy = entities_found / len(expected_entities) if expected_entities else 0
        total_accuracy += accuracy

    improved_accuracy = total_accuracy / len(improved_results) if improved_results else 0
    baseline_accuracy = sum(r.metrics["accuracy"] for r in baseline_results) / len(baseline_results)
    improvement = (improved_accuracy - baseline_accuracy) * 100

    total_cost = sum(
        r.response.cost_usd for r in improved_results if r.response and r.response.cost_usd
    )

    console.print(
        f"[green]Improved Accuracy: {improved_accuracy:.1%} ({improvement:+.1f}%)[/green]"
    )
    console.print(f"[green]Cost: ${total_cost:.4f}[/green]")

    baseline_metrics = EvaluationMetrics()
    baseline_metrics.add_metric("accuracy", baseline_accuracy)

    improved_metrics = EvaluationMetrics()
    improved_metrics.add_metric("accuracy", improved_accuracy)

    result = ImprovementResult(
        strategy_name="Prompt Engineering",
        config=config,
        baseline_metrics=baseline_metrics,
        improved_metrics=improved_metrics,
        improvement_delta={"accuracy": improved_accuracy - baseline_accuracy},
        total_cost=total_cost,
        total_time_seconds=0.0,
    )

    return result


def run_rag_system(test_cases, baseline_results, llm_client):
    """Run with RAG system."""
    console.print("\n[bold cyan]Step 3: Applying RAG System[/bold cyan]")

    strategy = RAGSystem()
    config = strategy.configure(top_k=2)

    improved_results = strategy.apply(
        test_cases=test_cases,
        baseline_results=baseline_results,
        config=config,
        llm_client=llm_client,
        knowledge_base=MEDICAL_KB,
    )

    # Recompute accuracy from the entity-overlap metric used for the baseline
    total_accuracy = 0
    for i, result in enumerate(improved_results):
        prediction = result.prediction.lower()
        expected_entities = EXPECTED_ENTITIES[i]
        entities_found = sum(1 for entity in expected_entities if entity.lower() in prediction)
        accuracy = entities_found / len(expected_entities) if expected_entities else 0
        total_accuracy += accuracy

    improved_accuracy = total_accuracy / len(improved_results) if improved_results else 0
    baseline_accuracy = sum(r.metrics["accuracy"] for r in baseline_results) / len(baseline_results)
    improvement = (improved_accuracy - baseline_accuracy) * 100

    total_cost = sum(
        r.response.cost_usd for r in improved_results if r.response and r.response.cost_usd
    )

    console.print(
        f"[green]Improved Accuracy: {improved_accuracy:.1%} ({improvement:+.1f}%)[/green]"
    )
    console.print(f"[green]Cost: ${total_cost:.4f}[/green]")

    baseline_metrics = EvaluationMetrics()
    baseline_metrics.add_metric("accuracy", baseline_accuracy)

    improved_metrics = EvaluationMetrics()
    improved_metrics.add_metric("accuracy", improved_accuracy)

    result = ImprovementResult(
        strategy_name="RAG System",
        config=config,
        baseline_metrics=baseline_metrics,
        improved_metrics=improved_metrics,
        improvement_delta={"accuracy": improved_accuracy - baseline_accuracy},
        total_cost=total_cost,
        total_time_seconds=0.0,
    )

    return result


def generate_summary(baseline_acc, baseline_cost, prompt_result, rag_result):
    """Generate final summary table."""
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]     CASE STUDY: FINAL SUMMARY       [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")

    table = Table(show_header=True, title="Medical Entity Extraction Results")
    table.add_column("Strategy", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Improvement", style="magenta")
    table.add_column("Cost", style="yellow")
    table.add_column("Cost/Point", style="red")

    # Baseline
    table.add_row("Baseline (Zero-shot)", f"{baseline_acc:.1%}", "-", f"${baseline_cost:.4f}", "-")

    # Prompt Engineering
    prompt_acc = prompt_result.improved_metrics.metrics.get("accuracy", 0)
    prompt_improvement = (prompt_acc - baseline_acc) * 100
    prompt_cost_per_point = (
        prompt_result.total_cost / prompt_improvement if prompt_improvement > 0 else float("inf")
    )

    table.add_row(
        "Prompt Engineering",
        f"{prompt_acc:.1%}",
        f"+{prompt_improvement:.1f}%",
        f"${prompt_result.total_cost:.4f}",
        f"${prompt_cost_per_point:.2f}" if prompt_cost_per_point != float("inf") else "N/A",
    )

    # RAG System
    rag_acc = rag_result.improved_metrics.metrics.get("accuracy", 0)
    rag_improvement = (rag_acc - baseline_acc) * 100
    rag_cost_per_point = (
        rag_result.total_cost / rag_improvement if rag_improvement > 0 else float("inf")
    )

    table.add_row(
        "RAG System",
        f"{rag_acc:.1%}",
        f"+{rag_improvement:.1f}%",
        f"${rag_result.total_cost:.4f}",
        f"${rag_cost_per_point:.2f}" if rag_cost_per_point != float("inf") else "N/A",
    )

    console.print(table)

    # Recommendation
    best_strategy = "RAG System" if rag_acc > prompt_acc else "Prompt Engineering"
    console.print(f"\n[bold green]✓ Recommended: {best_strategy}[/bold green]")

    if best_strategy == "RAG System":
        console.print("  RAG provides domain knowledge augmentation, critical for medical accuracy")
    else:
        console.print("  Prompt engineering sufficient for this task, lower cost")


def main():
    console.print("[bold cyan]╔════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   Medical Entity Extraction Case Study   ║[/bold cyan]")
    console.print("[bold cyan]╚════════════════════════════════════════════╝[/bold cyan]")

    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        console.print(f"[green]✓ API Key configured: {api_key[:20]}...[/green]")
    else:
        console.print("[red]✗ No API Key found![/red]")
        return

    # Initialize
    console.print("\n[bold]Initializing...[/bold]")
    test_cases = create_test_cases()
    console.print(f"[green]Created {len(test_cases)} test cases[/green]")

    # MODIFIER ICI : Changer le nom du modèle
    # model_name = "microsoft/phi-2"
    model_name = "gpt-4o-mini"
    # model_name = 'gpt-4o'

    llm_client = get_llm_client(model_name)
    console.print(f"[green]Initialized LLM client: {model_name}[/green]")

    # Run baseline
    baseline_results, baseline_acc, baseline_cost = run_baseline(test_cases, llm_client)

    # Run improvements
    prompt_result = run_prompt_engineering(test_cases, baseline_results, llm_client)
    rag_result = run_rag_system(test_cases, baseline_results, llm_client)

    # Generate summary
    generate_summary(baseline_acc, baseline_cost, prompt_result, rag_result)

    # ============================================
    # NOUVEAU : SAUVEGARDE AUTOMATIQUE
    # ============================================
    console.print("\n[bold cyan]Saving results...[/bold cyan]")

    json_path, html_path = save_case_study_results(
        study_name="medical_entity_extraction",
        model_name=model_name,
        baseline_accuracy=baseline_acc,
        baseline_cost=baseline_cost,
        prompt_result=prompt_result,
        rag_result=rag_result,
        test_cases=test_cases,
        output_dir="results/case_studies",
    )

    console.print(f"[green]✓ JSON saved: {json_path}[/green]")
    console.print(f"[green]✓ HTML report: {html_path}[/green]")

    console.print("\n[bold green]✓ Case study complete![/bold green]")
    console.print("[cyan]Results demonstrate systematic approach to LLM improvement[/cyan]")
    console.print("\n[yellow]📊 Open the HTML report to view full results:[/yellow]")
    console.print(f"[yellow]   file://{html_path.absolute()}[/yellow]")


if __name__ == "__main__":
    main()
