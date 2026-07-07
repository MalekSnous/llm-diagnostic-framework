"""
Module pour sauvegarder et générer des rapports pour les case studies.
"""

import html as _html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from . import report_theme


class CaseStudyReporter:
    """Gère la sauvegarde et les rapports des case studies."""

    def __init__(self, study_name: str, output_dir: str = "results/case_studies"):
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_results(
        self, model_name: str, baseline_results: Dict[str, Any], improvement_results: Dict[str, Any]
    ) -> Path:
        """
        Sauvegarde les résultats en JSON.

        Args:
            model_name: Nom du modèle utilisé
            baseline_results: Résultats baseline
            improvement_results: Dict des résultats d'amélioration par stratégie

        Returns:
            Path du fichier JSON sauvegardé
        """
        data = {
            "study_name": self.study_name,
            "model": model_name,
            "timestamp": self.timestamp,
            "baseline": baseline_results,
            "improvements": improvement_results,
            "summary": self._generate_summary(baseline_results, improvement_results),
        }

        # Sauvegarder en JSON
        filename = f"{self.study_name}_{model_name.replace('/', '_')}_{self.timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def _generate_summary(
        self, baseline: Dict[str, Any], improvements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère un résumé des résultats."""
        baseline_acc = baseline.get("accuracy", 0)

        summary = {
            "baseline_accuracy": baseline_acc,
            "best_strategy": None,
            "best_improvement": 0,
            "strategies": {},
        }

        best_acc = baseline_acc
        best_strategy = "baseline"

        for strategy_name, result in improvements.items():
            acc = result.get("accuracy", 0)
            improvement = acc - baseline_acc

            summary["strategies"][strategy_name] = {
                "accuracy": acc,
                "improvement": improvement,
                "improvement_pct": improvement * 100,
            }

            if acc > best_acc:
                best_acc = acc
                best_strategy = strategy_name
                summary["best_improvement"] = improvement

        summary["best_strategy"] = best_strategy

        return summary

    def generate_html_report(
        self,
        model_name: str,
        baseline_results: Dict[str, Any],
        improvement_results: Dict[str, Any],
        test_cases_details: List[Dict[str, Any]] = None,
    ) -> Path:
        """
        Génère un rapport HTML complet.

        Args:
            model_name: Nom du modèle
            baseline_results: Résultats baseline
            improvement_results: Résultats des améliorations
            test_cases_details: Détails des test cases (optionnel)

        Returns:
            Path du fichier HTML
        """
        html_content = self._build_html(
            model_name, baseline_results, improvement_results, test_cases_details
        )

        filename = f"{self.study_name}_{model_name.replace('/', '_')}_{self.timestamp}.html"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(html_content)

        return filepath

    def _build_html(
        self,
        model_name: str,
        baseline: Dict[str, Any],
        improvements: Dict[str, Any],
        test_cases: List[Dict[str, Any]] = None,
    ) -> str:
        """Construit le HTML du rapport (design system partagé, clair/sombre)."""

        baseline_acc = baseline.get("accuracy", 0)
        baseline_cost = baseline.get("cost", 0)

        # Tableau des résultats
        results_rows = f"""
            <tr>
                <td><strong>Baseline (Zero-shot)</strong></td>
                <td class="num">{baseline_acc:.1%}</td>
                <td class="num">-</td>
                <td class="num">${baseline_cost:.4f}</td>
                <td class="num">-</td>
            </tr>
        """

        for strategy_name, result in improvements.items():
            acc = result.get("accuracy", 0)
            cost = result.get("cost", 0)
            improvement = (acc - baseline_acc) * 100
            cost_per_point = cost / improvement if improvement > 0 else 0

            results_rows += f"""
            <tr>
                <td><strong>{_html.escape(strategy_name)}</strong></td>
                <td class="num">{acc:.1%}</td>
                <td class="num {"pos" if improvement > 0 else "neg"}">{improvement:+.1f}%</td>
                <td class="num">${cost:.4f}</td>
                <td class="num">${cost_per_point:.2f}</td>
            </tr>
            """

        # Section des test cases (si fournis)
        test_cases_section = ""
        if test_cases:
            test_cases_rows = ""
            for i, tc in enumerate(test_cases):
                ok = tc.get("success", False)
                test_cases_rows += f"""
                <tr>
                    <td class="num">{i+1}</td>
                    <td><code>{_html.escape(str(tc.get('input', 'N/A'))[:100])}...</code></td>
                    <td><code>{_html.escape(str(tc.get('expected', 'N/A')))}</code></td>
                    <td class="{"pos" if ok else "neg"}">
                        {_html.escape(str(tc.get('baseline_prediction', 'N/A'))[:50])}...
                    </td>
                </tr>
                """

            test_cases_section = f"""
            <h2>📋 Test cases details</h2>
            <div class="tablewrap">
                <table>
                    <thead>
                        <tr>
                            <th class="num">ID</th>
                            <th>Input</th>
                            <th>Expected</th>
                            <th>Baseline prediction</th>
                        </tr>
                    </thead>
                    <tbody>
                        {test_cases_rows}
                    </tbody>
                </table>
            </div>
            """

        # Métadonnées, graphique et recommandation
        try:
            run_date = datetime.strptime(self.timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
        except ValueError:
            run_date = self.timestamp

        best_delta = (
            max(r.get("accuracy", 0) - baseline_acc for r in improvements.values())
            if improvements
            else 0.0
        )
        tiles = report_theme.stat_tile("Model", _html.escape(model_name), f"run {run_date}")
        tiles += report_theme.stat_tile("Baseline accuracy", f"{baseline_acc:.1%}", "zero-shot")
        tiles += report_theme.stat_tile(
            "Best improvement",
            f"{best_delta * 100:+.1f} pts",
            "vs baseline",
            delta_good=best_delta >= 0,
        )
        tiles += report_theme.stat_tile("Baseline cost", f"${baseline_cost:.4f}", "full run")

        bar_rows = [("Baseline (zero-shot)", baseline_acc)] + [
            (name, r.get("accuracy", 0)) for name, r in improvements.items()
        ]
        chart = report_theme.chart_figure(
            "Accuracy by strategy",
            "Same test set, same model — only the strategy changes.",
            [],
            report_theme.svg_strategy_bars(bar_rows),
        )

        recommendation = ""
        if improvements:
            best_name, best_res = max(improvements.items(), key=lambda x: x[1].get("accuracy", 0))
            recommendation = f"""
            <div class="callout">
                <h3>✅ Recommendation</h3>
                <p>
                    Based on the results, <strong>{_html.escape(best_name)}</strong>
                    achieves the best performance with
                    {best_res.get('accuracy', 0):.1%} accuracy
                    ({best_delta * 100:+.1f} pts vs baseline).
                </p>
            </div>
            """

        title = self.study_name.replace("_", " ").title()
        head = report_theme.page_head(
            f"{title} — {model_name} — Case Study Report",
            f"Baseline vs improvement strategies for {model_name}: accuracy and real token cost.",
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
{head}
</head>
<body>
    <div class="wrap">
        <a class="crumb" href="index.html">&larr; LLM Diagnostic Framework</a>
        <h1>📊 {_html.escape(title)}</h1>
        <p class="lead">Case study report — <strong>{_html.escape(model_name)}</strong>,
        baseline vs improvement strategies, scored on the same test set with real token cost.</p>

        <div class="tiles">{tiles}</div>

        {chart}

        <h2>📈 Results summary</h2>
        <div class="tablewrap">
            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th class="num">Accuracy</th>
                        <th class="num">Improvement</th>
                        <th class="num">Cost</th>
                        <th class="num">Cost per point</th>
                    </tr>
                </thead>
                <tbody>
                    {results_rows}
                </tbody>
            </table>
        </div>

        {test_cases_section}

        {recommendation}

        <footer>Generated by
            <a href="https://github.com/MalekSnous/llm-diagnostic-framework">LLM Diagnostic
            Framework</a> · run {run_date} · <a href="index.html">Back to overview</a>
        </footer>
    </div>
</body>
</html>
"""

        return html


def save_case_study_results(
    study_name: str,
    model_name: str,
    baseline_accuracy: float,
    baseline_cost: float,
    prompt_result: Any,
    rag_result: Any,
    test_cases: List[Any] = None,
    output_dir: str = "results/case_studies",
) -> tuple:
    """
    Fonction helper pour sauvegarder facilement les résultats d'un case study.

    Returns:
        tuple: (json_path, html_path)
    """
    reporter = CaseStudyReporter(study_name, output_dir)

    # Préparer les données baseline
    baseline_results = {"accuracy": baseline_accuracy, "cost": baseline_cost}

    # Préparer les résultats d'amélioration
    improvement_results = {
        "Prompt Engineering": {
            "accuracy": prompt_result.improved_metrics.metrics.get("accuracy", 0),
            "cost": prompt_result.total_cost,
        },
        "RAG System": {
            "accuracy": rag_result.improved_metrics.metrics.get("accuracy", 0),
            "cost": rag_result.total_cost,
        },
    }

    # Préparer les détails des test cases (optionnel)
    test_cases_details = None
    if test_cases:
        test_cases_details = [
            {
                "input": tc.input,
                "expected": tc.expected_output,
                "success": True,  # À ajuster selon vos besoins
            }
            for tc in test_cases
        ]

    # Sauvegarder JSON
    json_path = reporter.save_results(model_name, baseline_results, improvement_results)

    # Générer HTML
    html_path = reporter.generate_html_report(
        model_name, baseline_results, improvement_results, test_cases_details
    )

    return json_path, html_path
