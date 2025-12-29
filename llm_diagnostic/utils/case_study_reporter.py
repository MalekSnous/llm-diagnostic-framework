"""
Module pour sauvegarder et générer des rapports pour les case studies.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class CaseStudyReporter:
    """Gère la sauvegarde et les rapports des case studies."""
    
    def __init__(self, study_name: str, output_dir: str = "results/case_studies"):
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_results(
        self,
        model_name: str,
        baseline_results: Dict[str, Any],
        improvement_results: Dict[str, Any]
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
            "summary": self._generate_summary(baseline_results, improvement_results)
        }
        
        # Sauvegarder en JSON
        filename = f"{self.study_name}_{model_name.replace('/', '_')}_{self.timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def _generate_summary(
        self,
        baseline: Dict[str, Any],
        improvements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère un résumé des résultats."""
        baseline_acc = baseline.get("accuracy", 0)
        
        summary = {
            "baseline_accuracy": baseline_acc,
            "best_strategy": None,
            "best_improvement": 0,
            "strategies": {}
        }
        
        best_acc = baseline_acc
        best_strategy = "baseline"
        
        for strategy_name, result in improvements.items():
            acc = result.get("accuracy", 0)
            improvement = acc - baseline_acc
            
            summary["strategies"][strategy_name] = {
                "accuracy": acc,
                "improvement": improvement,
                "improvement_pct": improvement * 100
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
        test_cases_details: List[Dict[str, Any]] = None
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
            model_name,
            baseline_results,
            improvement_results,
            test_cases_details
        )
        
        filename = f"{self.study_name}_{model_name.replace('/', '_')}_{self.timestamp}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return filepath
    
    def _build_html(
        self,
        model_name: str,
        baseline: Dict[str, Any],
        improvements: Dict[str, Any],
        test_cases: List[Dict[str, Any]] = None
    ) -> str:
        """Construit le HTML du rapport."""
        
        baseline_acc = baseline.get("accuracy", 0)
        baseline_cost = baseline.get("cost", 0)
        
        # Tableau des résultats
        results_rows = f"""
            <tr>
                <td><strong>Baseline (Zero-shot)</strong></td>
                <td>{baseline_acc:.1%}</td>
                <td>-</td>
                <td>${baseline_cost:.4f}</td>
                <td>-</td>
            </tr>
        """
        
        for strategy_name, result in improvements.items():
            acc = result.get("accuracy", 0)
            cost = result.get("cost", 0)
            improvement = (acc - baseline_acc) * 100
            cost_per_point = cost / improvement if improvement > 0 else 0
            
            results_rows += f"""
            <tr>
                <td><strong>{strategy_name}</strong></td>
                <td>{acc:.1%}</td>
                <td class="{"positive" if improvement > 0 else "negative"}">{improvement:+.1f}%</td>
                <td>${cost:.4f}</td>
                <td>${cost_per_point:.2f}</td>
            </tr>
            """
        
        # Section des test cases (si fournis)
        test_cases_section = ""
        if test_cases:
            test_cases_rows = ""
            for i, tc in enumerate(test_cases):
                test_cases_rows += f"""
                <tr>
                    <td>{i+1}</td>
                    <td class="test-input">{tc.get('input', 'N/A')[:100]}...</td>
                    <td class="test-output">{tc.get('expected', 'N/A')}</td>
                    <td class="{"success" if tc.get('success', False) else "failure"}">
                        {tc.get('baseline_prediction', 'N/A')[:50]}...
                    </td>
                </tr>
                """
            
            test_cases_section = f"""
            <div class="section">
                <h2>📋 Test Cases Details</h2>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Input</th>
                            <th>Expected</th>
                            <th>Baseline Prediction</th>
                        </tr>
                    </thead>
                    <tbody>
                        {test_cases_rows}
                    </tbody>
                </table>
            </div>
            """
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.study_name} - Case Study Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .metadata {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .metadata-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .metadata-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
        }}
        
        .metadata-value {{
            font-size: 1.1em;
            font-weight: 600;
            color: #495057;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        tbody tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .positive {{
            color: #28a745;
            font-weight: 600;
        }}
        
        .negative {{
            color: #dc3545;
            font-weight: 600;
        }}
        
        .success {{
            background-color: #d4edda;
        }}
        
        .failure {{
            background-color: #f8d7da;
        }}
        
        .test-input, .test-output {{
            font-size: 0.9em;
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 5px;
            border-radius: 3px;
        }}
        
        .recommendation {{
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-top: 30px;
            border-radius: 5px;
        }}
        
        .recommendation h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            border-top: 2px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {self.study_name}</h1>
            <p class="subtitle">Case Study Report</p>
        </div>
        
        <div class="metadata">
            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="metadata-label">Model</span>
                    <span class="metadata-value">{model_name}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Date</span>
                    <span class="metadata-value">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Baseline Accuracy</span>
                    <span class="metadata-value">{baseline_acc:.1%}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Best Improvement</span>
                    <span class="metadata-value positive">
                        {max([r.get('accuracy', 0) - baseline_acc for r in improvements.values()]) * 100:+.1f}%
                    </span>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📈 Results Summary</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Accuracy</th>
                            <th>Improvement</th>
                            <th>Cost</th>
                            <th>Cost per Point</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results_rows}
                    </tbody>
                </table>
            </div>
            
            {test_cases_section}
            
            <div class="recommendation">
                <h3>✅ Recommendation</h3>
                <p>
                    Based on the results, <strong>{max(improvements.items(), key=lambda x: x[1].get('accuracy', 0))[0]}</strong> 
                    achieves the best performance with 
                    {max([r.get('accuracy', 0) for r in improvements.values()]):.1%} accuracy 
                    ({max([r.get('accuracy', 0) - baseline_acc for r in improvements.values()]) * 100:+.1f}% improvement).
                </p>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by LLM Diagnostic Framework | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
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
    output_dir: str = "results/case_studies"
) -> tuple:
    """
    Fonction helper pour sauvegarder facilement les résultats d'un case study.
    
    Returns:
        tuple: (json_path, html_path)
    """
    reporter = CaseStudyReporter(study_name, output_dir)
    
    # Préparer les données baseline
    baseline_results = {
        "accuracy": baseline_accuracy,
        "cost": baseline_cost
    }
    
    # Préparer les résultats d'amélioration
    improvement_results = {
        "Prompt Engineering": {
            "accuracy": prompt_result.improved_metrics.metrics.get("accuracy", 0),
            "cost": prompt_result.total_cost
        },
        "RAG System": {
            "accuracy": rag_result.improved_metrics.metrics.get("accuracy", 0),
            "cost": rag_result.total_cost
        }
    }
    
    # Préparer les détails des test cases (optionnel)
    test_cases_details = None
    if test_cases:
        test_cases_details = [
            {
                "input": tc.input,
                "expected": tc.expected_output,
                "success": True  # À ajuster selon vos besoins
            }
            for tc in test_cases
        ]
    
    # Sauvegarder JSON
    json_path = reporter.save_results(model_name, baseline_results, improvement_results)
    
    # Générer HTML
    html_path = reporter.generate_html_report(
        model_name,
        baseline_results,
        improvement_results,
        test_cases_details
    )
    
    return json_path, html_path