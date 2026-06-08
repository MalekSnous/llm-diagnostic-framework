#!/usr/bin/env python3
"""
Generate HTML report from diagnostic and improvement results.
Usage: python scripts/generate_report.py --input results/ --output report.html
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Simple console replacement for rich
class SimpleConsole:
    def print(self, text):
        # Remove rich formatting tags
        import re

        clean_text = re.sub(r"\[.*?\]", "", text)
        print(clean_text)


console = SimpleConsole()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Diagnostic Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
        }}
        .success {{
            color: #27ae60;
        }}
        .warning {{
            color: #f39c12;
        }}
        .error {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <h1>🔍 LLM Diagnostic Report</h1>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <div class="metric-label">Report Generated</div>
            <div class="metric-value">{timestamp}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Model Tested</div>
            <div class="metric-value">{model}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Tests</div>
            <div class="metric-value">{total_tests}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Overall Success Rate</div>
            <div class="metric-value {success_class}">{overall_success_rate:.1%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value">${total_cost:.4f}</div>
        </div>
    </div>
    
    {diagnostic_sections}
    
    {improvement_sections}
    
    <div class="section">
        <h2>📊 Visualizations</h2>
        <div id="success-rate-chart"></div>
        <div id="cost-comparison-chart"></div>
    </div>
    
    <div class="section">
        <h2>💡 Key Recommendations</h2>
        {recommendations_html}
    </div>
    
    <script>
        {plotly_scripts}
    </script>
</body>
</html>
"""


def load_results_from_dir(results_dir: Path):
    """Load all JSON result files from directory."""
    diagnostic_results = []
    improvement_results = []

    for json_file in results_dir.glob("**/*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Determine if diagnostic or improvement
            if "test_name" in data:
                diagnostic_results.append(data)
            elif "strategy_name" in data:
                improvement_results.append(data)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {json_file}: {e}[/yellow]")

    return diagnostic_results, improvement_results


def generate_diagnostic_sections(diagnostic_results):
    """Generate HTML sections for diagnostic results."""
    sections = []

    for result in diagnostic_results:
        test_name = result.get("test_name", "Unknown Test")
        agg = result.get("aggregated_results", {})
        insights = result.get("diagnostic_insights", {})

        # Build recommendations HTML
        recs = insights.get("recommendations", [])
        recs_html = ""
        if recs:
            recs_html = "<ul>"
            for rec in recs[:3]:  # Top 3
                recs_html += f"<li>{rec}</li>"
            recs_html += "</ul>"

        # Success rate styling
        success_rate = agg.get("success_rate", 0)
        success_class = (
            "success" if success_rate > 0.8 else ("warning" if success_rate > 0.5 else "error")
        )

        section = f"""
        <div class="section">
            <h2>🧪 {test_name}</h2>
            <p><em>{result.get('description', '')}</em></p>
            
            <div class="metric">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value {success_class}">{success_rate:.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Test Cases</div>
                <div class="metric-value">{agg.get('num_test_cases', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Cost</div>
                <div class="metric-value">${agg.get('total_cost_usd', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Latency</div>
                <div class="metric-value">{agg.get('avg_latency_ms', 0):.0f}ms</div>
            </div>
            
            {f'<div class="recommendation"><strong>⚠️ Issues Detected:</strong>{recs_html}</div>' if recs else ''}
            
            {f'<p><strong>Explanation:</strong> {insights.get("theoretical_explanation", "")}</p>' if insights.get("theoretical_explanation") else ''}
        </div>
        """
        sections.append(section)

    return "\n".join(sections)


def generate_improvement_sections(improvement_results):
    """Generate HTML sections for improvement results."""
    if not improvement_results:
        return ""

    sections = ["<div class='section'><h2>🚀 Improvement Strategies</h2>"]

    for result in improvement_results:
        strategy_name = result.get("strategy_name", "Unknown Strategy")

        # Format 1: Avec "results" array (format wrapper)
        results_list = result.get("results", [])
        if results_list:
            first_result = results_list[0]
        # Format 2: Données directement dans result (format direct)
        elif "baseline_metrics" in result or "config" in result:
            first_result = result
        else:
            # Skip si pas de données
            continue

        # Handle the various metrics structures produced by older runs
        baseline_metrics = first_result.get("baseline_metrics", {})
        improved_metrics = first_result.get("improved_metrics", {})

        # Les metrics peuvent être directement dans baseline_metrics ou dans .metrics
        if isinstance(baseline_metrics, dict) and "metrics" in baseline_metrics:
            baseline = baseline_metrics.get("metrics", {})
        else:
            baseline = baseline_metrics

        if isinstance(improved_metrics, dict) and "metrics" in improved_metrics:
            improved = improved_metrics.get("metrics", {})
        else:
            improved = improved_metrics

        delta = first_result.get("improvement_delta", {})

        # Si pas de métriques, skip
        if not baseline and not improved:
            continue

        # Build comparison table
        table_html = (
            "<table><tr><th>Metric</th><th>Baseline</th><th>Improved</th><th>Delta</th></tr>"
        )

        # Combiner les clés des deux dicts
        all_metrics = set(baseline.keys()) | set(improved.keys())

        for metric_name in sorted(all_metrics):
            b_val = baseline.get(metric_name, 0)
            i_val = improved.get(metric_name, 0)
            d_val = delta.get(metric_name, i_val - b_val)

            delta_class = "success" if d_val > 0 else "error" if d_val < 0 else ""
            delta_str = f"+{d_val:.3f}" if d_val >= 0 else f"{d_val:.3f}"

            table_html += f"<tr><td>{metric_name.capitalize()}</td><td>{b_val:.3f}</td><td>{i_val:.3f}</td><td class='{delta_class}'>{delta_str}</td></tr>"

        table_html += "</table>"

        # Calcul de l'amélioration d'accuracy
        accuracy_improvement = (
            delta.get("accuracy", improved.get("accuracy", 0) - baseline.get("accuracy", 0)) * 100
        )

        section = f"""
        <h3>📈 {strategy_name}</h3>
        <p><em>{result.get('description', '')}</em></p>
        
        <div class="metric">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value">${first_result.get('total_cost', 0):.4f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Time</div>
            <div class="metric-value">{first_result.get('total_time_seconds', 0):.1f}s</div>
        </div>
        <div class="metric">
            <div class="metric-label">Accuracy Improvement</div>
            <div class="metric-value {'success' if accuracy_improvement >= 0 else 'error'}">{"+" if accuracy_improvement >= 0 else ""}{accuracy_improvement:.1f}%</div>
        </div>
        
        {table_html}
        """
        sections.append(section)

    sections.append("</div>")
    return "\n".join(sections)
    return "\n".join(sections)


def generate_plotly_scripts(diagnostic_results, improvement_results):
    """Generate Plotly visualization scripts."""
    # Extract data for charts
    test_names = [r.get("test_name", "Unknown") for r in diagnostic_results]
    success_rates = [
        r.get("aggregated_results", {}).get("success_rate", 0) * 100 for r in diagnostic_results
    ]
    costs = [r.get("aggregated_results", {}).get("total_cost_usd", 0) for r in diagnostic_results]

    scripts = f"""
    // Success Rate Chart
    var successData = [{{
        x: {json.dumps(test_names)},
        y: {json.dumps(success_rates)},
        type: 'bar',
        marker: {{
            color: {json.dumps(['#27ae60' if sr > 80 else '#f39c12' if sr > 50 else '#e74c3c' for sr in success_rates])}
        }}
    }}];
    
    var successLayout = {{
        title: 'Success Rate by Test',
        yaxis: {{title: 'Success Rate (%)'}}
    }};
    
    Plotly.newPlot('success-rate-chart', successData, successLayout);
    
    // Cost Comparison Chart
    var costData = [{{
        x: {json.dumps(test_names)},
        y: {json.dumps(costs)},
        type: 'bar',
        marker: {{color: '#3498db'}}
    }}];
    
    var costLayout = {{
        title: 'Cost by Test',
        yaxis: {{title: 'Cost (USD)'}}
    }};
    
    Plotly.newPlot('cost-comparison-chart', costData, costLayout);
    """

    return scripts


def generate_recommendations_html(diagnostic_results):
    """Generate key recommendations HTML."""
    all_recs = []

    for result in diagnostic_results:
        insights = result.get("diagnostic_insights", {})
        recs = insights.get("recommendations", [])
        all_recs.extend(recs)

    if not all_recs:
        return "<p>No recommendations at this time. Model performs well!</p>"

    html = ""
    for i, rec in enumerate(all_recs[:5], 1):  # Top 5
        html += f'<div class="recommendation"><strong>{i}.</strong> {rec}</div>'

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from results")

    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing JSON result files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./report.html",
        help="Output HTML file path (default: ./report.html)",
    )

    parser.add_argument(
        "--improvements",
        type=str,
        default=None,
        help="Directory containing improvement results (optional)",
    )

    parser.add_argument("--model", type=str, default="Unknown Model", help="Model name for report")

    args = parser.parse_args()

    console.print("[bold cyan]Generating LLM Diagnostic Report...[/bold cyan]")

    # Load results
    results_dir = Path(args.input)
    if not results_dir.exists():
        console.print(f"[bold red]Error: {results_dir} does not exist[/bold red]")
        sys.exit(1)

    # Load diagnostics from input directory
    diagnostic_results, _ = load_results_from_dir(results_dir)

    # Load improvements from separate directory if specified
    improvement_results = []
    if args.improvements:
        improvements_dir = Path(args.improvements)
        if improvements_dir.exists():
            _, improvement_results = load_results_from_dir(improvements_dir)
            console.print(
                f"[green]Found {len(improvement_results)} improvement results from {improvements_dir}[/green]"
            )
        else:
            console.print(
                f"[yellow]Warning: Improvements directory {improvements_dir} does not exist[/yellow]"
            )
    else:
        # Try to find improvements in subdirectory of input
        improvements_dir = results_dir / "improvements"
        if improvements_dir.exists():
            _, improvement_results = load_results_from_dir(improvements_dir)
            console.print(
                f"[green]Found {len(improvement_results)} improvement results from {improvements_dir}[/green]"
            )

    console.print(f"[green]Found {len(diagnostic_results)} diagnostic results[/green]")
    console.print(f"[green]Total improvements loaded: {len(improvement_results)}[/green]")

    # DEBUG: Afficher la structure du premier improvement
    if improvement_results:
        console.print("\n[yellow]DEBUG: First improvement structure:[/yellow]")
        first = improvement_results[0]
        console.print(f"  strategy_name: {first.get('strategy_name', 'N/A')}")
        console.print(f"  Has 'results' key: {'results' in first}")
        console.print(f"  Has 'baseline_metrics' key: {'baseline_metrics' in first}")
        console.print(f"  Has 'config' key: {'config' in first}")
        console.print(f"  Top-level keys: {list(first.keys())[:10]}")
        console.print("")

    if not diagnostic_results:
        console.print("[yellow]Warning: No diagnostic results found[/yellow]")

    # Calculate summary metrics
    total_tests = len(diagnostic_results)
    overall_success = (
        sum(r.get("aggregated_results", {}).get("success_rate", 0) for r in diagnostic_results)
        / total_tests
        if total_tests > 0
        else 0
    )

    total_cost = sum(
        r.get("aggregated_results", {}).get("total_cost_usd", 0) for r in diagnostic_results
    )

    success_class = (
        "success" if overall_success > 0.8 else ("warning" if overall_success > 0.5 else "error")
    )

    # Generate HTML sections
    diagnostic_sections = generate_diagnostic_sections(diagnostic_results)
    improvement_sections = generate_improvement_sections(improvement_results)
    plotly_scripts = generate_plotly_scripts(diagnostic_results, improvement_results)
    recommendations_html = generate_recommendations_html(diagnostic_results)

    # Fill template
    html_content = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model=args.model,
        total_tests=total_tests,
        overall_success_rate=overall_success,
        success_class=success_class,
        total_cost=total_cost,
        diagnostic_sections=diagnostic_sections,
        improvement_sections=improvement_sections,
        plotly_scripts=plotly_scripts,
        recommendations_html=recommendations_html,
    )

    # Write to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write(html_content)

    console.print(f"[bold green]✓ Report generated: {output_path}[/bold green]")
    console.print(f"[cyan]Open {output_path} in your browser to view[/cyan]")


if __name__ == "__main__":
    main()
