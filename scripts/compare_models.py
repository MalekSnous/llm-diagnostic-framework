#!/usr/bin/env python3
"""
Build a cross-model comparison from case-study result JSONs.

Reads the latest ``results/case_studies/<study>_<model>_<ts>.json`` per model,
prints a Rich comparison table, and writes a self-contained HTML + Markdown
summary so several models can be compared side by side (e.g. gpt-4o vs
gpt-4o-mini vs microsoft/phi-2 vs claude-sonnet-4-6 on medical entity
extraction).

Usage:
    python scripts/compare_models.py --study medical_entity_extraction

Typically run via ``make compare-medical`` (which runs the studies first).
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "case_studies"
DOCS = ROOT / "docs"
console = Console()


def latest_per_model(study: str) -> dict[str, dict]:
    """Return {model: parsed_json} keeping the most recent run per model."""
    # Anchor on the known study name so a model like "microsoft_phi-2" (slash
    # replaced) isn't mis-split by a greedy study group.
    pattern = re.compile(rf"^{re.escape(study)}_(?P<model>.+)_(?P<stamp>\d{{8}}_\d{{6}})$")
    latest: dict[str, tuple[str, Path]] = {}
    for path in RESULTS.glob(f"{study}_*.json"):
        m = pattern.match(path.stem)
        if not m:
            continue
        model, stamp = m.group("model"), m.group("stamp")
        if model not in latest or stamp > latest[model][0]:
            latest[model] = (stamp, path)
    return {model: json.loads(p.read_text()) for model, (_, p) in latest.items()}


def _rows(data_by_model: dict[str, dict]):
    """Flatten into rows: (model, strategy, score, cost)."""
    for model, data in sorted(data_by_model.items()):
        base = data.get("baseline", {})
        yield model, "Baseline", base.get("accuracy", 0.0), base.get("cost", 0.0)
        for strat, vals in data.get("improvements", {}).items():
            yield model, strat, vals.get("accuracy", 0.0), vals.get("cost", 0.0)


def render_console(study: str, data_by_model: dict[str, dict]) -> None:
    table = Table(title=f"Model comparison — {study} (F1)", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Strategy", style="white")
    table.add_column("F1", style="green", justify="right")
    table.add_column("Cost (USD)", style="yellow", justify="right")
    for model, strat, score, cost in _rows(data_by_model):
        table.add_row(model, strat, f"{score:.1%}", f"${cost:.4f}")
    console.print(table)


def render_markdown(study: str, data_by_model: dict[str, dict]) -> str:
    lines = [
        f"# Model comparison — {study} (F1)\n",
        "| Model | Strategy | F1 | Cost (USD) |",
        "|---|---|---:|---:|",
    ]
    for model, strat, score, cost in _rows(data_by_model):
        lines.append(f"| {model} | {strat} | {score:.1%} | ${cost:.4f} |")
    return "\n".join(lines) + "\n"


def render_html(study: str, data_by_model: dict[str, dict]) -> str:
    body = "\n".join(
        f"<tr><td>{html.escape(model)}</td><td>{html.escape(strat)}</td>"
        f"<td>{score:.1%}</td><td>${cost:.4f}</td></tr>"
        for model, strat, score, cost in _rows(data_by_model)
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Model comparison — {html.escape(study)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0d1117;color:#e6edf3;max-width:820px;margin:40px auto;padding:0 20px}}
 h1{{font-size:1.5rem}} a{{color:#6e8efb}}
 table{{width:100%;border-collapse:collapse;background:#161b22;border-radius:10px;overflow:hidden}}
 th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid #30363d}}
 td:nth-child(3),td:nth-child(4),th:nth-child(3),th:nth-child(4){{text-align:right}}
 th{{background:#1f2630}}
</style></head><body>
<h1>Model comparison — {html.escape(study)} (F1)</h1>
<p>Headline metric is F1 (precision/recall, verbosity-robust). Costs are real token costs.</p>
<table><thead><tr><th>Model</th><th>Strategy</th><th>F1</th><th>Cost (USD)</th></tr></thead>
<tbody>
{body}
</tbody></table>
<p><a href="index.html">&larr; Back</a></p>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model case-study comparison")
    parser.add_argument("--study", default="medical_entity_extraction")
    args = parser.parse_args()

    data = latest_per_model(args.study)
    if not data:
        console.print(
            f"[yellow]No results found for '{args.study}' in {RESULTS}. "
            f"Run the case study for some models first.[/yellow]"
        )
        return

    render_console(args.study, data)

    DOCS.mkdir(exist_ok=True)
    html_path = DOCS / f"comparison_{args.study}.html"
    md_path = RESULTS / f"comparison_{args.study}.md"
    html_path.write_text(render_html(args.study, data), encoding="utf-8")
    md_path.write_text(render_markdown(args.study, data), encoding="utf-8")
    console.print(f"\n[green]HTML:[/green] {html_path}\n[green]Markdown:[/green] {md_path}")
    console.print(f"[cyan]Compared {len(data)} model(s): {', '.join(sorted(data))}[/cyan]")


if __name__ == "__main__":
    main()
