#!/usr/bin/env python3
"""
Build a cross-model comparison from case-study result JSONs.

Reads the latest ``results/case_studies/<study>_<model>_<ts>.json`` per model
(keyed on the model name stored inside the JSON), prints a Rich table, and
writes a self-contained HTML + Markdown report with:
  - a global table (baseline F1, +prompt-engineering F1, delta, cost);
  - a per-difficulty F1 breakdown (when the runs captured it);
  - auto-generated findings (best model, biggest gain, best value).

Usage:
    python scripts/compare_models.py --study medical_entity_extraction

Typically run via ``make compare-medical`` (which runs the studies first).
"""

from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "case_studies"
DOCS = ROOT / "docs"
console = Console()

TIERS = ["easy", "medium", "hard", "expert"]
PE = "Prompt Engineering"


def latest_per_model(study: str) -> dict[str, dict]:
    """Return {model_name: parsed_json}, keeping the most recent run per model."""
    latest: dict[str, tuple[str, dict]] = {}
    for path in glob.glob(str(RESULTS / f"{study}_*.json")):
        if "comparison" in Path(path).name:
            continue
        data = json.loads(Path(path).read_text())
        model, ts = data.get("model"), data.get("timestamp", "")
        if not model:
            continue
        if model not in latest or ts > latest[model][0]:
            latest[model] = (ts, data)
    return {m: d for m, (_, d) in latest.items()}


def _best_improvement(data: dict) -> tuple[str | None, dict]:
    """Pick the highest-F1 improvement strategy for a model."""
    imps = data.get("improvements", {})
    if not imps:
        return None, {}
    name = max(imps, key=lambda k: imps[k].get("accuracy", 0))
    return name, imps[name]


def render_console(data_by_model: dict[str, dict]) -> None:
    table = Table(title="Model comparison — medical entity extraction (F1)", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Baseline F1", justify="right", style="green")
    table.add_column("+Prompt Eng", justify="right", style="green")
    table.add_column("Δ", justify="right", style="magenta")
    table.add_column("Cost (USD)", justify="right", style="yellow")
    for model in sorted(data_by_model):
        d = data_by_model[model]
        base = d["baseline"]["accuracy"]
        name, imp = _best_improvement(d)
        if name:
            pe, cost = imp.get("accuracy", 0), imp.get("cost", 0)
            table.add_row(
                model, f"{base:.1%}", f"{pe:.1%}", f"{(pe - base) * 100:+.1f}", f"${cost:.4f}"
            )
        else:
            table.add_row(model, f"{base:.1%}", "—", "—", f"${d['baseline'].get('cost', 0):.4f}")
    console.print(table)


def _findings(data_by_model: dict[str, dict]) -> list[str]:
    """Auto-generate interpretation bullets from the numbers."""
    out: list[str] = []
    rows = []
    for model, d in data_by_model.items():
        base = d["baseline"]["accuracy"]
        _, imp = _best_improvement(d)
        best = max(base, imp.get("accuracy", 0)) if imp else base
        cost = (
            imp.get("cost", d["baseline"].get("cost", 0)) if imp else d["baseline"].get("cost", 0)
        )
        rows.append((model, base, imp.get("accuracy", base), best, cost))

    if not rows:
        return out

    top = max(rows, key=lambda r: r[3])
    out.append(f"**Best quality:** {top[0]} reaches F1 {top[3]:.1%}.")

    gains = [(m, pe - base) for m, base, pe, _, _ in rows]
    mg = max(gains, key=lambda x: x[1])
    if mg[1] > 0.01:
        out.append(f"**Prompt engineering helps most** on {mg[0]} ({mg[1] * 100:+.1f} F1 points).")

    # A very low baseline that jumps with few-shot is usually a formatting
    # artifact (model answered in prose), not weak extraction.
    fmt = [m for m, base, pe, _, _ in rows if base < 0.30 and (pe - base) > 0.30]
    if fmt:
        out.append(
            f"**Low baseline ≠ weak model:** {', '.join(fmt)} scored low at baseline mainly by "
            f"answering in prose; few-shot standardised the output format (hence the large jump). "
            f"Read baseline-vs-prompt as much about output format as raw capability."
        )
    hurts = [m for m, delta in gains if delta < -0.01]
    if hurts:
        out.append(
            f"**Prompt engineering *hurt*** {', '.join(hurts)} — generic few-shot added noise."
        )

    paid = [r for r in rows if r[4] > 0]
    if paid:
        cheapest = min(paid, key=lambda r: r[4])
        priciest = max(paid, key=lambda r: r[4])
        if priciest[4] > 0 and cheapest[4] > 0:
            ratio = priciest[4] / cheapest[4]
            out.append(
                f"**Cost spread is huge:** {priciest[0]} costs ~{ratio:.0f}× more than "
                f"{cheapest[0]} for this run."
            )
        # value = F1 per dollar, paid models only
        value = max(paid, key=lambda r: r[3] / r[4] if r[4] else 0)
        out.append(f"**Best value (F1 per $):** {value[0]}.")

    # bigger-isn't-better: a cheaper model matching/beating a pricier one
    if len(paid) >= 2:
        s = sorted(paid, key=lambda r: r[4])
        if s[0][3] >= s[-1][3] - 0.02:
            out.append(
                f"**Bigger ≠ better:** {s[0][0]} matches the most expensive model "
                f"within 2 F1 points at a fraction of the cost."
            )
    return out


def _difficulty_rows(data_by_model: dict[str, dict]):
    """(model, {tier: baseline_f1}) for models that captured per-difficulty data."""
    rows = []
    for model in sorted(data_by_model):
        by = data_by_model[model].get("baseline", {}).get("by_difficulty") or {}
        if by:
            rows.append((model, by))
    return rows


def render_markdown(data_by_model: dict[str, dict]) -> str:
    lines = [
        "# Model comparison — medical entity extraction (F1)\n",
        "| Model | Baseline F1 | +Prompt Eng | Δ | Cost (USD) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in sorted(data_by_model):
        d = data_by_model[model]
        base = d["baseline"]["accuracy"]
        name, imp = _best_improvement(d)
        pe = imp.get("accuracy", base)
        cost = imp.get("cost", d["baseline"].get("cost", 0))
        lines.append(
            f"| {model} | {base:.1%} | {pe:.1%} | {(pe - base) * 100:+.1f} | ${cost:.4f} |"
        )

    diff = _difficulty_rows(data_by_model)
    if diff:
        lines += ["\n## Baseline F1 by difficulty\n", "| Model | " + " | ".join(TIERS) + " |"]
        lines.append("|---" * (len(TIERS) + 1) + "|")
        for model, by in diff:
            cells = " | ".join(f"{by[t]:.0%}" if t in by else "—" for t in TIERS)
            lines.append(f"| {model} | {cells} |")

    findings = _findings(data_by_model)
    if findings:
        lines += ["\n## Findings\n"] + [f"- {f}" for f in findings]
    return "\n".join(lines) + "\n"


def render_html(data_by_model: dict[str, dict]) -> str:
    grows = ""
    for model in sorted(data_by_model):
        d = data_by_model[model]
        base = d["baseline"]["accuracy"]
        name, imp = _best_improvement(d)
        pe = imp.get("accuracy", base)
        cost = imp.get("cost", d["baseline"].get("cost", 0))
        delta = (pe - base) * 100
        cls = "pos" if delta >= 0 else "neg"
        grows += (
            f"<tr><td>{html.escape(model)}</td><td>{base:.1%}</td>"
            f"<td>{pe:.1%}</td><td class='{cls}'>{delta:+.1f}</td><td>${cost:.4f}</td></tr>\n"
        )

    diff = _difficulty_rows(data_by_model)
    diff_html = ""
    if diff:
        head = "".join(f"<th>{t}</th>" for t in TIERS)
        body = ""
        for model, by in diff:
            cells = "".join(f"<td>{by[t]:.0%}</td>" if t in by else "<td>—</td>" for t in TIERS)
            body += f"<tr><td>{html.escape(model)}</td>{cells}</tr>\n"
        diff_html = f"""
<h2>Baseline F1 by difficulty</h2>
<p>Where each model breaks down as notes get denser (explicit → clinical shorthand).</p>
<table><thead><tr><th>Model</th>{head}</tr></thead><tbody>
{body}</tbody></table>"""
    else:
        diff_html = (
            "<h2>Baseline F1 by difficulty</h2><p><em>No per-difficulty data in these runs — "
            "re-run the studies (the saver now captures it) to populate this table.</em></p>"
        )

    findings = _findings(data_by_model)
    findings_html = ""
    if findings:
        items = "".join(f"<li>{html.escape(f).replace('**','')}</li>" for f in findings)
        findings_html = f"<h2>Findings</h2><ul>{items}</ul>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Model comparison — medical entity extraction</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0d1117;color:#e6edf3;max-width:860px;margin:40px auto;padding:0 20px;line-height:1.6}}
 h1{{font-size:1.7rem}} h2{{font-size:1.25rem;margin-top:34px}} a{{color:#6e8efb}}
 table{{width:100%;border-collapse:collapse;background:#161b22;border-radius:10px;overflow:hidden;margin:12px 0}}
 th,td{{padding:10px 14px;text-align:right;border-bottom:1px solid #30363d}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#1f2630}} .pos{{color:#3fb950}} .neg{{color:#f85149}}
 .lead{{color:#9da7b3}}
</style></head><body>
<h1>📊 Model comparison — medical entity extraction</h1>
<p class="lead">Same 97-case clinical dataset (easy → expert), one harness. Quality is
entity-extraction <strong>F1</strong> (precision/recall, verbosity-robust); cost is real
token cost. Baseline = zero-shot; +Prompt Eng = few-shot prompting.</p>
<table><thead><tr><th>Model</th><th>Baseline F1</th><th>+Prompt Eng</th><th>Δ (pts)</th><th>Cost (USD)</th></tr></thead>
<tbody>
{grows}</tbody></table>
{diff_html}
{findings_html}
<p style="margin-top:28px"><a href="index.html">&larr; Back to overview</a></p>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model case-study comparison")
    parser.add_argument("--study", default="medical_entity_extraction")
    args = parser.parse_args()

    data = latest_per_model(args.study)
    if not data:
        console.print(
            f"[yellow]No results for '{args.study}' in {RESULTS}. Run the study first.[/yellow]"
        )
        return

    render_console(data)
    for f in _findings(data):
        console.print(f"  • {f.replace('**', '')}")

    DOCS.mkdir(exist_ok=True)
    html_path = DOCS / f"comparison_{args.study}.html"
    md_path = RESULTS / f"comparison_{args.study}.md"
    html_path.write_text(render_html(data), encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    console.print(f"\n[green]HTML:[/green] {html_path}\n[green]Markdown:[/green] {md_path}")
    console.print(f"[cyan]Compared {len(data)} model(s): {', '.join(sorted(data))}[/cyan]")


if __name__ == "__main__":
    main()
