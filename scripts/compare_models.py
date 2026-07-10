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
    python scripts/compare_models.py --study text_to_sql

Typically run via ``make benchmark`` (which runs the studies first).
"""

from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_diagnostic.utils import report_theme

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "case_studies"
DOCS = ROOT / "docs"
console = Console()

TIERS = ["easy", "medium", "hard", "expert"]
PE = "Prompt Engineering"

# Per-study display metadata; unknown studies fall back to a generic entry.
# imp_label = column header for the best improvement arm; imp_name = how the
# improvement is called in prose findings.
STUDY_META = {
    "medical_entity_extraction": {
        "title": "medical entity extraction",
        "metric": "F1",
        "imp_label": "+Prompt Eng",
        "imp_name": "Prompt engineering",
        "lead": (
            "Same clinical dataset (easy → expert), one harness. Quality is "
            "entity-extraction <strong>F1</strong> (precision/recall, verbosity-robust); "
            "cost is real token cost. Baseline = zero-shot; +Prompt Eng = few-shot prompting."
        ),
        "diff_lead": "Where each model breaks down as notes get denser "
        "(explicit → clinical shorthand).",
    },
    "text_to_sql": {
        "title": "text-to-SQL",
        "metric": "execution accuracy",
        "imp_label": "+Prompt Eng",
        "imp_name": "Prompt engineering",
        "lead": (
            "Same question set over one SQLite schema, one harness. Quality is "
            "<strong>execution accuracy</strong> — the generated SQL is executed and its "
            "result set compared to the gold query's, so verbosity and formatting tricks "
            "can't inflate the score. Cost is real token cost. Baseline = zero-shot with "
            "the schema in the prompt; +Prompt Eng = few-shot SQL examples."
        ),
        "diff_lead": "Where each model breaks down as queries get harder "
        "(single-table filters → multi-join analytics).",
    },
    "rag_document_qa": {
        "title": "RAG document QA",
        "metric": "accuracy",
        "imp_label": "+RAG",
        "imp_name": "RAG",
        "lead": (
            "100 questions about a fictional internal platform whose facts exist only in "
            "a private knowledge base — no model can know them from training. Baseline = "
            "closed-book; +RAG = the same model reading chunks retrieved (and reranked) "
            "from the indexed docs. Accuracy uses digit-boundary matching against accepted "
            "answers, and the expert tier includes unanswerable questions where the right "
            "behaviour is to abstain. Cost is real token cost."
        ),
        "diff_lead": "Closed-book accuracy by difficulty (verbatim lookups → "
        "cross-document reasoning and abstention).",
    },
}


def study_meta(study: str) -> dict:
    return STUDY_META.get(
        study,
        {
            "title": study.replace("_", " "),
            "metric": "accuracy",
            "imp_label": "+Prompt Eng",
            "imp_name": "Prompt engineering",
            "lead": (
                "Same dataset, one harness. Quality vs real token cost; "
                "baseline = zero-shot, +Prompt Eng = few-shot prompting."
            ),
            "diff_lead": "Per-difficulty breakdown of the baseline.",
        },
    )


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


def render_console(data_by_model: dict[str, dict], study: str) -> None:
    meta = study_meta(study)
    table = Table(title=f"Model comparison — {meta['title']} ({meta['metric']})", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column(f"Baseline {meta['metric']}", justify="right", style="green")
    table.add_column(meta["imp_label"], justify="right", style="green")
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


def _findings(data_by_model: dict[str, dict], meta: dict) -> list[str]:
    """Auto-generate interpretation bullets from the numbers."""
    metric, imp_name = meta["metric"], meta["imp_name"]
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
    out.append(f"**Best quality:** {top[0]} reaches {metric} {top[3]:.1%}.")

    gains = [(m, pe - base) for m, base, pe, _, _ in rows]
    mg = max(gains, key=lambda x: x[1])
    if mg[1] > 0.01:
        out.append(f"**{imp_name} helps most** on {mg[0]} ({mg[1] * 100:+.1f} {metric} points).")

    # A very low baseline that jumps with few-shot is usually a format artifact
    # (prose / invalid output the parser rejects), not weak capability. Only a
    # prompt-engineering story: for RAG a low closed-book baseline is the point.
    if imp_name == "Prompt engineering":
        fmt = [m for m, base, pe, _, _ in rows if base < 0.30 and (pe - base) > 0.30]
        if fmt:
            out.append(
                f"**Low baseline ≠ weak model:** {', '.join(fmt)} scored low at baseline mainly by "
                f"answering in an unstructured format the scorer rejects; few-shot standardised the "
                f"output (hence the large jump). Read baseline-vs-prompt as much about output format "
                f"as raw capability."
            )
    hurts = [m for m, delta in gains if delta < -0.01]
    if hurts:
        out.append(
            f"**{imp_name} *hurt*** {', '.join(hurts)} — the added context brought more "
            f"noise than signal."
        )

    paid = [r for r in rows if r[4] > 0]
    if len(paid) >= 2:
        cheapest = min(paid, key=lambda r: r[4])
        priciest = max(paid, key=lambda r: r[4])
        if priciest[0] != cheapest[0]:
            ratio = priciest[4] / cheapest[4]
            out.append(
                f"**Cost spread is huge:** {priciest[0]} costs ~{ratio:.0f}× more than "
                f"{cheapest[0]} for this run."
            )
        # value = quality per dollar, paid models only
        value = max(paid, key=lambda r: r[3] / r[4] if r[4] else 0)
        out.append(f"**Best value ({metric} per $):** {value[0]}.")

    # bigger-isn't-better: a cheaper model matching/beating a pricier one
    if len(paid) >= 2:
        s = sorted(paid, key=lambda r: r[4])
        if s[0][3] >= s[-1][3] - 0.02:
            out.append(
                f"**Bigger ≠ better:** {s[0][0]} matches the most expensive model "
                f"within 2 {metric} points at a fraction of the cost."
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


def render_markdown(data_by_model: dict[str, dict], study: str) -> str:
    meta = study_meta(study)
    lines = [
        f"# Model comparison — {meta['title']} ({meta['metric']})\n",
        f"| Model | Baseline {meta['metric']} | {meta['imp_label']} | Δ | Cost (USD) |",
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
        lines += [
            f"\n## Baseline {meta['metric']} by difficulty\n",
            "| Model | " + " | ".join(TIERS) + " |",
        ]
        lines.append("|---" * (len(TIERS) + 1) + "|")
        for model, by in diff:
            cells = " | ".join(f"{by[t]:.0%}" if t in by else "—" for t in TIERS)
            lines.append(f"| {model} | {cells} |")

    findings = _findings(data_by_model, meta)
    if findings:
        lines += ["\n## Findings\n"] + [f"- {f}" for f in findings]
    return "\n".join(lines) + "\n"


def render_html(data_by_model: dict[str, dict], study: str) -> str:
    meta = study_meta(study)
    metric = meta["metric"]
    # (model, baseline_quality, pe_quality, best_quality, cost) — même dérivation que _findings
    rows = []
    for model in sorted(data_by_model):
        d = data_by_model[model]
        base = d["baseline"]["accuracy"]
        _, imp = _best_improvement(d)
        pe = imp.get("accuracy", base)
        cost = imp.get("cost", d["baseline"].get("cost", 0))
        rows.append((model, base, pe, max(base, pe), cost))

    # Stat tiles
    tiles = ""
    if rows:
        top = max(rows, key=lambda r: r[3])
        mg = max(rows, key=lambda r: r[2] - r[1])
        tiles = report_theme.stat_tile("Models compared", str(len(rows)), "same harness")
        tiles += report_theme.stat_tile(f"Best {metric}", f"{top[3]:.1%}", html.escape(top[0]))
        tiles += report_theme.stat_tile(
            f"Biggest {meta['imp_name'].lower()} gain",
            f"{(mg[2] - mg[1]) * 100:+.1f} pts",
            html.escape(mg[0]),
            delta_good=(mg[2] - mg[1]) >= 0,
        )
        paid = [r for r in rows if r[4] > 0]
        if paid:
            value = max(paid, key=lambda r: r[3] / r[4])
            tiles += report_theme.stat_tile(
                f"Best value ({metric} per $)",
                html.escape(value[0]),
                f"{metric} {value[3]:.1%} for ${value[4]:.4f}",
            )
    tiles_html = f'<div class="tiles">{tiles}</div>' if tiles else ""

    # Graphiques
    bars = report_theme.chart_figure(
        f"{metric.capitalize()} by model — baseline vs {meta['imp_name'].lower()}",
        f"{metric.capitalize()} on the same dataset; longer is better.",
        [("Baseline", "var(--s1)"), (meta["imp_label"], "var(--s2)")],
        report_theme.svg_grouped_bars([(m, b, p) for m, b, p, _, _ in rows]),
    )
    scatter_svg = report_theme.svg_cost_quality_scatter([(m, c, best) for m, _, _, best, c in rows])
    scatter = (
        report_theme.chart_figure(
            "Cost vs quality",
            f"Best {metric} per model against its run cost — up and to the left wins.",
            [],
            scatter_svg,
        )
        if scatter_svg
        else ""
    )

    # Tableau global
    trows = ""
    for model, base, pe, _, cost in rows:
        delta = (pe - base) * 100
        cls = "pos" if delta >= 0 else "neg"
        trows += (
            f"<tr><td>{html.escape(model)}</td><td class='num'>{base:.1%}</td>"
            f"<td class='num'>{pe:.1%}</td><td class='num {cls}'>{delta:+.1f}</td>"
            f"<td class='num'>${cost:.4f}</td></tr>\n"
        )
    table_html = f"""<div class="tablewrap"><table>
<thead><tr><th>Model</th><th class="num">Baseline {metric}</th><th class="num">{meta["imp_label"]}</th>
<th class="num">Δ (pts)</th><th class="num">Cost (USD)</th></tr></thead>
<tbody>
{trows}</tbody></table></div>"""

    diff = _difficulty_rows(data_by_model)
    if diff:
        head = '<th class="num">' + '</th><th class="num">'.join(TIERS) + "</th>"
        body = ""
        for model, by in diff:
            cells = "".join(
                f"<td class='num'>{by[t]:.0%}</td>" if t in by else "<td class='num'>—</td>"
                for t in TIERS
            )
            body += f"<tr><td>{html.escape(model)}</td>{cells}</tr>\n"
        diff_html = f"""
<h2>Baseline {metric} by difficulty</h2>
<p class="lead">{meta["diff_lead"]}</p>
<div class="tablewrap"><table><thead><tr><th>Model</th>{head}</tr></thead><tbody>
{body}</tbody></table></div>"""
    else:
        diff_html = (
            f'<h2>Baseline {metric} by difficulty</h2><p class="lead"><em>No per-difficulty '
            "data in these runs — re-run the studies (the saver now captures it) to populate "
            "this table.</em></p>"
        )

    findings = _findings(data_by_model, meta)
    findings_html = ""
    if findings:
        items = ""
        for f in findings:
            parts = html.escape(f).split("**")
            styled = "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))
            items += f"<li>{styled}</li>"
        findings_html = f'<h2>Findings</h2><ul class="findings">{items}</ul>'

    head_html = report_theme.page_head(
        f"Model comparison — {meta['title']}",
        f"Cross-model benchmark: {metric} vs real token cost, "
        f"baseline vs {meta['imp_name'].lower()}.",
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head>
{head_html}
</head><body>
<div class="wrap">
<a class="crumb" href="index.html">&larr; LLM Diagnostic Framework</a>
<h1>📊 Model comparison — {html.escape(meta["title"])}</h1>
<p class="lead">{meta["lead"]}</p>
{tiles_html}
{bars}
{scatter}
<h2>Full results</h2>
{table_html}
{diff_html}
{findings_html}
<footer>Generated by <a href="https://github.com/MalekSnous/llm-diagnostic-framework">
LLM Diagnostic Framework</a> · <a href="index.html">Back to overview</a></footer>
</div>
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

    render_console(data, args.study)
    for f in _findings(data, study_meta(args.study)):
        console.print(f"  • {f.replace('**', '')}")

    DOCS.mkdir(exist_ok=True)
    html_path = DOCS / f"comparison_{args.study}.html"
    md_path = RESULTS / f"comparison_{args.study}.md"
    html_path.write_text(render_html(data, args.study), encoding="utf-8")
    md_path.write_text(render_markdown(data, args.study), encoding="utf-8")
    console.print(f"\n[green]HTML:[/green] {html_path}\n[green]Markdown:[/green] {md_path}")
    console.print(f"[cyan]Compared {len(data)} model(s): {', '.join(sorted(data))}[/cyan]")


if __name__ == "__main__":
    main()
