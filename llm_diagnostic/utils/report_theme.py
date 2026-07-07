"""
Design system partagé pour toutes les pages HTML publiées (GitHub Pages).

Expose le CSS commun (thème clair/sombre adaptatif via ``prefers-color-scheme``)
et des helpers qui génèrent des fragments HTML/SVG auto-contenus (pas de CDN) :
stat tiles, barres groupées (F1 baseline vs stratégie) et nuage coût-vs-qualité.

Utilisé par ``scripts/compare_models.py``, ``scripts/publish_reports.py`` et
``CaseStudyReporter`` pour que l'index, la comparaison et les rapports par
modèle partagent une seule identité visuelle.
"""

from __future__ import annotations

import html
import math
from typing import List, Optional, Sequence, Tuple

# --- Tokens -----------------------------------------------------------------
# Palette de données validée (script dataviz) : série 1 bleu, série 2 aqua.
# En clair l'aqua est < 3:1 sur la surface → toujours accompagner les marques
# de labels de valeur directs et d'un tableau (règle de "relief").

THEME_CSS = """
:root {
  --bg: #f6f8fa; --card: #ffffff; --card2: #eef1f4; --border: #d0d7de;
  --text: #1f2328; --muted: #59636e; --faint: #818b98;
  --accent: #4f63d2; --accent2: #8250df; --good: #1a7f37; --bad: #d1242f;
  --s1: #2a78d6; --s2: #1baf7a; --grid: #e4e7eb;
  --hero-from: #3b4db8; --hero-to: #7b3fd4;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --card: #161b22; --card2: #1f2630; --border: #30363d;
    --text: #e6edf3; --muted: #9da7b3; --faint: #6e7a87;
    --accent: #6e8efb; --accent2: #a777e3; --good: #3fb950; --bad: #f85149;
    --s1: #3987e5; --s2: #199e70; --grid: #262d36;
    --hero-from: #2b3a8f; --hero-to: #5b2fa3;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text); line-height: 1.6;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 960px; margin: 0 auto; padding: 0 20px; }
h1 { font-size: 1.9rem; margin: 0 0 10px; letter-spacing: -0.02em; }
h2 { font-size: 1.35rem; margin: 40px 0 8px; letter-spacing: -0.01em; }
.lead { color: var(--muted); margin: 0 0 20px; }
.crumb { font-size: .9rem; color: var(--muted); margin: 28px 0 18px; display: inline-block; }

/* Hero (bandeau haut de page) */
.hero {
  background: linear-gradient(135deg, var(--hero-from) 0%, var(--hero-to) 100%);
  color: #fff; padding: 56px 0 48px;
}
.hero h1 { color: #fff; }
.hero .lead { color: rgba(255,255,255,.85); max-width: 640px; }
.hero a.crumb { color: rgba(255,255,255,.8); margin: 0 0 14px; }

/* Stat tiles */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
         gap: 14px; margin: 22px 0; }
.tile { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
        padding: 16px 18px; }
.tile .label { font-size: .82rem; color: var(--muted); margin-bottom: 4px; }
.tile .value { font-size: 1.45rem; font-weight: 650; line-height: 1.2; }
.tile .detail { font-size: .82rem; color: var(--muted); margin-top: 4px;
                overflow-wrap: anywhere; }
.tile .delta-good { color: var(--good); font-weight: 600; }
.tile .delta-bad { color: var(--bad); font-weight: 600; }

/* Figures / charts */
figure.chart { margin: 22px 0; background: var(--card); border: 1px solid var(--border);
               border-radius: 12px; padding: 18px 18px 10px; }
figure.chart figcaption { font-weight: 600; margin-bottom: 2px; }
figure.chart .sub { font-size: .85rem; color: var(--muted); margin-bottom: 12px; }
figure.chart svg { width: 100%; height: auto; display: block; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: .84rem;
          color: var(--muted); margin: 6px 0 10px; }
.legend .key { display: inline-flex; align-items: center; gap: 7px; }
.legend .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

/* Tables */
.tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px;
             background: var(--card); margin: 16px 0; }
table { width: 100%; border-collapse: collapse; font-size: .95rem; }
th, td { padding: 11px 16px; text-align: left; border-bottom: 1px solid var(--border);
         white-space: nowrap; }
th { background: var(--card2); font-weight: 600; font-size: .86rem; color: var(--muted); }
tr:last-child td { border-bottom: none; }
tbody tr:hover { background: color-mix(in srgb, var(--accent) 6%, transparent); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: var(--good); font-weight: 600; }
.neg { color: var(--bad); font-weight: 600; }

/* Cards & callouts */
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
        padding: 20px; transition: border-color .15s, transform .15s; }
.card:hover { border-color: var(--accent); transform: translateY(-2px); }
.card h3 { margin: 0 0 6px; font-size: 1.05rem; }
.card p { color: var(--muted); margin: 0 0 12px; font-size: .92rem; }
.tag { display: inline-block; font-size: .72rem; font-weight: 600; padding: 2px 9px;
       border-radius: 999px; margin-bottom: 10px;
       background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); }
.callout { background: color-mix(in srgb, var(--accent) 7%, var(--card));
           border: 1px solid var(--border); border-left: 4px solid var(--accent);
           border-radius: 10px; padding: 16px 20px; margin: 16px 0; }
.callout h3 { margin: 0 0 6px; font-size: 1rem; }
.findings { list-style: none; padding: 0; margin: 12px 0; display: grid; gap: 10px; }
.findings li { background: var(--card); border: 1px solid var(--border);
               border-radius: 10px; padding: 12px 16px; font-size: .94rem; }
.findings li strong { color: var(--accent); }

/* Buttons */
.btns { display: flex; gap: 12px; flex-wrap: wrap; }
.btn { display: inline-block; padding: 11px 20px; border-radius: 10px; font-weight: 600;
       border: 1px solid var(--border); background: var(--card); color: var(--text); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn:hover { text-decoration: none; border-color: var(--accent); }
.hero .btn { background: rgba(255,255,255,.14); border-color: rgba(255,255,255,.45); color: #fff; }
.hero .btn.primary { background: #fff; border-color: #fff; color: var(--hero-from); }

code { background: var(--card2); padding: 2px 6px; border-radius: 5px; font-size: .88em; }
pre { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
      padding: 18px; overflow-x: auto; }
pre code { background: none; padding: 0; }
footer { padding: 34px 0 40px; color: var(--muted); font-size: .9rem;
         border-top: 1px solid var(--border); margin-top: 48px; }
"""

# --- Géométrie SVG -----------------------------------------------------------

_W = 720  # largeur logique (viewBox) de tous les graphiques


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Barre horizontale : base carrée à gauche, bout arrondi côté donnée."""
    r = min(r, w / 2, h / 2)
    return (
        f"M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} "
        f"v{h - 2 * r:.1f} a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"
    )


def page_head(title: str, description: str = "") -> str:
    """Bloc <head> complet partagé par les pages générées."""
    desc = f'\n<meta name="description" content="{_esc(description)}"/>' if description else ""
    return (
        '<meta charset="UTF-8"/>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0"/>\n'
        f"<title>{_esc(title)}</title>{desc}\n"
        f"<style>{THEME_CSS}</style>"
    )


def stat_tile(label: str, value: str, detail: str = "", delta_good: Optional[bool] = None) -> str:
    """Une carte KPI. ``delta_good`` colore le détail en bon/mauvais."""
    cls = "detail"
    if delta_good is True:
        cls = "detail delta-good"
    elif delta_good is False:
        cls = "detail delta-bad"
    detail_html = f'<div class="{cls}">{detail}</div>' if detail else ""
    return (
        f'<div class="tile"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div>{detail_html}</div>'
    )


def chart_figure(title: str, subtitle: str, legend: Sequence[Tuple[str, str]], svg: str) -> str:
    """Enrobe un SVG dans une figure avec titre, sous-titre et légende."""
    keys = "".join(
        f'<span class="key"><span class="swatch" style="background:{color}"></span>'
        f"{_esc(name)}</span>"
        for name, color in legend
    )
    legend_html = f'<div class="legend">{keys}</div>' if keys else ""
    return (
        f'<figure class="chart"><figcaption>{_esc(title)}</figcaption>'
        f'<div class="sub">{_esc(subtitle)}</div>{legend_html}{svg}</figure>'
    )


def svg_grouped_bars(rows: Sequence[Tuple[str, float, float]]) -> str:
    """
    Barres horizontales groupées : pour chaque (label, v1, v2) avec des valeurs
    dans [0, 1], deux barres (série 1 puis série 2) avec labels de valeur au bout.
    """
    left, right, top, bottom = 8, 64, 6, 28
    bar_h, pair_gap, label_h, group_gap = 16, 2, 20, 18
    group_h = label_h + bar_h * 2 + pair_gap
    plot_w = _W - left - right
    height = top + len(rows) * (group_h + group_gap) - group_gap + bottom

    parts: List[str] = []
    # Grille verticale (0/25/50/75/100 %) + ticks
    grid_bottom = height - bottom + 6
    for pct in (0, 25, 50, 75, 100):
        x = left + plot_w * pct / 100
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{grid_bottom}" '
            'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" '
            f'font-size="11" fill="var(--faint)">{pct}%</text>'
        )

    y = top
    for label, v1, v2 in rows:
        parts.append(
            f'<text x="{left}" y="{y + 12}" font-size="12.5" font-weight="600" '
            f'fill="var(--text)">{_esc(label)}</text>'
        )
        for value, color, name in ((v1, "var(--s1)", "Baseline"), (v2, "var(--s2)", "+Prompt Eng")):
            by = y + label_h if name == "Baseline" else y + label_h + bar_h + pair_gap
            w = max(plot_w * max(0.0, min(1.0, value)), 2)
            parts.append(
                f'<path d="{_bar_path(left, by, w, bar_h)}" fill="{color}">'
                f"<title>{_esc(label)} — {_esc(name)}: {value:.1%}</title></path>"
            )
            parts.append(
                f'<text x="{left + w + 7:.1f}" y="{by + bar_h - 4:.1f}" font-size="11.5" '
                f'fill="var(--muted)">{value:.1%}</text>'
            )
        y += group_h + group_gap

    return (
        f'<svg viewBox="0 0 {_W} {height}" role="img" '
        f'aria-label="Grouped bar chart">{"".join(parts)}</svg>'
    )


def svg_cost_quality_scatter(points: Sequence[Tuple[str, float, float]]) -> str:
    """
    Nuage coût (USD, échelle log) vs qualité (F1, [0,1]) : un point étiqueté par
    modèle. Les points à coût nul sont ignorés (log).
    """
    pts = [(label, cost, f1) for label, cost, f1 in points if cost > 0]
    if not pts:
        return ""

    height, left, right, top, bottom = 320, 56, 24, 14, 46
    plot_w, plot_h = _W - left - right, height - top - bottom

    lo = math.floor(math.log10(min(c for _, c, _ in pts)))
    hi = math.ceil(math.log10(max(c for _, c, _ in pts)))
    if hi == lo:
        hi = lo + 1

    def sx(cost: float) -> float:
        return left + plot_w * (math.log10(cost) - lo) / (hi - lo)

    def sy(f1: float) -> float:
        return top + plot_h * (1 - max(0.0, min(1.0, f1)))

    parts: List[str] = []
    for pct in (0, 25, 50, 75, 100):
        gy = sy(pct / 100)
        parts.append(
            f'<line x1="{left}" y1="{gy:.1f}" x2="{_W - right}" y2="{gy:.1f}" '
            'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="var(--faint)">{pct}%</text>'
        )
    for exp in range(lo, hi + 1):
        gx = sx(10**exp)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" '
            'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{height - 22}" text-anchor="middle" font-size="11" '
            f'fill="var(--faint)">${10.0 ** exp:g}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 5}" text-anchor="middle" '
        'font-size="11" fill="var(--muted)">Run cost (USD, log scale)</text>'
    )

    # Labels : centrés au-dessus du point ; on essaie plusieurs positions et on
    # garde la première qui ne chevauche ni un label déjà posé, ni un point.
    dot_boxes = [(sx(c) - 9, sy(f) - 9, sx(c) + 9, sy(f) + 9) for _, c, f in pts]
    label_boxes: List[Tuple[float, float, float, float]] = []

    def _overlaps(a: Tuple[float, float, float, float]) -> bool:
        boxes = label_boxes + dot_boxes
        return any(a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1] for b in boxes)

    for label, cost, f1 in sorted(pts, key=lambda p: p[1]):
        x, cy = sx(cost), sy(f1)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="6" fill="var(--s1)" '
            f'stroke="var(--card)" stroke-width="2">'
            f"<title>{_esc(label)} — F1 {f1:.1%} at ${cost:.4f}</title></circle>"
        )
        half_w = len(label) * 3.3  # ~6.6px par caractère à 11.5px
        lx = min(max(x, left + half_w), _W - right - half_w)
        for dy in (-12, 22, -26, 36, -40):
            box = (lx - half_w, cy + dy - 11, lx + half_w, cy + dy + 3)
            if not _overlaps(box):
                break
        label_boxes.append(box)
        parts.append(
            f'<text x="{lx:.1f}" y="{cy + dy:.1f}" text-anchor="middle" font-size="11.5" '
            f'fill="var(--muted)">{_esc(label)}</text>'
        )

    return (
        f'<svg viewBox="0 0 {_W} {height}" role="img" '
        f'aria-label="Cost versus quality scatter plot">{"".join(parts)}</svg>'
    )


def svg_strategy_bars(rows: Sequence[Tuple[str, float]]) -> str:
    """Barres horizontales simples (une série) : (stratégie, valeur [0,1])."""
    left, right, top, bottom = 8, 64, 6, 28
    bar_h, label_h, group_gap = 18, 20, 16
    plot_w = _W - left - right
    group_h = label_h + bar_h
    height = top + len(rows) * (group_h + group_gap) - group_gap + bottom

    parts: List[str] = []
    grid_bottom = height - bottom + 6
    for pct in (0, 25, 50, 75, 100):
        x = left + plot_w * pct / 100
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{grid_bottom}" '
            'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" '
            f'font-size="11" fill="var(--faint)">{pct}%</text>'
        )

    y = top
    for label, value in rows:
        parts.append(
            f'<text x="{left}" y="{y + 12}" font-size="12.5" font-weight="600" '
            f'fill="var(--text)">{_esc(label)}</text>'
        )
        w = max(plot_w * max(0.0, min(1.0, value)), 2)
        parts.append(
            f'<path d="{_bar_path(left, y + label_h, w, bar_h)}" fill="var(--s1)">'
            f"<title>{_esc(label)}: {value:.1%}</title></path>"
        )
        parts.append(
            f'<text x="{left + w + 7:.1f}" y="{y + label_h + bar_h - 5:.1f}" '
            f'font-size="11.5" fill="var(--muted)">{value:.1%}</text>'
        )
        y += group_h + group_gap

    return (
        f'<svg viewBox="0 0 {_W} {height}" role="img" '
        f'aria-label="Bar chart">{"".join(parts)}</svg>'
    )
