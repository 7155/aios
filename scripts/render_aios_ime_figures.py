#!/usr/bin/env python3
"""Render deterministic SVG figures for the AIOS-IME README.

The visual language follows the existing AIOS Graphviz and Matplotlib assets:
light backgrounds, rounded modules, restrained semantic colors, and explicit
arrows. The script uses only the Python standard library so the figures can be
reproduced without installing a plotting package.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path


FONT = "'DejaVu Sans', Helvetica, Arial, sans-serif"
MONO = "'Cascadia Mono', 'SFMono-Regular', Consolas, monospace"

COLORS = {
    "background": "#f8fafc",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#64748b",
    "line": "#607d8b",
    "border": "#455a64",
    "grid": "#dbe4ee",
    "slate": "#eceff1",
    "slate_dark": "#94a3b8",
    "blue": "#2563eb",
    "blue_fill": "#dbeafe",
    "cyan": "#0891b2",
    "cyan_fill": "#cffafe",
    "green": "#059669",
    "green_fill": "#d1fae5",
    "orange": "#d97706",
    "orange_fill": "#fef3c7",
    "red": "#dc2626",
    "red_fill": "#fee2e2",
    "violet": "#7c3aed",
    "violet_fill": "#ede9fe",
}


class Svg:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.width = width
        self.height = height
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" '
                'role="img" aria-labelledby="title desc">'
            ),
            f'<title id="title">{html.escape(title)}</title>',
            '<desc id="desc">Deterministically rendered AIOS-IME technical diagram.</desc>',
            "<defs>",
            (
                '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M 0 0 L 10 5 L 0 10 z" fill="#607d8b"/></marker>'
            ),
            (
                '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
                '<feDropShadow dx="0" dy="3" stdDeviation="5" '
                'flood-color="#0f172a" flood-opacity="0.10"/></filter>'
            ),
            "</defs>",
            f'<rect width="100%" height="100%" fill="{COLORS["background"]}"/>',
        ]

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = COLORS["panel"],
        stroke: str = COLORS["border"],
        stroke_width: float = 1.8,
        rx: float = 14,
        shadow: bool = False,
        dash: str | None = None,
    ) -> None:
        attrs = [
            f'x="{x}"',
            f'y="{y}"',
            f'width="{width}"',
            f'height="{height}"',
            f'rx="{rx}"',
            f'fill="{fill}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width}"',
        ]
        if shadow:
            attrs.append('filter="url(#shadow)"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.parts.append(f'<rect {" ".join(attrs)}/>')

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = COLORS["line"],
        width: float = 2.2,
        arrow: bool = False,
        dash: str | None = None,
    ) -> None:
        attrs = [
            f'x1="{x1}"',
            f'y1="{y1}"',
            f'x2="{x2}"',
            f'y2="{y2}"',
            f'stroke="{color}"',
            f'stroke-width="{width}"',
            'stroke-linecap="round"',
        ]
        if arrow:
            attrs.append('marker-end="url(#arrow)"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.parts.append(f'<line {" ".join(attrs)}/>')

    def path(
        self,
        d: str,
        *,
        color: str = COLORS["line"],
        width: float = 2.2,
        arrow: bool = False,
        dash: str | None = None,
        fill: str = "none",
    ) -> None:
        attrs = [
            f'd="{d}"',
            f'fill="{fill}"',
            f'stroke="{color}"',
            f'stroke-width="{width}"',
            'stroke-linecap="round"',
            'stroke-linejoin="round"',
        ]
        if arrow:
            attrs.append('marker-end="url(#arrow)"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.parts.append(f'<path {" ".join(attrs)}/>')

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        *,
        fill: str,
        stroke: str = COLORS["border"],
        stroke_width: float = 1.5,
    ) -> None:
        self.parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 18,
        fill: str = COLORS["ink"],
        weight: int = 500,
        anchor: str = "middle",
        family: str = FONT,
        line_height: float | None = None,
    ) -> None:
        lines = value.splitlines() or [""]
        step = line_height or size * 1.35
        first_y = y - (len(lines) - 1) * step / 2
        spans = "".join(
            f'<tspan x="{x}" y="{first_y + index * step}">{html.escape(line)}</tspan>'
            for index, line in enumerate(lines)
        )
        self.parts.append(
            f'<text text-anchor="{anchor}" dominant-baseline="middle" '
            f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{spans}</text>'
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([*self.parts, "</svg>", ""]), encoding="utf-8")


def title(svg: Svg, heading: str, subtitle: str) -> None:
    svg.text(48, 46, heading, size=30, weight=700, anchor="start", fill="#1e3a5f")
    svg.text(48, 82, subtitle, size=16, weight=400, anchor="start", fill=COLORS["muted"])


def module(
    svg: Svg,
    x: float,
    y: float,
    width: float,
    height: float,
    heading: str,
    detail: str,
    *,
    fill: str = COLORS["slate"],
    stroke: str = COLORS["border"],
    heading_color: str = COLORS["ink"],
) -> None:
    svg.rect(x, y, width, height, fill=fill, stroke=stroke, shadow=True)
    svg.text(x + width / 2, y + height * 0.40, heading, size=19, weight=700, fill=heading_color)
    svg.text(x + width / 2, y + height * 0.70, detail, size=13, weight=400, fill=COLORS["muted"])


def render_runtime(path: Path) -> None:
    svg = Svg(1600, 760, "AIOS-IME single-prefix multi-candidate runtime")
    title(
        svg,
        "AIOS-IME Chinese-Prefix Top-3 Inference",
        "Local single user · one keystroke · one Prefill · complete Top-3",
    )

    y, height = 145, 145
    modules = [
        (40, 170, "Chinese Prefix", "Raw text + BOS", COLORS["slate"], COLORS["border"]),
        (240, 190, "Tokenizer", "Retokenize · token-LCP", COLORS["cyan_fill"], COLORS["cyan"]),
        (460, 185, "Prefix Prefill", "Run once", COLORS["blue_fill"], COLORS["blue"]),
        (675, 215, "Shared Prefix KV", "Physical pages shared", COLORS["violet_fill"], COLORS["violet"]),
    ]
    for x, width, heading, detail, fill, stroke in modules:
        module(svg, x, y, width, height, heading, detail, fill=fill, stroke=stroke)
    for x1, x2 in ((210, 240), (430, 460), (645, 675)):
        svg.line(x1, y + height / 2, x2, y + height / 2, arrow=True)

    group_x, group_y, group_w, group_h = 930, 120, 260, 300
    svg.rect(
        group_x,
        group_y,
        group_w,
        group_h,
        fill=COLORS["panel"],
        stroke=COLORS["blue"],
        shadow=True,
    )
    svg.text(
        group_x + group_w / 2,
        group_y + 28,
        "CandidateGroup\nBatched Decode",
        size=15,
        weight=700,
        line_height=18,
    )
    for row in range(8):
        row_y = group_y + 55 + row * 28
        svg.rect(
            group_x + 18,
            row_y,
            group_w - 36,
            21,
            fill=COLORS["blue_fill"] if row < 5 else COLORS["slate"],
            stroke=COLORS["blue"] if row < 5 else COLORS["slate_dark"],
            stroke_width=1.0,
            rx=6,
        )
        svg.text(group_x + 36, row_y + 11, str(row + 1), size=11, weight=700, fill=COLORS["blue"])
        for token in range(6):
            token_x = group_x + 60 + token * 23
            svg.rect(
                token_x,
                row_y + 5,
                15,
                11,
                fill=COLORS["cyan"] if token < 3 + row % 3 else COLORS["grid"],
                stroke="none",
                stroke_width=0,
                rx=3,
            )
    svg.path(
        f"M 890 {y + height / 2} C 910 {y + height / 2}, 910 270, {group_x} 270",
        arrow=True,
    )
    rank_x, rank_y, rank_w, rank_h = 1215, 145, 200, 190
    module(
        svg,
        rank_x,
        rank_y,
        rank_w,
        rank_h,
        "Filter · Dedup · MMR",
        "Validity filter\nDisplay dedup\nRaw logprob\nMMR",
        fill=COLORS["orange_fill"],
        stroke=COLORS["orange"],
    )
    svg.line(group_x + group_w, 270, rank_x, 270, arrow=True)

    top_x, top_y, top_w, top_h = 1445, 145, 115, 190
    svg.rect(top_x, top_y, top_w, top_h, fill=COLORS["green_fill"], stroke=COLORS["green"], shadow=True)
    svg.text(top_x + top_w / 2, top_y + 28, "Top-3", size=20, weight=700, fill=COLORS["green"])
    for index in range(3):
        card_y = top_y + 53 + index * 42
        svg.rect(top_x + 15, card_y, top_w - 30, 30, fill="#ffffff", stroke=COLORS["green"], rx=8)
        svg.text(top_x + top_w / 2, card_y + 16, f"Candidate {index + 1}", size=12, weight=600)
    svg.line(rank_x + rank_w, 240, top_x, 240, arrow=True)

    svg.text(1060, 445, "EOS / sentence-end rows exit immediately; no padded Decode", size=14, weight=500, fill=COLORS["muted"])

    cards = [
        (
            40,
            "latest-wins",
            "Generation N becomes stale\nRelease old suffix KV after current token step",
            COLORS["red_fill"],
            COLORS["red"],
        ),
        (
            435,
            "KV ownership",
            "Prefix pages shared by candidate group\nSuffix pages owned per candidate",
            COLORS["violet_fill"],
            COLORS["violet"],
        ),
        (
            830,
            "On-demand refill",
            "Start with 8 rows; sample 4 more\nonly when fewer than 3 survive",
            COLORS["orange_fill"],
            COLORS["orange"],
        ),
        (
            1225,
            "Top-3 diversity selection",
            "Raw LM score + soft penalties\nCharacter-bigram MMR",
            COLORS["green_fill"],
            COLORS["green"],
        ),
    ]
    for x, heading, detail, fill, stroke in cards:
        module(svg, x, 525, 335, 155, heading, detail, fill=fill, stroke=stroke)

    svg.save(path)


def bar(
    svg: Svg,
    *,
    x: float,
    y: float,
    value: float,
    maximum: float,
    width: float,
    color: str,
    label: str,
    value_label: str,
) -> None:
    svg.text(x - 22, y + 14, label, size=16, weight=600, anchor="end")
    svg.rect(x, y, width, 28, fill="#e2e8f0", stroke="none", stroke_width=0, rx=6)
    svg.rect(x, y, width * value / maximum, 28, fill=color, stroke="none", stroke_width=0, rx=6)
    svg.text(x + width + 18, y + 15, value_label, size=16, weight=700, anchor="start", fill=color)


def render_performance(path: Path) -> None:
    svg = Svg(1500, 760, "AIOS-IME end-to-end Top-3 latency benchmark")
    title(
        svg,
        "AIOS-IME Chinese-Prefix Top-3 Latency",
        "NVIDIA GeForce RTX 4080 Laptop GPU · BF16 · 5 warmups · 30 timed prompts",
    )

    chart_x, chart_y, chart_w, chart_h = 55, 125, 1080, 450
    svg.rect(chart_x, chart_y, chart_w, chart_h, fill=COLORS["panel"], stroke=COLORS["grid"], shadow=True)
    plot_x, plot_w, maximum = 315, 650, 300.0
    for tick in (0, 100, 200, 300):
        tick_x = plot_x + plot_w * tick / maximum
        svg.line(tick_x, 165, tick_x, 530, color=COLORS["grid"], width=1.2, dash="4 6")
        svg.text(tick_x, 150, f"{tick} ms", size=12, weight=400, fill=COLORS["muted"])

    svg.text(90, 235, "p50", size=32, weight=700, anchor="start", fill=COLORS["violet"])
    bar(
        svg,
        x=plot_x,
        y=195,
        value=258.64,
        maximum=maximum,
        width=plot_w,
        color=COLORS["slate_dark"],
        label="MiniMind PyTorch",
        value_label="258.64 ms",
    )
    bar(
        svg,
        x=plot_x,
        y=245,
        value=81.98,
        maximum=maximum,
        width=plot_w,
        color=COLORS["blue"],
        label="AIOS-IME",
        value_label="81.98 ms",
    )
    svg.rect(90, 295, 970, 2, fill=COLORS["grid"], stroke="none", stroke_width=0, rx=0)

    svg.text(90, 400, "p95", size=32, weight=700, anchor="start", fill=COLORS["violet"])
    bar(
        svg,
        x=plot_x,
        y=360,
        value=279.43,
        maximum=maximum,
        width=plot_w,
        color=COLORS["slate_dark"],
        label="MiniMind PyTorch",
        value_label="279.43 ms",
    )
    bar(
        svg,
        x=plot_x,
        y=410,
        value=109.97,
        maximum=maximum,
        width=plot_w,
        color=COLORS["blue"],
        label="AIOS-IME",
        value_label="109.97 ms",
    )
    svg.text(1090, 285, "Complete Top-3 wall-clock latency", size=13, weight=400, anchor="end", fill=COLORS["muted"])
    svg.text(1090, 310, "Excludes model loading and first JIT", size=13, weight=400, anchor="end", fill=COLORS["muted"])

    module(
        svg,
        1175,
        145,
        270,
        180,
        "p50 · 3.15×",
        "258.64 ms\n↓\n81.98 ms",
        fill=COLORS["blue_fill"],
        stroke=COLORS["blue"],
        heading_color=COLORS["blue"],
    )
    module(
        svg,
        1175,
        360,
        270,
        180,
        "p95 · 2.54×",
        "279.43 ms\n↓\n109.97 ms",
        fill=COLORS["cyan_fill"],
        stroke=COLORS["cyan"],
        heading_color=COLORS["cyan"],
    )

    metrics = [
        (55, "Peak allocated", "227.10 MiB", COLORS["violet_fill"], COLORS["violet"]),
        (420, "Complete Top-3", "100%", COLORS["green_fill"], COLORS["green"]),
        (785, "Distinct candidates", "100%", COLORS["green_fill"], COLORS["green"]),
        (1150, "Low-memory KV profile", "256 pages", COLORS["orange_fill"], COLORS["orange"]),
    ]
    for x, heading, detail, fill, stroke in metrics:
        module(svg, x, 615, 295, 100, heading, detail, fill=fill, stroke=stroke, heading_color=stroke)

    svg.save(path)


def render_model(path: Path) -> None:
    svg = Svg(1600, 900, "MiniMind-IME 0.1B deployment model architecture")
    title(
        svg,
        "MiniMind-IME 0.1B Deployment Model",
        "100,687,360 online parameters · Dense Decoder-only Transformer · BF16",
    )

    top_y, top_h = 135, 125
    module(svg, 45, top_y, 155, top_h, "Token IDs", "BOS + Chinese context")
    module(svg, 250, top_y, 190, top_h, "Tied Embedding", "16,384 × 768", fill=COLORS["blue_fill"], stroke=COLORS["blue"])
    for offset in (18, 12, 6):
        svg.rect(495 + offset, top_y - offset, 315, top_h, fill=COLORS["violet_fill"], stroke=COLORS["violet"], rx=14)
    module(svg, 495, top_y, 315, top_h, "Decoder Block × 14", "GQA Attention + SwiGLU MLP", fill=COLORS["violet_fill"], stroke=COLORS["violet"])
    module(svg, 860, top_y, 175, top_h, "Final RMSNorm", "eps = 1e-6", fill=COLORS["cyan_fill"], stroke=COLORS["cyan"])
    module(svg, 1085, top_y, 190, top_h, "Tied LM Head", "Shares embedding weights", fill=COLORS["blue_fill"], stroke=COLORS["blue"])
    module(svg, 1325, top_y, 225, top_h, "Next-token logits", "Vocabulary = 16,384", fill=COLORS["green_fill"], stroke=COLORS["green"])
    for x1, x2 in ((200, 250), (440, 495), (828, 860), (1035, 1085), (1275, 1325)):
        svg.line(x1, top_y + top_h / 2, x2, top_y + top_h / 2, arrow=True)
    svg.path("M 345 272 C 345 320, 1178 320, 1178 272", color=COLORS["blue"], width=2, dash="6 5")
    svg.text(760, 308, "tied weights", size=13, weight=600, fill=COLORS["blue"])

    spec_x, spec_y, spec_w, spec_h = 45, 370, 245, 430
    svg.rect(spec_x, spec_y, spec_w, spec_h, fill=COLORS["panel"], stroke=COLORS["grid"], shadow=True)
    svg.text(spec_x + 24, spec_y + 35, "Deployment profile", size=21, weight=700, anchor="start", fill="#1e3a5f")
    specs = [
        ("Hidden size", "768"),
        ("Intermediate", "2,048"),
        ("Q / KV heads", "12 / 4"),
        ("Head dim", "64"),
        ("Context", "512 tokens"),
        ("Precision", "BF16"),
        ("Weights", "192.05 MiB"),
        ("KV / token", "14 KiB"),
    ]
    for index, (name, value) in enumerate(specs):
        row_y = spec_y + 80 + index * 40
        svg.text(spec_x + 24, row_y, name, size=13, weight=400, anchor="start", fill=COLORS["muted"])
        svg.text(spec_x + spec_w - 24, row_y, value, size=14, weight=700, anchor="end")
        if index < len(specs) - 1:
            svg.line(spec_x + 24, row_y + 20, spec_x + spec_w - 24, row_y + 20, color=COLORS["grid"], width=1)

    block_x, block_y, block_w, block_h = 335, 365, 1215, 445
    svg.rect(block_x, block_y, block_w, block_h, fill=COLORS["panel"], stroke=COLORS["violet"], shadow=True)
    svg.text(block_x + 28, block_y + 35, "Single Decoder Block", size=21, weight=700, anchor="start", fill=COLORS["violet"])
    svg.text(block_x + block_w - 28, block_y + 35, "Pre-Norm · Residual · Full Attention", size=13, weight=500, anchor="end", fill=COLORS["muted"])

    flow_y = 570
    components = [
        (380, 115, "Input", "x", COLORS["slate"], COLORS["border"]),
        (535, 135, "RMSNorm", "", COLORS["cyan_fill"], COLORS["cyan"]),
        (710, 220, "GQA Attention", "QK Norm · RoPE\n12 Q heads · 4 KV heads", COLORS["blue_fill"], COLORS["blue"]),
        (985, 58, "+", "", COLORS["green_fill"], COLORS["green"]),
        (1085, 135, "RMSNorm", "", COLORS["cyan_fill"], COLORS["cyan"]),
        (1260, 220, "SwiGLU MLP", "768 → 2048 → 768", COLORS["orange_fill"], COLORS["orange"]),
    ]
    for x, width, heading, detail, fill, stroke in components:
        module(svg, x, flow_y, width, 105, heading, detail, fill=fill, stroke=stroke, heading_color=stroke)
    for x1, x2 in ((495, 535), (670, 710), (930, 985), (1043, 1085), (1220, 1260)):
        svg.line(x1, flow_y + 52, x2, flow_y + 52, arrow=True)

    svg.path(
        f"M 437 {flow_y} C 437 480, 1014 480, 1014 {flow_y}",
        color=COLORS["green"],
        width=2.2,
        arrow=True,
    )
    svg.text(725, 472, "residual", size=13, weight=600, fill=COLORS["green"])

    output_x = 1500
    module(svg, output_x, flow_y, 40, 105, "", "", fill=COLORS["green_fill"], stroke=COLORS["green"])
    svg.text(output_x + 20, flow_y + 53, "+", size=25, weight=700, fill=COLORS["green"])
    svg.line(1480, flow_y + 52, output_x, flow_y + 52, arrow=True)
    svg.path(
        f"M 1014 {flow_y + 105} C 1014 745, 1520 745, 1520 {flow_y + 105}",
        color=COLORS["green"],
        width=2.2,
        arrow=True,
    )
    svg.text(1267, 760, "residual", size=13, weight=600, fill=COLORS["green"])

    svg.text(
        942,
        850,
        "MTP auxiliary weights are stripped during export; deployment keeps the 100.69M backbone only",
        size=15,
        weight=500,
        fill=COLORS["muted"],
    )
    svg.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_runtime(args.output_dir / "aios-ime-runtime-architecture.svg")
    render_performance(args.output_dir / "aios-ime-performance.svg")
    render_model(args.output_dir / "minimind-ime-model-architecture.svg")
    for name in (
        "aios-ime-runtime-architecture.svg",
        "aios-ime-performance.svg",
        "minimind-ime-model-architecture.svg",
    ):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
