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


FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif"
CJK_FONT = "'Noto Sans CJK SC', 'Source Han Sans SC', 'PingFang SC', 'Microsoft YaHei', 'Droid Sans Fallback', sans-serif"
MONO = "'Cascadia Mono', 'SFMono-Regular', 'Roboto Mono', Consolas, monospace"

COLORS = {
    "background": "#f6f8fc",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#52627a",
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
                'role="img" aria-labelledby="title desc" '
                'text-rendering="geometricPrecision" shape-rendering="geometricPrecision">'
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
        size: float = 20,
        fill: str = COLORS["ink"],
        weight: int = 500,
        anchor: str = "middle",
        family: str = FONT,
        line_height: float | None = None,
    ) -> None:
        lines = value.splitlines() or [""]
        step = line_height or size * 1.30
        first_y = y - (len(lines) - 1) * step / 2
        def line_family(line: str) -> str:
            if family != FONT:
                return family
            has_cjk = any("\u3400" <= character <= "\u9fff" for character in line)
            return CJK_FONT if has_cjk else FONT

        spans = "".join(
            f'<tspan x="{x}" y="{first_y + index * step}" '
            f'font-family="{line_family(line)}">{html.escape(line)}</tspan>'
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


def _line_units(value: str) -> float:
    """Estimate rendered width in em units for responsive SVG labels."""
    units = 0.0
    for character in value:
        if "\u3400" <= character <= "\u9fff":
            units += 1.0
        elif character.isspace():
            units += 0.34
        elif character in "·.,:;/()[]×+-→↓":
            units += 0.45
        elif character.isupper() or character.isdigit():
            units += 0.64
        else:
            units += 0.56
    return units


def fit_text_size(value: str, available_width: float, desired: float, minimum: float) -> float:
    longest_line = max(value.splitlines() or [""], key=_line_units)
    units = max(_line_units(longest_line), 1.0)
    return max(minimum, min(desired, available_width / units))


def title(svg: Svg, heading: str, subtitle: str) -> None:
    svg.rect(38, 22, 7, 72, fill=COLORS["blue"], stroke="none", stroke_width=0, rx=4)
    svg.text(62, 46, heading, size=37, weight=750, anchor="start", fill="#17365d")
    svg.text(62, 88, subtitle, size=22, weight=500, anchor="start", fill=COLORS["muted"])


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
    heading_desired = min(24.0, 18.0 + height / 30.0)
    detail_desired = min(19.5, 14.0 + height / 35.0)
    heading_size = fit_text_size(heading, width - 28, heading_desired, 18.0)
    detail_size = fit_text_size(detail, width - 24, detail_desired, 14.0)
    svg.text(
        x + width / 2,
        y + height * 0.37,
        heading,
        size=heading_size,
        weight=700,
        fill=heading_color,
        line_height=heading_size * 1.18,
    )
    svg.text(
        x + width / 2,
        y + height * 0.72,
        detail,
        size=detail_size,
        weight=500,
        fill=COLORS["muted"],
        line_height=detail_size * 1.22,
    )


def render_runtime(path: Path) -> None:
    svg = Svg(1400, 560, "AIOS-IME Chinese-prefix Top-3 runtime overview")
    title(
        svg,
        "AIOS-IME Chinese-Prefix Top-3 Inference",
        "Local single user · one keystroke · one Prefill · complete Top-3",
    )

    y, height = 155, 190
    modules = [
        (40, 170, "Chinese Prefix", "没关系，\n你先忙你的，", COLORS["slate"], COLORS["border"]),
        (245, 180, "Tokenizer", "Retokenize\ntoken-LCP", COLORS["cyan_fill"], COLORS["cyan"]),
        (460, 180, "Prefix Prefill", "Run exactly\nonce", COLORS["blue_fill"], COLORS["blue"]),
        (675, 210, "Candidate\nGroup", "8 independent\ndecode rows", COLORS["violet_fill"], COLORS["violet"]),
        (920, 220, "Filter · Dedup\nMMR", "Raw LM score\nTop-3", COLORS["orange_fill"], COLORS["orange"]),
        (1175, 180, "Candidate Bar", "明确答复 / 发消息\n/ …", COLORS["green_fill"], COLORS["green"]),
    ]
    for x, width, heading, detail, fill, stroke in modules:
        module(svg, x, y, width, height, heading, detail, fill=fill, stroke=stroke)
    for x1, x2 in ((210, 245), (425, 460), (640, 675), (885, 920), (1140, 1175)):
        svg.line(x1, y + height / 2, x2, y + height / 2, arrow=True)

    svg.text(
        700,
        425,
        "The main path generates Chinese suffixes directly; pinyin lexicon recall is not required.",
        size=22,
        weight=500,
        fill=COLORS["muted"],
    )
    svg.save(path)


def render_candidate_group(path: Path) -> None:
    svg = Svg(1400, 700, "AIOS-IME candidate group and Top-3 selection")
    title(
        svg,
        "CandidateGroup: Generate More, Display Top-3",
        "Shared Prefix KV · ragged batched Decode · deadline-aware refill",
    )

    module(
        svg,
        45,
        220,
        260,
        170,
        "中文前缀",
        "没关系，\n你先忙你的，\n\n只预填一次",
        fill=COLORS["violet_fill"],
        stroke=COLORS["violet"],
    )

    group_x, group_y, group_w, group_h = 345, 125, 410, 420
    svg.rect(group_x, group_y, group_w, group_h, fill=COLORS["panel"], stroke=COLORS["blue"], shadow=True)
    svg.text(group_x + group_w / 2, group_y + 40, "八路独立候选解码", size=27, weight=700)
    for row in range(8):
        row_y = group_y + 75 + row * 36
        svg.rect(
            group_x + 25,
            row_y,
            group_w - 50,
            27,
            fill=COLORS["blue_fill"] if row < 5 else COLORS["slate"],
            stroke=COLORS["blue"] if row < 5 else COLORS["slate_dark"],
            stroke_width=1.2,
            rx=6,
        )
        svg.text(group_x + 47, row_y + 14, str(row + 1), size=20, weight=700, fill=COLORS["blue"])
        for token in range(8):
            token_x = group_x + 82 + token * 34
            svg.rect(
                token_x,
                row_y + 6,
                23,
                15,
                fill=COLORS["cyan"] if token < 3 + row % 4 else COLORS["grid"],
                stroke="none",
                stroke_width=0,
                rx=4,
            )
    svg.line(305, 305, group_x, 305, arrow=True)

    module(
        svg,
        810,
        220,
        210,
        170,
        "Filter · Dedup\nMMR",
        "中文合法性\n显示去重\n原始分数与多样性",
        fill=COLORS["orange_fill"],
        stroke=COLORS["orange"],
    )
    svg.line(group_x + group_w, 305, 810, 305, arrow=True)

    top_x, top_y, top_w, top_h = 1075, 145, 280, 350
    svg.rect(top_x, top_y, top_w, top_h, fill=COLORS["green_fill"], stroke=COLORS["green"], shadow=True)
    svg.text(top_x + top_w / 2, top_y + 40, "Top-3", size=31, weight=700, fill=COLORS["green"])
    candidates = [
        "我晚点给你一个\n明确答复。",
        "我晚点给你发消息。",
        "我晚一点给你一个\n明确答复。",
    ]
    for index, candidate in enumerate(candidates):
        card_y = top_y + 75 + index * 82
        svg.rect(top_x + 18, card_y, top_w - 36, 66, fill="#ffffff", stroke=COLORS["green"], rx=8)
        svg.text(top_x + top_w / 2, card_y + 34, candidate, size=22, weight=650)
    svg.line(1020, 305, top_x, 305, arrow=True)

    svg.text(700, 615, "结束分支立即离开批次 · 首轮生成八路 · 有效且互异候选不足三条时才补四路", size=22, weight=500, fill=COLORS["muted"])

    svg.save(path)


def render_prefix_kv(path: Path) -> None:
    svg = Svg(1400, 760, "AIOS-IME token-LCP Prefix KV reuse and latest-wins")
    title(
        svg,
        "Across Keystrokes: Token-LCP KV Reuse",
        "Retokenize first · reuse stable token pages only · latest generation wins",
    )

    svg.text(55, 135, "上一次按键：没关系，你先忙你", size=25, weight=700, anchor="start", fill="#1e3a5f")
    svg.text(55, 285, "当前按键：没关系，你先忙你的，", size=25, weight=700, anchor="start", fill="#1e3a5f")
    old_tokens = ["3497", "243", "192", "144", "297", "518", "3035", "297"]
    new_tokens = ["3497", "243", "192", "144", "297", "518", "3559", "243", "192", "144"]
    token_x, token_w, token_gap = 315, 72, 10
    for index, value in enumerate(old_tokens):
        x = token_x + index * (token_w + token_gap)
        svg.rect(x, 160, token_w, 52, fill=COLORS["blue_fill"], stroke=COLORS["blue"], rx=9)
        svg.text(x + token_w / 2, 187, value, size=21, weight=700, family=MONO)
    for index, value in enumerate(new_tokens):
        x = token_x + index * (token_w + token_gap)
        stable = index < 6
        svg.rect(
            x,
            310,
            token_w,
            52,
            fill=COLORS["green_fill"] if stable else COLORS["orange_fill"],
            stroke=COLORS["green"] if stable else COLORS["orange"],
            rx=9,
        )
        svg.text(x + token_w / 2, 337, value, size=21, weight=700, family=MONO)

    svg.text(235, 187, "token IDs", size=21, weight=650, fill=COLORS["muted"])
    svg.text(235, 337, "token IDs", size=21, weight=650, fill=COLORS["muted"])
    module(svg, 180, 405, 300, 90, "token-LCP = 6", "Reuse first 6 stable tokens", fill=COLORS["green_fill"], stroke=COLORS["green"])
    module(svg, 555, 405, 385, 90, "Tokenizer", "尾部发生重切\n3035 + 297  →  3559 + …", fill=COLORS["orange_fill"], stroke=COLORS["orange"])
    module(svg, 1015, 405, 300, 90, "重新计算尾部", "不按字符长度复用错误缓存", fill=COLORS["violet_fill"], stroke=COLORS["violet"])
    svg.line(480, 450, 555, 450, arrow=True)
    svg.line(940, 450, 1015, 450, arrow=True)

    svg.rect(45, 555, 1310, 155, fill=COLORS["panel"], stroke=COLORS["grid"], shadow=True)
    svg.text(75, 590, "latest-wins", size=26, weight=700, anchor="start", fill=COLORS["red"])
    module(svg, 210, 605, 235, 70, "上一次按键", "旧候选组失效", fill=COLORS["red_fill"], stroke=COLORS["red"])
    module(svg, 580, 605, 260, 70, "token-step boundary", "丢弃旧输出并释放缓存", fill=COLORS["orange_fill"], stroke=COLORS["orange"])
    module(svg, 975, 605, 300, 70, "当前按键", "唯一有效的候选组", fill=COLORS["green_fill"], stroke=COLORS["green"])
    svg.line(445, 640, 580, 640, arrow=True)
    svg.line(840, 640, 975, 640, arrow=True)

    svg.save(path)


def render_vllm_comparison(path: Path) -> None:
    svg = Svg(1400, 690, "vLLM general serving compared with AIOS-IME local workload")
    title(
        svg,
        "Same Inference Primitives, Different Workload",
        "vLLM optimizes general serving throughput · AIOS-IME optimizes one keystroke's complete Top-3 p95",
    )

    left_x, right_x, panel_y, panel_w, panel_h = 45, 755, 125, 600, 470
    svg.rect(left_x, panel_y, panel_w, panel_h, fill=COLORS["panel"], stroke=COLORS["blue"], shadow=True)
    svg.rect(right_x, panel_y, panel_w, panel_h, fill=COLORS["panel"], stroke=COLORS["green"], shadow=True)
    svg.text(left_x + panel_w / 2, 165, "vLLM · General Serving", size=29, weight=700, fill=COLORS["blue"])
    svg.text(right_x + panel_w / 2, 165, "AIOS-IME · Local IME", size=29, weight=700, fill=COLORS["green"])

    for index, label in enumerate(("User A", "User B", "User C")):
        module(svg, 80, 215 + index * 105, 130, 72, label, "独立请求", fill=COLORS["blue_fill"], stroke=COLORS["blue"])
        svg.line(210, 251 + index * 105, 300, 330, arrow=True)
    module(svg, 300, 245, 260, 170, "Continuous Batching", "跨请求调度\n提升 GPU 吞吐", fill=COLORS["cyan_fill"], stroke=COLORS["cyan"])
    module(svg, 300, 455, 260, 80, "优化目标", "tokens/s · request throughput", fill=COLORS["slate"], stroke=COLORS["slate_dark"])

    module(svg, 790, 215, 220, 100, "连续按键", "…你先忙你\n→ …你先忙你的，", fill=COLORS["green_fill"], stroke=COLORS["green"])
    module(svg, 1085, 215, 230, 100, "latest-wins", "只保留当前 generation", fill=COLORS["red_fill"], stroke=COLORS["red"])
    svg.line(1010, 265, 1085, 265, arrow=True)
    module(svg, 790, 360, 220, 120, "Prefix Prefill Once", "Reuse token-LCP KV", fill=COLORS["violet_fill"], stroke=COLORS["violet"])
    module(svg, 1085, 360, 230, 120, "内部候选组", "八路解码，显示三条", fill=COLORS["orange_fill"], stroke=COLORS["orange"])
    svg.line(1010, 420, 1085, 420, arrow=True)
    module(svg, 900, 490, 300, 90, "优化目标", "完整三条候选尾延迟\n显存与正确性", fill=COLORS["green_fill"], stroke=COLORS["green"])

    svg.text(700, 635, "Paged KV · Prefix Cache · Parallel Sampling are shared primitives; scheduling, lifecycle, ranking, and metrics differ.", size=23, weight=500, fill=COLORS["muted"])
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
    svg.text(x - 22, y + 14, label, size=22, weight=650, anchor="end")
    svg.rect(x, y, width, 28, fill="#e2e8f0", stroke="none", stroke_width=0, rx=6)
    svg.rect(x, y, width * value / maximum, 28, fill=color, stroke="none", stroke_width=0, rx=6)
    svg.text(x + width + 18, y + 15, value_label, size=22, weight=700, anchor="start", fill=color)


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
        svg.text(tick_x, 150, f"{tick} ms", size=19, weight=500, fill=COLORS["muted"])

    svg.text(90, 235, "p50", size=36, weight=700, anchor="start", fill=COLORS["violet"])
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

    svg.text(90, 400, "p95", size=36, weight=700, anchor="start", fill=COLORS["violet"])
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
    svg.text(1090, 285, "Complete Top-3 wall-clock latency", size=20, weight=500, anchor="end", fill=COLORS["muted"])
    svg.text(1090, 310, "Excludes model loading and first JIT", size=20, weight=500, anchor="end", fill=COLORS["muted"])

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
    svg = Svg(1400, 520, "MiniMind-IME 0.1B deployment model overview")
    title(
        svg,
        "MiniMind-IME 0.1B Deployment Model",
        "100,687,360 online parameters · Dense Decoder-only Transformer · BF16",
    )

    top_y, top_h = 145, 140
    module(svg, 40, top_y, 150, top_h, "Token IDs", "BOS + context")
    module(svg, 230, top_y, 180, top_h, "Tied\nEmbedding", "16,384 × 768", fill=COLORS["blue_fill"], stroke=COLORS["blue"])
    for offset in (16, 10, 5):
        svg.rect(450 + offset, top_y - offset, 280, top_h, fill=COLORS["violet_fill"], stroke=COLORS["violet"], rx=14)
    module(svg, 450, top_y, 280, top_h, "Decoder × 14", "GQA + SwiGLU", fill=COLORS["violet_fill"], stroke=COLORS["violet"])
    module(svg, 770, top_y, 170, top_h, "Final\nRMSNorm", "eps = 1e-6", fill=COLORS["cyan_fill"], stroke=COLORS["cyan"])
    module(svg, 980, top_y, 180, top_h, "Tied LM\nHead", "Shared weights", fill=COLORS["blue_fill"], stroke=COLORS["blue"])
    module(svg, 1200, top_y, 160, top_h, "Logits", "Vocab 16,384", fill=COLORS["green_fill"], stroke=COLORS["green"])
    for x1, x2 in ((190, 230), (410, 450), (746, 770), (940, 980), (1160, 1200)):
        svg.line(x1, top_y + top_h / 2, x2, top_y + top_h / 2, arrow=True)
    svg.path("M 320 300 C 320 345, 1070 345, 1070 300", color=COLORS["blue"], width=2, dash="6 5")
    svg.text(695, 337, "tied weights", size=20, weight=600, fill=COLORS["blue"])

    specs = [
        (40, "100.69M", "online parameters", COLORS["violet_fill"], COLORS["violet"]),
        (375, "14 layers", "hidden 768 · MLP 2,048", COLORS["blue_fill"], COLORS["blue"]),
        (710, "12 Q / 4 KV", "GQA · head dim 64", COLORS["cyan_fill"], COLORS["cyan"]),
        (1045, "512 tokens", "standard RoPE · no YaRN", COLORS["green_fill"], COLORS["green"]),
    ]
    for x, heading, detail, fill, stroke in specs:
        module(svg, x, 385, 300, 90, heading, detail, fill=fill, stroke=stroke, heading_color=stroke)

    svg.save(path)


def render_decoder_block(path: Path) -> None:
    svg = Svg(1400, 600, "MiniMind-IME single decoder block")
    title(
        svg,
        "MiniMind-IME Single Decoder Block",
        "Pre-Norm · residual connections · full attention · GQA",
    )

    flow_y = 245
    components = [
        (40, 120, "Input", "x", COLORS["slate"], COLORS["border"]),
        (210, 150, "RMSNorm", "", COLORS["cyan_fill"], COLORS["cyan"]),
        (410, 250, "GQA Attention", "QK Norm · RoPE\n12 Q heads · 4 KV heads", COLORS["blue_fill"], COLORS["blue"]),
        (710, 70, "+", "", COLORS["green_fill"], COLORS["green"]),
        (830, 150, "RMSNorm", "", COLORS["cyan_fill"], COLORS["cyan"]),
        (1030, 250, "SwiGLU MLP", "768 → 2,048 → 768", COLORS["orange_fill"], COLORS["orange"]),
        (1330, 50, "+", "", COLORS["green_fill"], COLORS["green"]),
    ]
    for x, width, heading, detail, fill, stroke in components:
        module(svg, x, flow_y, width, 120, heading, detail, fill=fill, stroke=stroke, heading_color=stroke)
    for x1, x2 in ((160, 210), (360, 410), (660, 710), (780, 830), (980, 1030), (1280, 1330)):
        svg.line(x1, flow_y + 60, x2, flow_y + 60, arrow=True)

    svg.path(
        f"M 100 {flow_y} C 100 145, 745 145, 745 {flow_y}",
        color=COLORS["green"],
        width=2.5,
        arrow=True,
    )
    svg.text(420, 155, "attention residual", size=23, weight=650, fill=COLORS["green"])
    svg.path(
        f"M 745 {flow_y + 120} C 745 490, 1355 490, 1355 {flow_y + 120}",
        color=COLORS["green"],
        width=2.5,
        arrow=True,
    )
    svg.text(1050, 505, "MLP residual", size=23, weight=650, fill=COLORS["green"])
    svg.text(700, 555, "Training-only MTP weights are stripped during export.", size=23, weight=500, fill=COLORS["muted"])
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
    render_candidate_group(args.output_dir / "aios-ime-candidate-group.svg")
    render_prefix_kv(args.output_dir / "aios-ime-prefix-kv.svg")
    render_vllm_comparison(args.output_dir / "aios-ime-vllm-comparison.svg")
    render_performance(args.output_dir / "aios-ime-performance.svg")
    render_model(args.output_dir / "minimind-ime-model-architecture.svg")
    render_decoder_block(args.output_dir / "minimind-ime-decoder-block.svg")
    for name in (
        "aios-ime-runtime-architecture.svg",
        "aios-ime-candidate-group.svg",
        "aios-ime-prefix-kv.svg",
        "aios-ime-vllm-comparison.svg",
        "aios-ime-performance.svg",
        "minimind-ime-model-architecture.svg",
        "minimind-ime-decoder-block.svg",
    ):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
