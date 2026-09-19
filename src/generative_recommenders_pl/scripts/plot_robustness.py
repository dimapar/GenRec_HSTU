"""Build static robustness plots from ``run_robustness.py`` results.

The script intentionally avoids plotting dependencies such as matplotlib. It
reads the CSV produced by the robustness experiment and writes SVG figures plus
CSV/Markdown summaries that can be opened directly in a browser or inserted
into a report.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable


POSITION_ORDER = ["first", "p25", "p50", "p75", "last"]
POSITION_TITLES = {
    "first": "first",
    "p25": "25%",
    "p50": "50%",
    "p75": "75%",
    "last": "last",
}
PERTURBATION_TITLES = {
    "delete": "delete",
    "replace_random": "replace random",
}
SERIES_COLORS = {
    "delete": "#2563eb",
    "replace_random": "#d97706",
}
DEFAULT_COLOR = "#475569"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create robustness SVG plots and summary tables from CSV results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/robustness/ml1m_hstu_robustness.csv"),
        help="Path to the CSV produced by run_robustness.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/robustness/plots"),
        help="Directory where SVG figures and summary files will be written.",
    )
    parser.add_argument(
        "--history-bins",
        type=int,
        default=5,
        help="Number of equal-count history-length bins for the optional history plot.",
    )
    return parser.parse_args()


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def to_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["user_index"] = to_int(row.get("user_index"))
            row["user_id"] = to_int(row.get("user_id"))
            row["history_length"] = to_int(row.get("history_length"))
            row["position_index"] = to_int(row.get("position_index"))
            row["position_fraction"] = to_float(row.get("position_fraction"))
            row["top_k"] = to_int(row.get("top_k"))
            row["topk_overlap"] = to_float(row.get("topk_overlap"))
            row["jaccard"] = to_float(row.get("jaccard"))
            row["js_divergence"] = to_float(row.get("js_divergence"))
            row["target_rank_delta"] = to_float(row.get("target_rank_delta"))
            row["top1_changed"] = to_bool(row.get("top1_changed"))
            row["original_target_in_topk"] = to_bool(row.get("original_target_in_topk"))
            row["perturbed_target_in_topk"] = to_bool(row.get("perturbed_target_in_topk"))
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def metric_values(rows: Iterable[dict[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append(float(value))
    return values


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile for an empty list.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def compact_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def regular_ticks(low: float, high: float, step: float) -> list[float]:
    ticks: list[float] = []
    count = int(round((high - low) / step))
    for idx in range(count + 1):
        ticks.append(round(low + idx * step, 10))
    return ticks


def summarize(rows: list[dict[str, object]], group_keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    summary_rows: list[dict[str, object]] = []
    metric_names = [
        "topk_overlap",
        "jaccard",
        "js_divergence",
        "target_rank_delta",
        "top1_changed",
        "original_target_in_topk",
        "perturbed_target_in_topk",
    ]
    for key_values, group_rows in grouped.items():
        item = {key: value for key, value in zip(group_keys, key_values)}
        item["n"] = len(group_rows)
        for metric in metric_names:
            values = metric_values(group_rows, metric)
            item[f"{metric}_mean"] = mean(values) if values else None
            item[f"{metric}_std"] = sample_std(values)
        summary_rows.append(item)

    return sorted(summary_rows, key=summary_sort_key)


def summary_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    perturbation = str(row.get("perturbation_type", ""))
    perturb_key = 0 if perturbation == "delete" else 1 if perturbation == "replace_random" else 2
    position = str(row.get("position_label", ""))
    position_key = (
        POSITION_ORDER.index(position)
        if position in POSITION_ORDER
        else float(row.get("position_fraction") or 999)
    )
    history_bin_index = int(row.get("history_bin_index") or 0)
    return perturb_key, position_key, history_bin_index, perturbation


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: compact_csv_value(row.get(key)) for key in fieldnames})


def compact_csv_value(value: object) -> object:
    if isinstance(value, float):
        return compact_number(value, digits=6)
    return value if value is not None else ""


def position_labels(summary_rows: list[dict[str, object]]) -> list[str]:
    observed = {str(row["position_label"]) for row in summary_rows if row.get("position_label")}
    ordered = [label for label in POSITION_ORDER if label in observed]
    extras = sorted(observed - set(ordered))
    return ordered + extras


def perturbation_labels(rows: list[dict[str, object]]) -> list[str]:
    observed = {str(row["perturbation_type"]) for row in rows if row.get("perturbation_type")}
    ordered = [label for label in ["delete", "replace_random"] if label in observed]
    extras = sorted(observed - set(ordered))
    return ordered + extras


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_document(width: int, height: int, elements: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            *elements,
            "</svg>",
            "",
        ]
    )


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    commands = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(commands)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        return [low]
    raw_step = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    ticks = []
    value = start
    while value <= end + step / 2:
        if low - 1e-9 <= value <= high + 1e-9:
            ticks.append(0.0 if abs(value) < 1e-12 else value)
        value += step
    return ticks or [low, high]


def draw_axes(
    elements: list[str],
    *,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    x_labels: list[str],
    y_ticks: list[float],
    y_to_px: Callable[[float], float],
    x_to_px: Callable[[int], float],
    title: str,
    x_title: str,
    y_title: str,
    y_tick_formatter: Callable[[float], str] | None = None,
) -> None:
    bottom = top + plot_height
    right = left + plot_width
    elements.append(
        f'<text x="{left}" y="30" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="18" font-weight="600" fill="#111827">{escape(title)}</text>'
    )
    elements.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        f'fill="none" stroke="#cbd5e1" stroke-width="1"/>'
    )
    for tick in y_ticks:
        y = y_to_px(tick)
        elements.append(
            f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" '
            f'stroke="#cbd5e1" stroke-width="1"/>'
        )
        elements.append(
            f'<line x1="{left - 5}" x2="{left}" y1="{y:.2f}" y2="{y:.2f}" '
            f'stroke="#64748b" stroke-width="1"/>'
        )
        tick_label = y_tick_formatter(tick) if y_tick_formatter else compact_number(tick, digits=3)
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12" '
            f'fill="#334155">{escape(tick_label)}</text>'
        )
    for idx, label in enumerate(x_labels):
        x = x_to_px(idx)
        elements.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{bottom}" y2="{bottom + 5}" '
            f'stroke="#64748b" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{bottom + 23}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12" '
            f'fill="#475569">{escape(POSITION_TITLES.get(label, label))}</text>'
        )
    elements.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{bottom + 48}" text-anchor="middle" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">'
        f"{escape(x_title)}</text>"
    )
    elements.append(
        f'<text transform="translate(18 {top + plot_height / 2:.2f}) rotate(-90)" '
        f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" fill="#334155">{escape(y_title)}</text>'
    )


def draw_legend(
    elements: list[str], series: list[str], *, x: float, y: float, line_style: bool = True
) -> None:
    cursor = x
    for label in series:
        color = SERIES_COLORS.get(label, DEFAULT_COLOR)
        title = PERTURBATION_TITLES.get(label, label)
        if line_style:
            elements.append(
                f'<line x1="{cursor}" x2="{cursor + 24}" y1="{y}" y2="{y}" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            cursor += 31
        else:
            elements.append(
                f'<rect x="{cursor}" y="{y - 6}" width="12" height="12" '
                f'fill="{color}" opacity="0.3" stroke="{color}" stroke-width="1.5"/>'
            )
            cursor += 19
        elements.append(
            f'<text x="{cursor}" y="{y + 4}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="12" fill="#334155">{escape(title)}</text>'
        )
        cursor += 105


def plot_line_by_position(
    summary_rows: list[dict[str, object]],
    metric: str,
    *,
    title: str,
    y_title: str,
    output_path: Path,
    fixed_domain: tuple[float, float] | None = None,
    display_scale: float = 1.0,
    value_digits: int = 3,
    y_ticks: list[float] | None = None,
    y_tick_formatter: Callable[[float], str] | None = None,
) -> None:
    labels = position_labels(summary_rows)
    series = perturbation_labels(summary_rows)
    values_by_series = {
        label: {
            str(row["position_label"]): (
                float(row[f"{metric}_mean"]) * display_scale
                if isinstance(row.get(f"{metric}_mean"), (int, float))
                else None
            )
            for row in summary_rows
            if row.get("perturbation_type") == label
        }
        for label in series
    }
    plotted_values = [
        float(value)
        for values in values_by_series.values()
        for value in values.values()
        if isinstance(value, (int, float))
    ]
    if not plotted_values:
        return

    if fixed_domain:
        y_min, y_max = fixed_domain
    else:
        y_min = min(0.0, min(plotted_values))
        y_max = max(plotted_values)
        padding = max((y_max - y_min) * 0.12, 0.001)
        y_max += padding
    if y_max <= y_min:
        y_max = y_min + 1.0

    width, height = 1040, 580
    left, top = 88, 64
    plot_width, plot_height = 900, 400
    bottom = top + plot_height

    def x_to_px(idx: int) -> float:
        if len(labels) == 1:
            return left + plot_width / 2
        return left + idx * plot_width / (len(labels) - 1)

    def y_to_px(value: float) -> float:
        return bottom - (value - y_min) * plot_height / (y_max - y_min)

    elements: list[str] = []
    draw_axes(
        elements,
        left=left,
        top=top,
        plot_width=plot_width,
        plot_height=plot_height,
        x_labels=labels,
        y_ticks=y_ticks or nice_ticks(y_min, y_max),
        y_to_px=y_to_px,
        x_to_px=x_to_px,
        title=title,
        x_title="position in the user history",
        y_title=y_title,
        y_tick_formatter=y_tick_formatter,
    )
    draw_legend(elements, series, x=left + 625, y=33)

    for series_idx, label in enumerate(series):
        color = SERIES_COLORS.get(label, DEFAULT_COLOR)
        points: list[tuple[float, float]] = []
        for idx, position in enumerate(labels):
            value = values_by_series[label].get(position)
            if isinstance(value, (int, float)):
                points.append((x_to_px(idx), y_to_px(float(value))))
        elements.append(
            f'<path d="{line_path(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for idx, position in enumerate(labels):
            value = values_by_series[label].get(position)
            if not isinstance(value, (int, float)):
                continue
            x = x_to_px(idx)
            y = y_to_px(float(value))
            label_offset = 18 if series_idx == 0 else -12
            text_y = max(top + 14, min(bottom - 6, y + label_offset))
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#ffffff" '
                f'stroke="{color}" stroke-width="2"><title>'
                f"{escape(PERTURBATION_TITLES.get(label, label))}, "
                f"{escape(POSITION_TITLES.get(position, position))}: "
                f"{compact_number(float(value), digits=4)}</title></circle>"
            )
            elements.append(
                f'<text x="{x:.2f}" y="{text_y:.2f}" text-anchor="middle" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="12" '
                f'font-weight="600" fill="{color}" stroke="#ffffff" stroke-width="4" '
                f'paint-order="stroke">{float(value):.{value_digits}f}</text>'
            )

    output_path.write_text(svg_document(width, height, elements), encoding="utf-8")


def plot_retained_topk_heatmap_by_position(
    rows: list[dict[str, object]], *, output_path: Path
) -> None:
    labels = position_labels(summarize(rows, ["perturbation_type", "position_label"]))
    series = perturbation_labels(rows)
    top_k_values = {
        int(row["top_k"])
        for row in rows
        if isinstance(row.get("top_k"), int) and int(row["top_k"]) > 0
    }
    if not labels or not series or not top_k_values:
        return
    if len(top_k_values) != 1:
        raise ValueError("The retained-items heatmap requires a single top_k value.")
    top_k = next(iter(top_k_values))

    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        overlap = row.get("topk_overlap")
        row_top_k = row.get("top_k")
        if not isinstance(overlap, (int, float)) or row_top_k != top_k:
            continue
        perturbation = str(row.get("perturbation_type"))
        position = str(row.get("position_label"))
        retained = max(0, min(top_k, round(float(overlap) * top_k)))
        counts[(perturbation, position, retained)] += 1
        totals[(perturbation, position)] += 1

    width, height = 1120, 590
    grid_top = 94
    cell_width = 82
    cell_height = 34
    grid_width = len(labels) * cell_width
    grid_height = (top_k + 1) * cell_height
    panel_lefts = [100, 620] if len(series) > 1 else [360]
    elements = [
        '<text x="560" y="28" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        'font-size="20" font-weight="600" fill="#0f172a">'
        f'Distribution of retained top-{top_k} items by perturbation position</text>',
        '<text x="560" y="51" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        'font-size="12" fill="#475569">Each cell shows the percentage of experiments</text>',
    ]

    for panel_idx, perturbation in enumerate(series):
        if panel_idx >= len(panel_lefts):
            break
        left = panel_lefts[panel_idx]
        color = SERIES_COLORS.get(perturbation, DEFAULT_COLOR)
        elements.append(
            f'<text x="{left + grid_width / 2:.2f}" y="78" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" '
            f'fill="{color}">{escape(PERTURBATION_TITLES.get(perturbation, perturbation))}</text>'
        )

        for retained in range(top_k, -1, -1):
            row_idx = top_k - retained
            y = grid_top + row_idx * cell_height
            elements.append(
                f'<text x="{left - 12}" y="{y + cell_height / 2 + 4:.2f}" text-anchor="end" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#334155">'
                f'{retained}</text>'
            )
            for position_idx, position in enumerate(labels):
                x = left + position_idx * cell_width
                total = totals.get((perturbation, position), 0)
                count = counts.get((perturbation, position, retained), 0)
                share = count / total if total else 0.0
                fill_opacity = 0.0 if share == 0.0 else max(0.08, share)
                text_color = "#ffffff" if share >= 0.55 else "#0f172a"
                elements.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width}" height="{cell_height}" '
                    f'fill="{color}" fill-opacity="{fill_opacity:.4f}" stroke="#cbd5e1" '
                    f'stroke-width="1"><title>{escape(PERTURBATION_TITLES.get(perturbation, perturbation))}, '
                    f'{escape(POSITION_TITLES.get(position, position))}, {retained} retained: '
                    f'{share * 100:.1f}% ({count}/{total})</title></rect>'
                )
                if count:
                    percent_label = "<0.1%" if share * 100 < 0.05 else f"{share * 100:.1f}%"
                    elements.append(
                        f'<text x="{x + cell_width / 2:.2f}" y="{y + cell_height / 2 + 4:.2f}" '
                        f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
                        f'font-size="10.5" font-weight="600" fill="{text_color}">'
                        f'{escape(percent_label)}</text>'
                    )

        bottom = grid_top + grid_height
        for position_idx, position in enumerate(labels):
            x = left + (position_idx + 0.5) * cell_width
            elements.append(
                f'<text x="{x:.2f}" y="{bottom + 20}" text-anchor="middle" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#334155">'
                f'{escape(POSITION_TITLES.get(position, position))}</text>'
            )
        elements.append(
            f'<text x="{left + grid_width / 2:.2f}" y="{bottom + 45}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334155">'
            'position in the user history</text>'
        )

    elements.append(
        f'<text x="24" y="{grid_top + grid_height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {grid_top + grid_height / 2:.2f})" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334155">'
        f'retained recommendations (out of {top_k})</text>'
    )
    output_path.write_text(svg_document(width, height, elements), encoding="utf-8")


def plot_topk_overlap_boxplot_by_position(
    rows: list[dict[str, object]], *, output_path: Path
) -> None:
    labels = position_labels(summarize(rows, ["perturbation_type", "position_label"]))
    series = perturbation_labels(rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("topk_overlap")
        if isinstance(value, (int, float)):
            grouped[(str(row.get("perturbation_type")), str(row.get("position_label")))].append(
                float(value)
            )
    if not labels or not series or not grouped:
        return

    width, height = 1120, 620
    plot_top = 104
    plot_height = 410
    plot_width = 410
    panel_lefts = [90, 620] if len(series) > 1 else [355]
    bottom = plot_top + plot_height
    box_width = 46

    def y_to_px(value: float) -> float:
        return bottom - value * plot_height

    elements = [
        '<text x="560" y="28" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        'font-size="20" font-weight="600" fill="#0f172a">'
        'Top-10 overlap distribution by perturbation position</text>',
        '<rect x="135" y="44" width="28" height="16" fill="#64748b" fill-opacity="0.20" '
        'stroke="#475569" stroke-width="1.5"/>',
        '<line x1="135" x2="163" y1="52" y2="52" stroke="#475569" stroke-width="2.3"/>',
        '<text x="174" y="56" font-family="Segoe UI, Arial, sans-serif" font-size="11" '
        'fill="#334155">Q1-Q3; line = median</text>',
        '<line x1="442" x2="442" y1="43" y2="61" stroke="#475569" stroke-width="1.5"/>',
        '<line x1="434" x2="450" y1="43" y2="43" stroke="#475569" stroke-width="1.5"/>',
        '<line x1="434" x2="450" y1="61" y2="61" stroke="#475569" stroke-width="1.5"/>',
        '<text x="460" y="56" font-family="Segoe UI, Arial, sans-serif" font-size="11" '
        'fill="#334155">whiskers = 1.5 IQR</text>',
        '<polygon points="680,46 686,52 680,58 674,52" fill="#ffffff" '
        'stroke="#475569" stroke-width="1.7"/>',
        '<text x="696" y="56" font-family="Segoe UI, Arial, sans-serif" font-size="11" '
        'fill="#334155">mean</text>',
        '<circle cx="787" cy="52" r="3.5" fill="#ffffff" stroke="#475569" stroke-width="1.4"/>',
        '<text x="800" y="56" font-family="Segoe UI, Arial, sans-serif" font-size="11" '
        'fill="#334155">outlier level</text>',
    ]

    for panel_idx, perturbation in enumerate(series):
        if panel_idx >= len(panel_lefts):
            break
        left = panel_lefts[panel_idx]
        color = SERIES_COLORS.get(perturbation, DEFAULT_COLOR)
        x_step = plot_width / len(labels)

        elements.append(
            f'<text x="{left + plot_width / 2:.2f}" y="82" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="600" '
            f'fill="{color}">{escape(PERTURBATION_TITLES.get(perturbation, perturbation))}</text>'
        )

        for tick_idx in range(11):
            tick = tick_idx / 10
            y = y_to_px(tick)
            elements.append(
                f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" y2="{y:.2f}" '
                'stroke="#cbd5e1" stroke-width="1"/>'
            )
            elements.append(
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#334155">'
                f'{tick:.1f}</text>'
            )

        elements.append(
            f'<rect x="{left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" '
            'fill="none" stroke="#94a3b8" stroke-width="1"/>'
        )

        for position_idx, position in enumerate(labels):
            values = grouped.get((perturbation, position), [])
            x = left + (position_idx + 0.5) * x_step
            elements.append(
                f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#334155">'
                f'{escape(POSITION_TITLES.get(position, position))}</text>'
            )
            if not values:
                continue

            ordered = sorted(values)
            q1 = quantile(ordered, 0.25)
            median = quantile(ordered, 0.50)
            q3 = quantile(ordered, 0.75)
            average = mean(ordered)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            lower_whisker = next(value for value in ordered if value >= lower_limit)
            upper_whisker = next(value for value in reversed(ordered) if value <= upper_limit)

            y_q1 = y_to_px(q1)
            y_median = y_to_px(median)
            y_q3 = y_to_px(q3)
            y_low = y_to_px(lower_whisker)
            y_high = y_to_px(upper_whisker)
            box_y = min(y_q1, y_q3)
            box_height = max(2.0, abs(y_q1 - y_q3))

            elements.extend(
                [
                    f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y_high:.2f}" y2="{y_low:.2f}" '
                    f'stroke="{color}" stroke-width="1.6"/>',
                    f'<line x1="{x - 12:.2f}" x2="{x + 12:.2f}" y1="{y_high:.2f}" y2="{y_high:.2f}" '
                    f'stroke="{color}" stroke-width="1.6"/>',
                    f'<line x1="{x - 12:.2f}" x2="{x + 12:.2f}" y1="{y_low:.2f}" y2="{y_low:.2f}" '
                    f'stroke="{color}" stroke-width="1.6"/>',
                    f'<rect x="{x - box_width / 2:.2f}" y="{box_y:.2f}" width="{box_width}" '
                    f'height="{box_height:.2f}" fill="{color}" fill-opacity="0.22" '
                    f'stroke="{color}" stroke-width="1.8"><title>'
                    f'{escape(PERTURBATION_TITLES.get(perturbation, perturbation))}, '
                    f'{escape(POSITION_TITLES.get(position, position))}: '
                    f'Q1={q1:.2f}, median={median:.2f}, Q3={q3:.2f}, mean={average:.3f}, n={len(values)}'
                    '</title></rect>',
                    f'<line x1="{x - box_width / 2:.2f}" x2="{x + box_width / 2:.2f}" '
                    f'y1="{y_median:.2f}" y2="{y_median:.2f}" stroke="{color}" stroke-width="2.6"/>',
                ]
            )

            mean_y = y_to_px(average)
            diamond = [
                (x, mean_y - 5),
                (x + 5, mean_y),
                (x, mean_y + 5),
                (x - 5, mean_y),
            ]
            points = " ".join(f"{px:.2f},{py:.2f}" for px, py in diamond)
            elements.append(
                f'<polygon points="{points}" fill="#ffffff" stroke="{color}" stroke-width="1.8">'
                f'<title>mean={average:.3f}</title></polygon>'
            )

            outlier_counts: dict[float, int] = defaultdict(int)
            for value in ordered:
                if value < lower_whisker or value > upper_whisker:
                    outlier_counts[value] += 1
            for outlier_idx, (value, count) in enumerate(sorted(outlier_counts.items())):
                offset = -8 if outlier_idx % 2 == 0 else 8
                elements.append(
                    f'<circle cx="{x + offset:.2f}" cy="{y_to_px(value):.2f}" r="3" '
                    f'fill="#ffffff" stroke="{color}" stroke-width="1.3"><title>'
                    f'outlier {value:.1f}: {count} observations</title></circle>'
                )

            label_y = max(plot_top + 13, y_median - 8)
            elements.append(
                f'<text x="{x:.2f}" y="{label_y:.2f}" text-anchor="middle" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="600" '
                f'fill="{color}" stroke="#ffffff" stroke-width="4" paint-order="stroke">'
                f'{median:.2f}</text>'
            )

        elements.append(
            f'<text x="{left + plot_width / 2:.2f}" y="{bottom + 50}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334155">'
            'position in the user history</text>'
        )

    elements.append(
        f'<text x="24" y="{plot_top + plot_height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {plot_top + plot_height / 2:.2f})" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334155">'
        'top-10 overlap</text>'
    )
    output_path.write_text(svg_document(width, height, elements), encoding="utf-8")


def build_history_bin_lookup(
    rows: list[dict[str, object]], bin_count: int
) -> dict[tuple[object, object], tuple[int, str]]:
    by_user: dict[tuple[object, object], int] = {}
    for row in rows:
        length = row.get("history_length")
        if isinstance(length, int):
            by_user[(row.get("user_index"), row.get("user_id"))] = length

    sorted_users = sorted(by_user.items(), key=lambda item: (item[1], str(item[0])))
    if not sorted_users:
        return {}

    bin_count = max(1, min(bin_count, len(sorted_users)))
    lookup: dict[tuple[object, object], tuple[int, str]] = {}
    for idx in range(bin_count):
        start = round(idx * len(sorted_users) / bin_count)
        end = round((idx + 1) * len(sorted_users) / bin_count)
        chunk = sorted_users[start:end]
        if not chunk:
            continue
        lengths = [length for _, length in chunk]
        label = f"{min(lengths)}-{max(lengths)}"
        for user_key, _ in chunk:
            lookup[user_key] = (idx, label)
    return lookup


def assign_history_bins(
    rows: list[dict[str, object]], bin_lookup: dict[tuple[object, object], tuple[int, str]]
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        bin_info = bin_lookup.get((row.get("user_index"), row.get("user_id")))
        if bin_info is None:
            continue
        bin_index, label = bin_info
        copied = dict(row)
        copied["history_bin_index"] = bin_index
        copied["history_bin"] = label
        enriched.append(copied)
    return enriched


def plot_line_by_history_bin(
    summary_rows: list[dict[str, object]],
    *,
    metric: str,
    title: str,
    y_title: str,
    output_path: Path,
    fixed_domain: tuple[float, float] | None = None,
    value_digits: int = 3,
    y_ticks: list[float] | None = None,
    y_tick_formatter: Callable[[float], str] | None = None,
) -> None:
    labels = [
        str(row["history_bin"])
        for row in sorted(summary_rows, key=lambda item: int(item.get("history_bin_index") or 0))
        if row.get("perturbation_type") == summary_rows[0].get("perturbation_type")
    ]
    labels = list(dict.fromkeys(labels))
    series = perturbation_labels(summary_rows)
    values_by_series = {
        label: {
            str(row["history_bin"]): row.get(f"{metric}_mean")
            for row in summary_rows
            if row.get("perturbation_type") == label
        }
        for label in series
    }
    plotted_values = [
        float(value)
        for values in values_by_series.values()
        for value in values.values()
        if isinstance(value, (int, float))
    ]
    if not plotted_values:
        return

    if fixed_domain:
        y_min, y_max = fixed_domain
    else:
        y_min = min(0.0, min(plotted_values))
        y_max = max(plotted_values)
        padding = max((y_max - y_min) * 0.12, 0.001)
        y_max += padding
    if y_max <= y_min:
        y_max = y_min + 1.0

    width, height = 1040, 580
    left, top = 88, 64
    plot_width, plot_height = 900, 400
    bottom = top + plot_height

    def x_to_px(idx: int) -> float:
        if len(labels) == 1:
            return left + plot_width / 2
        return left + idx * plot_width / (len(labels) - 1)

    def y_to_px(value: float) -> float:
        return bottom - (value - y_min) * plot_height / (y_max - y_min)

    elements: list[str] = []
    draw_axes(
        elements,
        left=left,
        top=top,
        plot_width=plot_width,
        plot_height=plot_height,
        x_labels=labels,
        y_ticks=y_ticks or nice_ticks(y_min, y_max),
        y_to_px=y_to_px,
        x_to_px=x_to_px,
        title=title,
        x_title="history length bin",
        y_title=y_title,
        y_tick_formatter=y_tick_formatter,
    )
    draw_legend(elements, series, x=left + 625, y=33)

    for series_idx, label in enumerate(series):
        color = SERIES_COLORS.get(label, DEFAULT_COLOR)
        points: list[tuple[float, float]] = []
        for idx, history_bin in enumerate(labels):
            value = values_by_series[label].get(history_bin)
            if isinstance(value, (int, float)):
                points.append((x_to_px(idx), y_to_px(float(value))))
        elements.append(
            f'<path d="{line_path(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for idx, history_bin in enumerate(labels):
            value = values_by_series[label].get(history_bin)
            if not isinstance(value, (int, float)):
                continue
            x = x_to_px(idx)
            y = y_to_px(float(value))
            label_offset = 18 if series_idx == 0 else -12
            text_y = max(top + 14, min(bottom - 6, y + label_offset))
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#ffffff" '
                f'stroke="{color}" stroke-width="2"><title>'
                f"{escape(PERTURBATION_TITLES.get(label, label))}, "
                f"{escape(history_bin)}: {compact_number(float(value), digits=4)}"
                f"</title></circle>"
            )
            elements.append(
                f'<text x="{x:.2f}" y="{text_y:.2f}" text-anchor="middle" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="12" '
                f'font-weight="600" fill="{color}" stroke="#ffffff" stroke-width="4" '
                f'paint-order="stroke">{float(value):.{value_digits}f}</text>'
            )

    output_path.write_text(svg_document(width, height, elements), encoding="utf-8")


def write_markdown_report(
    path: Path,
    *,
    input_path: Path,
    rows: list[dict[str, object]],
    position_summary: list[dict[str, object]],
    history_summary: list[dict[str, object]],
    generated_files: list[Path],
) -> None:
    unique_users = {
        (row.get("user_index"), row.get("user_id"))
        for row in rows
        if row.get("user_index") is not None
    }
    top_k_values = sorted(
        {int(row["top_k"]) for row in rows if isinstance(row.get("top_k"), int)}
    )
    best_overlap = min(
        position_summary,
        key=lambda item: float(item["topk_overlap_mean"])
        if isinstance(item.get("topk_overlap_mean"), (int, float))
        else float("inf"),
    )
    strongest_js = max(
        position_summary,
        key=lambda item: float(item["js_divergence_mean"])
        if isinstance(item.get("js_divergence_mean"), (int, float))
        else float("-inf"),
    )

    lines = [
        "# Robustness experiment summary",
        "",
        f"- Source CSV: `{input_path}`",
        f"- Rows: {len(rows)}",
        f"- Users: {len(unique_users)}",
        f"- Top-k: {', '.join(map(str, top_k_values)) if top_k_values else 'unknown'}",
        "",
        "## Main observations",
        "",
        (
            "- Lowest mean top-k overlap: "
            f"`{PERTURBATION_TITLES.get(str(best_overlap.get('perturbation_type')), str(best_overlap.get('perturbation_type')))}` "
            f"at `{POSITION_TITLES.get(str(best_overlap.get('position_label')), str(best_overlap.get('position_label')))}` "
            f"= {compact_number(float(best_overlap['topk_overlap_mean']), digits=4)}."
        ),
        (
            "- Highest mean Jensen-Shannon divergence: "
            f"`{PERTURBATION_TITLES.get(str(strongest_js.get('perturbation_type')), str(strongest_js.get('perturbation_type')))}` "
            f"at `{POSITION_TITLES.get(str(strongest_js.get('position_label')), str(strongest_js.get('position_label')))}` "
            f"= {compact_number(float(strongest_js['js_divergence_mean']), digits=6)}."
        ),
        "",
        "## Generated files",
        "",
    ]
    lines.extend(f"- `{file}`" for file in generated_files)
    if history_summary:
        lines.extend(
            [
                "",
                "The history-length plot uses equal-count bins over users, then averages row-level robustness metrics inside each bin.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_cell_number(row: dict[str, object], key: str, *, digits: int = 3, scale: float = 1.0) -> str:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value) * scale:.{digits}f}"


def table_cell_percent(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value) * 100:.1f}%"


def summary_table_html(
    title: str,
    rows: list[dict[str, object]],
    *,
    include_position: bool,
) -> str:
    header = [
        "Perturbation",
        "Position" if include_position else "History length",
        "n",
        "Top-10 overlap",
        "Jaccard",
        "JS x1000",
        "Top-1 changed",
        "Target@10 after",
    ]
    body_rows = []
    for row in rows:
        label_key = "position_label" if include_position else "history_bin"
        label_value = str(row.get(label_key, ""))
        label_title = POSITION_TITLES.get(label_value, label_value) if include_position else label_value
        perturbation = str(row.get("perturbation_type", ""))
        body_rows.append(
            "<tr>"
            f"<td>{escape(PERTURBATION_TITLES.get(perturbation, perturbation))}</td>"
            f"<td>{escape(label_title)}</td>"
            f"<td class=\"num\">{escape(row.get('n', ''))}</td>"
            f"<td class=\"num strong\">{table_cell_number(row, 'topk_overlap_mean')}</td>"
            f"<td class=\"num\">{table_cell_number(row, 'jaccard_mean')}</td>"
            f"<td class=\"num\">{table_cell_number(row, 'js_divergence_mean', scale=1000.0)}</td>"
            f"<td class=\"num\">{table_cell_percent(row, 'top1_changed_mean')}</td>"
            f"<td class=\"num\">{table_cell_percent(row, 'perturbed_target_in_topk_mean')}</td>"
            "</tr>"
        )

    return "\n".join(
        [
            '<section class="table-block">',
            f"<h2>{escape(title)}</h2>",
            '<div class="table-wrap">',
            "<table>",
            "<thead><tr>",
            "".join(
                f'<th class="num">{escape(column)}</th>'
                if index >= 2
                else f"<th>{escape(column)}</th>"
                for index, column in enumerate(header)
            ),
            "</tr></thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
            "</div>",
            "</section>",
        ]
    )


def write_html_gallery(
    path: Path,
    svg_files: list[Path],
    summary_md: Path,
    position_summary: list[dict[str, object]],
    history_summary: list[dict[str, object]],
) -> None:
    cards = []
    figure_titles = {
        "topk_overlap_boxplot_by_position": (
            "Top-10 overlap boxplot by perturbation position"
        ),
        "topk_overlap_distribution_by_position": (
            "Distribution of retained Top-10 items by perturbation position"
        ),
    }
    for svg_file in svg_files:
        title = figure_titles.get(svg_file.stem, svg_file.stem.replace("_", " "))
        cards.append(
            "\n".join(
                [
                    '<section class="figure">',
                    f"<h2>{escape(title)}</h2>",
                    f'<img src="{escape(svg_file.name)}" alt="{escape(title)}">',
                    "</section>",
                ]
            )
        )
    tables = [
        summary_table_html(
            "Exact means by perturbation position",
            position_summary,
            include_position=True,
        )
    ]
    if history_summary:
        tables.append(
            summary_table_html(
                "Exact means by history length",
                history_summary,
                include_position=False,
            )
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robustness plots</title>
  <style>
    body {{
      margin: 32px;
      font-family: Segoe UI, Arial, sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 24px;
    }}
    .table-block {{
      margin: 0 0 28px;
      padding: 18px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: #334155;
      background: #f8fafc;
      font-weight: 600;
    }}
    th.num, td.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    td.strong {{
      font-weight: 600;
    }}
    .figure {{
      margin: 0 0 28px;
      padding: 18px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
    }}
    h2 {{
      font-size: 16px;
      font-weight: 600;
      margin: 0 0 10px;
      text-transform: capitalize;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Robustness plots</h1>
    {''.join(tables)}
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    position_summary = summarize(rows, ["perturbation_type", "position_label"])
    position_summary_path = args.output_dir / "robustness_summary_by_position.csv"
    write_summary_csv(position_summary_path, position_summary)
    overlap_ticks = regular_ticks(0.0, 1.0, 0.1)

    svg_files = [
        args.output_dir / "mean_topk_overlap_by_position.svg",
        args.output_dir / "mean_js_divergence_by_position.svg",
        args.output_dir / "topk_overlap_boxplot_by_position.svg",
        args.output_dir / "topk_overlap_distribution_by_position.svg",
    ]
    plot_line_by_position(
        position_summary,
        "topk_overlap",
        title="Mean top-10 overlap by perturbation position",
        y_title="mean top-10 overlap",
        output_path=svg_files[0],
        fixed_domain=(0.0, 1.0),
        value_digits=3,
        y_ticks=overlap_ticks,
        y_tick_formatter=lambda value: f"{value:.1f}",
    )
    plot_line_by_position(
        position_summary,
        "js_divergence",
        title="Mean Jensen-Shannon divergence by perturbation position",
        y_title="mean JS divergence x 1000",
        output_path=svg_files[1],
        fixed_domain=(0.0, 1.8),
        display_scale=1000.0,
        value_digits=3,
        y_ticks=regular_ticks(0.0, 1.8, 0.2),
        y_tick_formatter=lambda value: f"{value:.1f}",
    )
    plot_topk_overlap_boxplot_by_position(rows, output_path=svg_files[2])
    plot_retained_topk_heatmap_by_position(rows, output_path=svg_files[3])

    history_summary: list[dict[str, object]] = []
    bin_lookup = build_history_bin_lookup(rows, args.history_bins)
    if bin_lookup:
        rows_with_bins = assign_history_bins(rows, bin_lookup)
        history_summary = summarize(rows_with_bins, ["perturbation_type", "history_bin_index", "history_bin"])
        history_summary_path = args.output_dir / "robustness_summary_by_history_length.csv"
        write_summary_csv(history_summary_path, history_summary)
        history_plot_path = args.output_dir / "mean_topk_overlap_by_history_length.svg"
        plot_line_by_history_bin(
            history_summary,
            metric="topk_overlap",
            title="Mean top-10 overlap by history length",
            y_title="mean top-10 overlap",
            output_path=history_plot_path,
            fixed_domain=(0.0, 1.0),
            value_digits=3,
            y_ticks=overlap_ticks,
            y_tick_formatter=lambda value: f"{value:.1f}",
        )
        svg_files.append(history_plot_path)

    existing_svg_files = [path for path in svg_files if path.exists()]
    generated_files = [
        position_summary_path,
        args.output_dir / "robustness_summary_by_history_length.csv",
        *existing_svg_files,
    ]
    generated_files = [path for path in generated_files if path.exists()]
    summary_md = args.output_dir / "robustness_summary.md"
    write_markdown_report(
        summary_md,
        input_path=args.input,
        rows=rows,
        position_summary=position_summary,
        history_summary=history_summary,
        generated_files=generated_files,
    )
    html_gallery = args.output_dir / "robustness_plots.html"
    write_html_gallery(html_gallery, existing_svg_files, summary_md, position_summary, history_summary)
    generated_files.extend([summary_md, html_gallery])

    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"Wrote {len(generated_files)} files to {args.output_dir}:")
    for path in generated_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
