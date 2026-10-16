"""Average perturbation metrics within users and compare three interventions.

Uses only the fixed original candidate mask and five legacy positions. Time
metrics have equal weights for the six requested shifts, including clipped/no-op
shifts. Bootstrap intervals are recomputed from user means, not averaged bounds.
This script never writes a Markdown note or reruns model inference.
"""
import argparse
import hashlib
import html
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Circle, String
from reportlab.lib.colors import HexColor, white

from generative_recommenders_pl.scripts.analyze_controlled_robustness import UserBootstrap, summarize_users, METRICS
from generative_recommenders_pl.scripts.plot_controlled_robustness import register_fonts, METRIC_LABELS, METRIC_SCALES, LEGACY_ORDER

SHIFTS = {-604800, -86400, -3600, 3600, 86400, 604800}
COUNTS = {'delete': 1, 'replace_random': 5, 'timestamp_only': 6}
SERIES = [
    {'filter': {'perturbation_type': 'replace_random'}, 'label': 'Замена фильма', 'color': '#d97706'},
    {'filter': {'perturbation_type': 'delete'}, 'label': 'Удаление фильма', 'color': '#2563eb'},
    {'filter': {'perturbation_type': 'timestamp_only'}, 'label': 'Изменение только времени', 'color': '#059669'},
]


def select_and_validate(frame):
    selected = frame.loc[(frame.mask_mode == 'fixed_original') & frame.is_legacy_position
                         & frame.perturbation_type.isin(COUNTS)].copy()
    keys = ['perturbation_type', 'position_label', 'user_index']
    counts = selected.groupby(keys).size()
    for (kind, position, user), count in counts.items():
        if count != COUNTS[kind]:
            raise ValueError(f'Incomplete repeats for {kind}/{position}/user {user}: {count}')
    users = set(selected.user_index)
    for kind in COUNTS:
        for position in LEGACY_ORDER:
            group = selected.loc[(selected.perturbation_type == kind) & (selected.position_label == position)]
            if set(group.user_index) != users:
                raise ValueError(f'Different user cohort for {kind}/{position}')
    temporal = selected.loc[selected.perturbation_type == 'timestamp_only']
    for key, group in temporal.groupby(['position_label', 'user_index']):
        if set(group.requested_timestamp_shift_seconds) != SHIFTS:
            raise ValueError(f'Incomplete time shifts for {key}')
    return selected



def comparison_chart(summary, metric):
    """Stage1-style chart: hollow markers, readable values and bilingual labels."""
    register_fonts()
    width, height = 700, 300
    left, right, bottom, top = 72, 650, 73, 222
    scale = METRIC_SCALES[metric]
    title = {'topk_overlap': 'Средний Top-10 overlap по позиции изменения',
             'js_divergence': 'Средняя JS divergence по позиции изменения',
             'top1_changed': 'Доля смен Top-1 по позиции изменения'}[metric]
    drawing = Drawing(width, height)
    drawing.add(String(width / 2, height - 19, title, textAnchor='middle', fontName='StudyBold',
                       fontSize=15, fillColor=HexColor('#17212b')))
    points = []
    for item in SERIES:
        selected = summary.loc[summary.perturbation_type == item['filter']['perturbation_type']].set_index('position_label')
        points.append([float(selected.loc[position, metric + '_mean']) * scale for position in LEGACY_ORDER])
    if metric == 'topk_overlap':
        ymax, ticks, tick_digits, value_digits = 1.25, [i / 5 for i in range(6)], 1, 3
    elif metric == 'top1_changed':
        ymax, ticks, tick_digits, value_digits = 112, list(range(0, 101, 20)), 0, 2
    else:
        maximum = max(max(values) for values in points)
        magnitude = 10 ** math.floor(math.log10(max(maximum, 1e-9)))
        step = magnitude / 5
        ymax = math.ceil(maximum * 1.18 / step) * step
        ticks = [i * ymax / 5 for i in range(6)]
        tick_digits, value_digits = 2 if ymax < 1 else 1, 4
    def xp(index):
        return left + (right - left) * index / 4
    def yp(value):
        return bottom + (top - bottom) * value / ymax
    grid_color = HexColor('#bfd1eb')
    axis_color = HexColor('#58697a')
    for tick in ticks:
        y = yp(tick)
        drawing.add(Line(left, y, right, y, strokeColor=grid_color, strokeWidth=.6))
        drawing.add(String(left - 10, y - 3.5, f'{tick:.{tick_digits}f}', textAnchor='end',
                           fontName='StudyRegular', fontSize=10, fillColor=axis_color))
    drawing.add(Line(left, bottom, left, top, strokeColor=grid_color, strokeWidth=.6))
    drawing.add(Line(right, bottom, right, top, strokeColor=grid_color, strokeWidth=.6))
    for index, label in enumerate(['first', '25%', '50%', '75%', 'last']):
        drawing.add(String(xp(index), bottom - 26, label, textAnchor='middle',
                           fontName='StudyRegular', fontSize=10, fillColor=axis_color))
    drawing.add(String((left + right) / 2, 14, 'Position in history', textAnchor='middle',
                       fontName='StudyRegular', fontSize=11, fillColor=axis_color))
    drawing.add(String(left, top + 8, METRIC_LABELS[metric], fontName='StudyRegular',
                       fontSize=10, fillColor=axis_color))
    for series_index, (item, values) in enumerate(zip(SERIES, points)):
        color = HexColor(item['color'])
        legend_x = [72, 275, 465][series_index]
        legend_y = 253
        drawing.add(Line(legend_x, legend_y, legend_x + 23, legend_y, strokeWidth=2.1, strokeColor=color))
        drawing.add(Circle(legend_x + 12, legend_y, 3.5, fillColor=white, strokeColor=color, strokeWidth=1.9))
        drawing.add(String(legend_x + 30, legend_y - 3.5, item['label'], fontName='StudyRegular', fontSize=10))
        for index, value in enumerate(values):
            if index:
                drawing.add(Line(xp(index - 1), yp(values[index - 1]), xp(index), yp(value), strokeWidth=2.1, strokeColor=color))
            drawing.add(Circle(xp(index), yp(value), 4, fillColor=white, strokeColor=color, strokeWidth=2.1))
    # Spread labels vertically when several curves are close. Edge labels are
    # placed in reserved side margins, so values stay clear of the curves.
    for index in range(5):
        labels = sorted((yp(values[index]), series_index, values[index]) for series_index, values in enumerate(points))
        placed, previous = [], -math.inf
        for y, series_index, value in labels:
            label_y = max(y + 9, previous + 13)
            placed.append([label_y, series_index, value])
            previous = label_y
        shift = max(0, placed[-1][0] - (top - 2))
        if metric == 'topk_overlap':
            offsets = (9, -15, 24) if index < 4 else (9, 9, 9)
            placed = [[yp(values[index]) + offsets[s], s, values[index]] for s, values in enumerate(points)]
            shift = 0
        elif metric == 'top1_changed':
            offsets = (9, 9, -12)
            placed = [[yp(values[index]) + offsets[s], s, values[index]] for s, values in enumerate(points)]
            shift = 0
        elif index == 4:
            offsets = (9, 9, -12)
            placed = [[yp(values[index]) + offsets[s], s, values[index]] for s, values in enumerate(points)]
            shift = 0
        for label_y, series_index, value in placed:
            anchor = 'start' if index in (0, 4) else 'middle'
            label_x = xp(index) + (7 if index in (0, 4) else 0)
            drawing.add(String(label_x, label_y - shift, f'{value:.{value_digits}f}',
                               textAnchor=anchor, fontName='StudyRegular', fontSize=9.5,
                               fillColor=HexColor(SERIES[series_index]['color'])))
    return drawing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('outputs/robustness/stage2'))
    parser.add_argument('--bootstrap-repeats', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=42)
    parser.add_argument('--pdftoppm', default='pdftoppm')
    parser.add_argument('--plots-only', action='store_true', help='Render existing summary without rewriting data, metadata, gallery or notes')
    args = parser.parse_args()
    executable = shutil.which(args.pdftoppm)
    if not executable:
        raise FileNotFoundError('pdftoppm missing; pass --pdftoppm PATH')
    output = args.input_dir / 'plots/timestamps'
    if args.plots_only:
        summary = pd.read_csv(output / 'averaged_impact_summary.csv')
    else:
        raw_path = args.input_dir / 'controlled_results.csv'
        frame = select_and_validate(pd.read_csv(raw_path))
        group_columns = ['perturbation_type', 'position_label']
        summary = summarize_users(frame, group_columns, UserBootstrap(args.bootstrap_repeats, args.bootstrap_seed))
        # Independent check: compare with the equal-weight means of the saved per-shift summaries.
        previous = pd.read_csv(args.input_dir / 'summary_legacy_positions.csv')
        previous = previous.loc[(previous.mask_mode == 'fixed_original') & previous.perturbation_type.isin(COUNTS)]
        expected = previous.groupby(group_columns)[[m + '_mean' for m in METRICS]].mean().sort_index()
        actual = summary.set_index(group_columns)[expected.columns].sort_index()
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-14)
        output = args.input_dir / 'plots/timestamps'
        output.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output / 'averaged_impact_summary.csv', index=False)
        user_means = frame.groupby(group_columns + ['user_index'], sort=True)[list(METRICS)].mean().reset_index()
        user_means.to_csv(output / 'averaged_impact_by_user.csv', index=False)
    cards = []
    for metric in METRIC_LABELS:
        drawing = comparison_chart(summary, metric)
        name = f'averaged_impact_{metric}'
        svg, pdf = output / f'{name}.svg', output / f'{name}.pdf'
        renderSVG.drawToFile(drawing, str(svg))
        svg.write_text(svg.read_text(encoding='utf-8').replace('font-family: StudyRegular', 'font-family: Arial').replace('font-family: StudyBold', 'font-family: Arial'), encoding='utf-8')
        renderPDF.drawToFile(drawing, str(pdf))
        subprocess.run([executable, '-r', '220', '-singlefile', '-png', str(pdf), str(output / name)], check=True, capture_output=True)
        cards.append(f'<section><h2>{html.escape(METRIC_LABELS[metric])}</h2><img src="{name}.svg" alt="{name}"></section>')
    if args.plots_only:
        print('Updated three SVG/PDF/PNG charts; summary data, metadata and Markdown unchanged.')
        return
    gallery = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение трёх воздействий</title>'
               '<style>body{font-family:Arial,sans-serif;background:#f8fafc;margin:30px}main{max-width:1000px;margin:auto}section{background:white;padding:20px;margin:20px 0;border:1px solid #d8e0ea}img{width:100%;height:auto}</style>'
               '<main><h1>Замена фильма, удаление и изменение времени</h1><p>Фиксированная маска; временные метрики усреднены по шести заданным сдвигам внутри пользователя. '
               'Замены усреднены по пяти повторениям. Ограниченные и нулевые сдвиги включены. '
               'На графиках показаны средние значения метрик.</p>' + ''.join(cards) + '</main></html>')
    (output / 'averaged_impact_plots.html').write_text(gallery, encoding='utf-8')
    metadata = {'mask_mode': 'fixed_original', 'positions': LEGACY_ORDER,
                'requested_time_shifts_seconds': sorted(SHIFTS), 'repetitions_by_kind': COUNTS,
                'aggregation': 'equal-weight repeat means within each user/position/kind, followed by equal-weight user mean',
                'clipped_and_zero_actual_time_shifts_included': True,
                'bootstrap_repeats': args.bootstrap_repeats, 'bootstrap_seed': args.bootstrap_seed,
                'confidence_intervals': 'pointwise 95% user-bootstrap percentile intervals after within-user averaging',
                'n_users': int(frame.user_index.nunique()), 'rows_used': len(frame),
                'source_csv_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'saved_summary_mean_check': 'passed', 'display_confidence_intervals': False}
    (output / 'averaged_impact_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Validated {len(frame)} comparisons for {metadata["n_users"]} users; wrote three averaged comparison charts to {output}.')


if __name__ == '__main__':
    main()
