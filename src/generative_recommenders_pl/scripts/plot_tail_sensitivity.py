"""Detailed last-ten sensitivity with user-bootstrap intervals only for this section."""
import argparse
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.graphics import renderPDF, renderSVG
from generative_recommenders_pl.scripts.analyze_controlled_robustness import UserBootstrap, summarize_users, METRICS
from generative_recommenders_pl.scripts.plot_controlled_robustness import chart, METRIC_LABELS, metric_title
from generative_recommenders_pl.scripts.plot_timestamp_robustness import SERIES, SHIFTS, COUNTS


def build_tail_figures(summary):
    figures = {}
    for metric in METRIC_LABELS:
        domain = (0, 1) if metric == 'topk_overlap' else (0, 100) if metric == 'top1_changed' else None
        figures[f'tail_impact_{metric}'] = chart(
            summary, metric=metric, x_column='distance_from_end', x_values=list(range(9, -1, -1)),
            x_labels=[str(i) for i in range(10, 0, -1)], series=SERIES,
            title=metric_title(metric, 'последние десять фильмов'), x_title='Position from end (1 = last)',
            domain=domain, show_intervals=False)
    return figures



def metric_table_rows(summary, metric):
    """Mean and numerical CI bounds for each position in left-to-right plot order."""
    scale = 1000 if metric == 'js_divergence' else 100 if metric == 'top1_changed' else 1
    digits = 6 if metric == 'js_divergence' else 2 if metric == 'top1_changed' else 4
    rows = [['Позиция от конца', 'Замена фильма', 'Удаление фильма', 'Только время']]
    for distance in range(9, -1, -1):
        cells = [str(distance + 1)]
        for kind in ['replace_random','delete','timestamp_only']:
            row = summary.loc[(summary.perturbation_type==kind) & (summary.distance_from_end==distance)].iloc[0]
            values = [float(row[f'{metric}_{part}'])*scale for part in ['mean','ci_low','ci_high']]
            cells.append(f'{values[0]:.{digits}f} [{values[1]:.{digits}f}; {values[2]:.{digits}f}]')
        rows.append(cells)
    return rows



def render_tail_section(directory, summary, executable, render_graphs=True):
    output = directory / 'plots/tail_sensitivity'
    output.mkdir(parents=True, exist_ok=True)
    cards = []
    for name, drawing in build_tail_figures(summary).items():
        if render_graphs:
            svg, pdf = output / f'{name}.svg', output / f'{name}.pdf'
            renderSVG.drawToFile(drawing, str(svg))
            text = svg.read_text(encoding='utf-8').replace('font-family: StudyRegular', 'font-family: Arial').replace('font-family: StudyBold', 'font-family: Arial; font-weight: bold')
            svg.write_text(text, encoding='utf-8')
            renderPDF.drawToFile(drawing, str(pdf))
            subprocess.run([executable, '-r', '180', '-singlefile', '-png', str(pdf), str(output / name)], check=True, capture_output=True)
        metric = name.removeprefix('tail_impact_')
        rows = metric_table_rows(summary, metric)
        table = '<table><thead><tr>' + ''.join('<th>'+html.escape(c)+'</th>' for c in rows[0]) + '</tr></thead><tbody>'
        def cell_markup(cell):
            if ' [' not in cell:
                return html.escape(cell)
            mean, interval = cell.split(' ',1)
            return '<strong>'+html.escape(mean)+'</strong><br><span class="ci">'+html.escape(interval)+'</span>'
        table += ''.join('<tr>'+''.join('<td>'+cell_markup(c)+'</td>' for c in row)+'</tr>' for row in rows[1:]) + '</tbody></table>'
        table_metric = {'topk_overlap':'Top-10 overlap', 'js_divergence':'JS × 1000', 'top1_changed':'Top-1 changed, %'}[metric]
        table_number = list(METRIC_LABELS).index(metric) + 1
        cards.append(f'<section><img src="{name}.svg" alt="{name}"><h2>Таблица {table_number}. {table_metric}: средние и 95%-е доверительные интервалы</h2><p>Жирным - среднее по 500 пользователям; ниже - [нижняя граница; верхняя граница] интервала.</p>{table}</section>')
    gallery = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Последние десять позиций</title>'
               '<style>body{font-family:Arial,sans-serif;background:#f8fafc;margin:30px}main{max-width:1200px;margin:auto}section{background:white;padding:20px;margin:20px 0;border:1px solid #d8e0ea}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #c6d3e1;padding:7px;text-align:center}th{background:#e2e8f0}td strong{font-size:15px}.ci{color:#58697a;font-size:13px}</style>'
               '<main><h1>Чувствительность последних десяти позиций</h1><p>1 = последний фильм, 10 = десятый от конца. Фиксированная маска. '
               'Числа показывают средние; границы 95%-х доверительных интервалов приведены в таблице под каждым графиком. Bootstrap по 500 пользователям, 5000 повторов. '
               'Время усреднено по шести заданным сдвигам, включая ограниченные и нулевые.</p>' + ''.join(cards) + '</main></html>')
    (output / 'tail_sensitivity_plots.html').write_text(gallery, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('outputs/robustness/stage2'))
    parser.add_argument('--bootstrap-repeats', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=42)
    parser.add_argument('--pdftoppm', default='pdftoppm')
    parser.add_argument('--render-only', action='store_true', help='Use saved statistics; update only graph files and HTML')
    parser.add_argument('--tables-only', action='store_true', help='Update HTML table styling without touching graph files or statistics')
    args = parser.parse_args()
    executable = shutil.which(args.pdftoppm)
    if not executable:
        raise FileNotFoundError('pdftoppm not found')
    directory = args.input_dir
    if args.render_only or args.tables_only:
        render_tail_section(directory, pd.read_csv(directory/'plots/tail_sensitivity/tail_impact_summary.csv'), executable, render_graphs=not args.tables_only)
        print('Rendered three mean-only graphs and confidence-interval tables from unchanged saved statistics.')
        return
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file()}
    raw = directory / 'controlled_results.csv'
    frame = pd.read_csv(raw)
    frame = frame.loc[(frame.mask_mode == 'fixed_original') & frame.distance_from_end.between(0, 9)
                      & frame.perturbation_type.isin(COUNTS)].copy()
    users = set(frame.user_index)
    groups = ['perturbation_type', 'distance_from_end']
    for kind in COUNTS:
        for distance in range(10):
            group = frame.loc[(frame.perturbation_type == kind) & (frame.distance_from_end == distance)]
            if set(group.user_index) != users or not (group.groupby('user_index').size() == COUNTS[kind]).all():
                raise ValueError(f'Incomplete user cohort or repeats: {kind}, distance {distance}')
            if kind == 'timestamp_only':
                if not all(set(g.requested_timestamp_shift_seconds) == SHIFTS for _, g in group.groupby('user_index')):
                    raise ValueError('Incomplete six-shift group')
    bootstrap = UserBootstrap(args.bootstrap_repeats, args.bootstrap_seed)
    summary = summarize_users(frame, groups, bootstrap)
    old = pd.read_csv(directory / 'summary_tail_positions.csv')
    old = old.loc[(old.mask_mode == 'fixed_original') & old.perturbation_type.isin(COUNTS)]
    expected = old.groupby(groups)[[m + '_mean' for m in METRICS]].mean().sort_index()
    actual = summary.set_index(groups)[expected.columns].sort_index()
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-14)
    means = frame.groupby(groups + ['user_index'], sort=True)[list(METRICS)].mean().reset_index()
    last = means.loc[means.distance_from_end == 0]
    penultimate = means.loc[means.distance_from_end == 1]
    paired = last.merge(penultimate, on=['perturbation_type','user_index'], validate='one_to_one', suffixes=('_last','_penultimate'))
    for metric in METRICS:
        paired[metric] = paired[metric + '_last'] - paired[metric + '_penultimate']
    contrasts = summarize_users(paired, ['perturbation_type'], bootstrap)
    output = directory / 'plots/tail_sensitivity'
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'tail_impact_summary.csv', index=False)
    means.to_csv(output / 'tail_impact_by_user.csv', index=False)
    contrasts.to_csv(output / 'last_minus_penultimate.csv', index=False)
    render_tail_section(directory, summary, executable)
    metadata = {'mask_mode': 'fixed_original', 'distance_from_end': list(range(10)), 'axis_position_equals_distance_plus_one': True,
                'n_users': len(users), 'rows_used': len(frame), 'repetitions_by_kind': COUNTS,
                'requested_time_shifts_seconds': sorted(SHIFTS), 'clipped_and_noop_time_shifts_included': True,
                'aggregation': 'within-user equal-weight repeat means, then equal-weight user means',
                'confidence_intervals': 'pointwise 95% percentile bootstrap over users, not prediction intervals',
                'bootstrap_repeats': args.bootstrap_repeats, 'bootstrap_seed': args.bootstrap_seed,
                'last_minus_penultimate_contrasts': 'paired within users after repeat averaging',
                'source_csv_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'mean_regression_check': 'passed', 'display_intervals_on_graphs': False, 'confidence_intervals_in_tables': True}
    (output / 'tail_impact_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    for p, digest in protected.items():
        if not p.is_relative_to(output) and hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Existing file changed: {p}')
    print(f'Validated {len(frame)} comparisons from {len(users)} users. Created 3 tail figures with CI tables; previous figures and notes unchanged.')


if __name__ == '__main__':
    main()
