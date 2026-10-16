"""Stage2 vector figures with labelled values and consistent bilingual styling."""

import argparse
import html
import math
import subprocess
from pathlib import Path

import pandas as pd
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, String, Circle, Rect
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

METRIC_LABELS = {"topk_overlap": "Top-10 overlap", "js_divergence": "JS divergence × 1000", "top1_changed": "Top-1 changed, %"}
METRIC_SCALES = {"topk_overlap": 1, "js_divergence": 1000, "top1_changed": 100}
LEGACY_ORDER = ["first", "p25", "p50", "p75", "last"]
SHIFT_LABELS = {-604800: "-7 дней", -86400: "-1 день", -3600: "-1 час", 3600: "+1 час", 86400: "+1 день", 604800: "+7 дней"}


def register_fonts():
    if "StudyRegular" in pdfmetrics.getRegisteredFontNames():
        return
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("StudyRegular", str(regular)))
            pdfmetrics.registerFont(TTFont("StudyBold", str(bold)))
            pdfmetrics.registerFontFamily("StudyRegular", normal="StudyRegular", bold="StudyBold", italic="StudyRegular", boldItalic="StudyBold")
            return
    raise FileNotFoundError("Install Arial or DejaVu Sans fonts with Cyrillic support")


def metric_title(metric, context):
    prefix = {'topk_overlap': 'Средний Top-10 overlap', 'js_divergence': 'Средняя JS divergence',
              'top1_changed': 'Доля смен Top-1', 'time_bucket_changed': 'Изменение временных интервалов'}[metric]
    if context.startswith('динамическая минус фиксированная'):
        prefix = {'topk_overlap': 'Разность Top-10 overlap', 'js_divergence': 'Разность JS divergence', 'top1_changed': 'Разность долей смен Top-1'}[metric]
    return f'{prefix}: {context}'


def chart(frame, *, metric, x_column, x_values, x_labels, series, title, x_title,
          y_title=None, scale=None, domain=None, height=250, show_intervals=False):
    """Readable point values, English axes, Russian title and legend.

    Candidate label positions are scored against other labels, markers and line
    segments. Dashed/solid pairs prefer opposite sides of a point. Values are
    rounded for display only; all data and scales are kept unchanged.
    """
    register_fonts()
    scale = METRIC_SCALES.get(metric, 1) if scale is None else scale
    y_title = METRIC_LABELS.get(metric, metric) if y_title is None else y_title
    y_title = {'Разность долей Top-1 changed, п.п.': 'Top-1 changed difference, pp',
               'Разность Top-10 overlap': 'Top-10 overlap difference',
               'Разность JS divergence × 1000': 'JS divergence difference × 1000',
               'Изменён хотя бы один временной интервал, %': 'Changed time buckets, %'}.get(y_title, y_title)
    tail = len(x_values) > 5
    width = 980 if tail else 700
    height = max(340 if tail else 300, round(height * (1.36 if tail else 1.2)))
    drawing = Drawing(width, height)
    left, right, bottom = 78, width - 70, 73
    legend_columns = 3 if len(series) == 3 else 2
    legend_rows = math.ceil(len(series) / legend_columns)
    top = height - 55 - legend_rows * 19
    x_title = 'Position from end (1 = last)' if tail else 'Position in history'
    if not tail:
        x_labels = [{'первая': 'first', 'последняя': 'last'}.get(v, v) for v in x_labels]
    prepared, limits = [], []
    for item in series:
        selected = frame
        for column, value in item['filter'].items():
            selected = selected.loc[selected[column] == value]
        points = {}
        for x in x_values:
            rows = selected.loc[selected[x_column] == x]
            if len(rows) > 1:
                raise ValueError('A figure series has duplicate x coordinates')
            if len(rows):
                row = rows.iloc[0]
                mean = float(row[f'{metric}_mean']) * scale
                low = float(row.get(f'{metric}_ci_low', mean / scale)) * scale
                high = float(row.get(f'{metric}_ci_high', mean / scale)) * scale
                points[x] = (mean, low, high)
                limits.extend((mean, low, high) if show_intervals else (mean,))
        prepared.append((item, points))
    finite = [v for v in limits if math.isfinite(v)]
    if not finite:
        raise ValueError(f'No data for {title}')
    if domain:
        lo, hi = domain
    else:
        lo, hi = min(0, min(finite)), max(finite)
        if lo == hi:
            hi = lo + 1
    span = hi - lo
    # Extra space accommodates labels on almost coinciding curves near 0 or 1.
    y_min = lo - (span * .08 if lo < 0 else 0)
    y_max = hi + span * (.22 if len(series) > 2 else .15)
    def xp(index):
        return left + (right - left) * index / max(1, len(x_values) - 1)
    def yp(value):
        return bottom + (top - bottom) * (value - y_min) / (y_max - y_min)
    ink, grid = HexColor('#58697a'), HexColor('#bfd1eb')
    font_size = 15
    while pdfmetrics.stringWidth(title, 'StudyBold', font_size) > width - 30:
        font_size -= .5
    drawing.add(String(width / 2, height - 20, title, textAnchor='middle',
                       fontName='StudyBold', fontSize=font_size, fillColor=HexColor('#17212b')))
    for index, (item, points) in enumerate(prepared):
        color = HexColor(item['color'])
        dash = [4, 3] if item.get('dashed') else None
        lx = left + (index % legend_columns) * ((width - left - 20) / legend_columns)
        ly = height - 43 - (index // legend_columns) * 19
        drawing.add(Line(lx, ly, lx + 23, ly, strokeWidth=2, strokeColor=color, strokeDashArray=dash))
        drawing.add(Circle(lx + 12, ly, 3.1, fillColor=white, strokeColor=color, strokeWidth=1.8))
        drawing.add(String(lx + 30, ly - 3.5, item['label'], fontName='StudyRegular', fontSize=10))
    drawing.add(String(left, top + 8, y_title, fontName='StudyRegular', fontSize=10, fillColor=ink))
    decimals = 4 if span < .01 else 3 if span < .1 else 2 if hi <= 2.5 else 1
    for i in range(6):
        value = lo + span * i / 5
        y = yp(value)
        drawing.add(Line(left, y, right, y, strokeColor=grid, strokeWidth=.6))
        drawing.add(String(left - 10, y - 3.5, f'{value:.{decimals}f}', textAnchor='end',
                           fontName='StudyRegular', fontSize=10, fillColor=ink))
    if y_min < 0 < y_max:
        drawing.add(Line(left, yp(0), right, yp(0), strokeWidth=.8, strokeColor=ink, strokeDashArray=[3, 3]))
    drawing.add(Line(left, bottom, left, top, strokeColor=grid, strokeWidth=.6))
    drawing.add(Line(right, bottom, right, top, strokeColor=grid, strokeWidth=.6))
    for i, label in enumerate(x_labels):
        drawing.add(String(xp(i), bottom - 26, label, textAnchor='middle',
                           fontName='StudyRegular', fontSize=10, fillColor=ink))
    drawing.add(String((left + right) / 2, 14, x_title, textAnchor='middle', fontName='StudyRegular', fontSize=11, fillColor=ink))
    segments, markers = [], []
    for item, points in prepared:
        color = HexColor(item['color'])
        dash = [4, 3] if item.get('dashed') else None
        previous = None
        for i, x_value in enumerate(x_values):
            if x_value not in points:
                previous = None
                continue
            value, low, high = points[x_value]
            x, y = xp(i), yp(value)
            if previous:
                drawing.add(Line(*previous, x, y, strokeColor=color, strokeWidth=2, strokeDashArray=dash))
                segments.append((*previous, x, y))
            if show_intervals and math.isfinite(low) and math.isfinite(high):
                drawing.add(Line(x, yp(low), x, yp(high), strokeColor=color, strokeWidth=.9))
                segments.append((x, yp(low), x, yp(high)))
                for bound in (low, high):
                    drawing.add(Line(x - 3, yp(bound), x + 3, yp(bound), strokeColor=color, strokeWidth=.9))
                    segments.append((x - 3, yp(bound), x + 3, yp(bound)))
            drawing.add(Circle(x, y, 3.4, fillColor=white, strokeColor=color, strokeWidth=2))
            markers.append((x, y))
            previous = (x, y)
    def overlap(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    def crossing(box, segment):
        x1, y1, x2, y2 = segment
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        lower, upper = max(box[0], x1), min(box[2], x2)
        if upper < lower:
            return False
        if x1 == x2:
            return min(y1, y2) <= box[3] and max(y1, y2) >= box[1]
        ya = y1 + (y2 - y1) * (lower - x1) / (x2 - x1)
        yb = y1 + (y2 - y1) * (upper - x1) / (x2 - x1)
        return min(ya, yb) <= box[3] and max(ya, yb) >= box[1]
    occupied = []
    for i in range(6):
        value = lo + span * i / 5
        label = f'{value:.{decimals}f}'
        tick_width = pdfmetrics.stringWidth(label, 'StudyRegular', 10)
        occupied.append((left - 12 - tick_width, yp(value) - 6, left - 7, yp(value) + 8))
    number_font = 9.2
    value_digits = 4 if metric == 'js_divergence' or (metric == 'topk_overlap' and 'difference' in y_title) else 3 if metric == 'topk_overlap' else 2 if metric == 'top1_changed' else 1
    for i, x_value in enumerate(x_values):
        for item, points in prepared:
            if x_value not in points:
                continue
            value = points[x_value][0]
            x, y = xp(i), yp(value)
            label = f'{value:.{value_digits}f}'
            text_width = pdfmetrics.stringWidth(label, 'StudyRegular', number_font)
            candidates = []
            preferred = -1 if item.get('dashed') else 1
            for side in (preferred, -preferred):
                tx = x + side * 7
                anchor = 'start' if side == 1 else 'end'
                bx = tx if side == 1 else tx - text_width
                for dy in (10, -17, 24, -31, 38, -45, 52, -59, 66, -73, 80, -87):
                    ty = y + dy
                    box = (bx - 1, ty - 2, bx + text_width + 1, ty + number_font + 1)
                    if box[0] < 8 or box[2] > width - 8 or box[1] < bottom - 18 or box[3] > top + 1:
                        continue
                    score = sum(overlap(box, old) for old in occupied) * 10000
                    score += sum(crossing(box, segment) for segment in segments) * 300
                    score += sum(box[0] - 3 < mx < box[2] + 3 and box[1] - 3 < my < box[3] + 3 for mx, my in markers) * 200
                    score += abs(dy) + (0 if side == preferred else 8)
                    candidates.append((score, tx, ty, anchor, box))
            if not candidates:
                raise ValueError(f'No label position for {title}/{x_value}/{label}')
            _, tx, ty, anchor, box = min(candidates, key=lambda row: row[0])
            occupied.append(box)
            if abs(ty - y) > 25 or 'mask_mode' in item['filter']:
                drawing.add(Line(x, y, tx, ty - 2, strokeWidth=.35, strokeColor=HexColor(item['color']), strokeDashArray=[2, 2] if item.get('dashed') else None))
            drawing.add(Rect(box[0], box[1], box[2] - box[0], box[3] - box[1], fillColor=white, strokeColor=None))
            drawing.add(String(tx, ty, label, textAnchor=anchor, fontName='StudyRegular', fontSize=number_font,
                               fillColor=HexColor(item['color'])))
    return drawing


def model_series():
    return [{"filter": {"perturbation_type": kind, "mask_mode": mode, "requested_timestamp_shift_seconds": 0},
             "label": f"{label}, {'динамическая' if mode == 'dynamic' else 'фиксированная'}",
             "color": color, "dashed": mode == "dynamic"}
            for kind, label, color in (("delete", "удаление фильма", "#2563eb"), ("replace_random", "замена фильма", "#d97706"))
            for mode in ("dynamic", "fixed_original")]


def timestamp_series():
    colors = {3600: "#2563eb", 86400: "#7c3aed", 604800: "#059669"}
    return [{"filter": {"requested_timestamp_shift_seconds": shift}, "label": SHIFT_LABELS[shift],
             "color": colors[abs(shift)], "dashed": shift < 0} for shift in SHIFT_LABELS]


def build_charts(directory):
    legacy = pd.read_csv(directory / "summary_legacy_positions.csv")
    tail = pd.read_csv(directory / "summary_tail_positions.csv")
    times = tail.loc[(tail.perturbation_type == "timestamp_only") & (tail.mask_mode == "fixed_original")]
    diagnostics = pd.read_csv(directory / "timestamp_diagnostics_tail.csv")
    mask_only = pd.read_csv(directory / "summary_mask_only_tail.csv")
    figures = {}
    for metric in METRIC_LABELS:
        domain = (0, 1) if metric == "topk_overlap" else (0, 100) if metric == "top1_changed" else None
        figures[f"legacy_{metric}"] = chart(legacy, metric=metric, x_column="position_label", x_values=LEGACY_ORDER,
            x_labels=["первая", "25%", "50%", "75%", "последняя"], series=model_series(), title=metric_title(metric, "сравнение масок"), x_title="Позиция изменяемого фильма", domain=domain)
        figures[f"tail_{metric}"] = chart(tail, metric=metric, x_column="distance_from_end", x_values=list(range(9, -1, -1)),
            x_labels=[str(i) for i in range(10, 0, -1)], series=model_series(), title=metric_title(metric, "последние десять фильмов"), x_title="Номер от конца (1 = последний фильм)", domain=domain)
        time_domain = None
        if metric == "topk_overlap":
            time_domain = (max(0, float(times.topk_overlap_ci_low.min()) - 0.01), 1.002)
        figures[f"timestamp_tail_{metric}"] = chart(times, metric=metric, x_column="distance_from_end", x_values=list(range(9, -1, -1)),
            x_labels=[str(i) for i in range(10, 0, -1)], series=timestamp_series(), title=metric_title(metric, "изменение только времени"), x_title="Номер от конца (1 = последний фильм)", domain=time_domain, height=270)
    figures["timestamp_bucket_coverage"] = chart(diagnostics, metric="time_bucket_changed", x_column="distance_from_end", x_values=list(range(9, -1, -1)),
        x_labels=[str(i) for i in range(10, 0, -1)], series=timestamp_series(), title="Доля возмущений, изменивших временные интервалы модели",
        x_title="Номер от конца (1 = последний фильм)", y_title="Изменён хотя бы один временной интервал, %", scale=100, domain=(0, 100), height=270)
    mask_series = [{"filter": {"perturbation_type": kind}, "label": label, "color": color}
                   for kind, label, color in (("delete", "маска после удаления фильма", "#2563eb"), ("replace_random", "маска после замены", "#d97706"))]
    figures["mask_only_tail_js"] = chart(mask_only, metric="js_divergence", x_column="distance_from_end", x_values=list(range(9, -1, -1)),
        x_labels=[str(i) for i in range(10, 0, -1)], series=mask_series, title=metric_title("js_divergence", "изменение только маски"), x_title="Номер от конца (1 = последний фильм)")
    return figures


def formatted_value(row, metric, decimals=None):
    scale = METRIC_SCALES[metric]
    suffix = "%" if metric == "top1_changed" else ""
    decimals = (1 if metric == "top1_changed" else 3) if decimals is None else decimals
    values = [float(row[f"{metric}_{field}"]) * scale for field in ("mean", "ci_low", "ci_high")]
    return f"{values[0]:.{decimals}f}{suffix} [{values[1]:.{decimals}f}; {values[2]:.{decimals}f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/robustness/stage2"))
    args = parser.parse_args()
    directory = args.input_dir
    output = directory / "plots"
    output.mkdir(parents=True, exist_ok=True)
    figures = build_charts(directory)
    cards = []
    for name, drawing in figures.items():
        svg_path = output / f"{name}.svg"
        renderSVG.drawToFile(drawing, str(svg_path))
        svg_text = svg_path.read_text(encoding="utf-8").replace("font-family: StudyRegular", "font-family: Arial").replace("font-family: StudyBold", "font-family: Arial")
        svg_path.write_text(svg_text, encoding="utf-8")
        renderPDF.drawToFile(drawing, str(output / f"{name}.pdf"))
        subprocess.run(["pdftoppm", "-r", "180", "-singlefile", "-png", str(output / f"{name}.pdf"), str(output / name)], check=True, capture_output=True)
        cards.append(f'<section><h2>{html.escape(name)}</h2><img src="{name}.svg" alt="{html.escape(name)}"></section>')
    text = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>GenRec_HSTU: контролируемая чувствительность</title><style>body{font-family:Arial,sans-serif;margin:32px;background:#f8fafc;color:#0f172a}main{max-width:1000px;margin:auto}section{background:white;padding:20px;margin:20px 0;border:1px solid #e2e8f0}img{width:100%;height:auto}h2{font-size:16px}</style><main><h1>Контролируемая чувствительность HSTU</h1><p>Числа у точек показывают средние значения метрик. Повторы замен сначала усреднены внутри пользователя. На графиках времени сохранены случаи ограниченного и нулевого фактического сдвига.</p>' + "".join(cards) + '</main></html>'
    (output / "controlled_plots.html").write_text(text, encoding="utf-8")
    print(f"Wrote {len(figures)} SVG/PDF/PNG figures and HTML gallery to {output}")


if __name__ == "__main__":
    main()
