"""Collect figures and tables for the dynamic-versus-fixed masking study only."""
import argparse
import html
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Line
from reportlab.lib.colors import HexColor

from generative_recommenders_pl.scripts.plot_controlled_robustness import (
    chart, metric_title, model_series, METRIC_LABELS, METRIC_SCALES, LEGACY_ORDER,
)

KINDS = (("delete", "удаление фильма", "#2563eb"), ("replace_random", "замена фильма", "#d97706"))


def build_masking_charts(directory):
    figures = {}
    tables = {}
    for family, suffix, x_column, x_values, x_labels, x_title in (
        ("legacy", "legacy_positions", "position_label", LEGACY_ORDER,
         ["первая", "25%", "50%", "75%", "последняя"], "Позиция изменяемого фильма"),
        ("tail", "tail_positions", "distance_from_end", list(range(9, -1, -1)),
         [str(i) for i in range(10, 0, -1)], "Номер от конца (1 = последний фильм)"),
    ):
        main = pd.read_csv(directory / f"summary_{suffix}.csv")
        main = main.loc[main.perturbation_type.isin([k[0] for k in KINDS]) & (main.requested_timestamp_shift_seconds == 0)]
        paired = pd.read_csv(directory / f"paired_mask_differences_{family}.csv")
        mask_only = pd.read_csv(directory / f"summary_mask_only_{family}.csv")
        tables[f"comparison_{family}"] = main
        tables[f"difference_{family}"] = paired
        tables[f"mask_only_{family}"] = mask_only
        pair_series = [{"filter": {"perturbation_type": kind}, "label": label, "color": color}
                       for kind, label, color in KINDS]
        prefix = "Пять позиций" if family == "legacy" else "Последние десять элементов"
        for metric in METRIC_LABELS:
            domain = (0, 1) if metric == "topk_overlap" else (0, 100) if metric == "top1_changed" else None
            figures[f"comparison_{family}_{metric}"] = chart(
                main, metric=metric, x_column=x_column, x_values=x_values, x_labels=x_labels,
                series=model_series(), title=metric_title(metric, "сравнение масок" + (" (последние десять)" if family == "tail" else "")),
                x_title=x_title, domain=domain)
            extent = max(abs(paired[f"{metric}_{bound}"]).max() * METRIC_SCALES[metric]
                         for bound in ("ci_low", "ci_high"))
            extent = max(float(extent) * 1.15, 1e-6)
            y_title = ("Разность долей Top-1 changed, п.п." if metric == "top1_changed"
                       else f"Разность {METRIC_LABELS[metric]}")
            drawing = chart(
                paired, metric=metric, x_column=x_column, x_values=x_values, x_labels=x_labels,
                series=pair_series, title=metric_title(metric, "динамическая минус фиксированная" + (" (последние десять)" if family == "tail" else "")),
                x_title=x_title, domain=(-extent, extent), y_title=y_title)
            figures[f"difference_{family}_{metric}"] = drawing
            figures[f"mask_only_{family}_{metric}"] = chart(
                mask_only, metric=metric, x_column=x_column, x_values=x_values, x_labels=x_labels,
                series=pair_series, title=metric_title(metric, "изменение только маски" + (" (последние десять)" if family == "tail" else "")),
                x_title=x_title, domain=domain)
    return figures, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/robustness/stage2"))
    parser.add_argument("--pdftoppm", default="pdftoppm", help="Poppler executable for Markdown/PDF PNG figures")
    args = parser.parse_args()
    executable = shutil.which(args.pdftoppm)
    if not executable:
        raise FileNotFoundError("pdftoppm not found; pass --pdftoppm PATH to the Poppler executable")
    output = args.input_dir / "plots/masking"
    output.mkdir(parents=True, exist_ok=True)
    figures, tables = build_masking_charts(args.input_dir)
    cards = []
    for name, drawing in figures.items():
        svg = output / f"{name}.svg"
        pdf = output / f"{name}.pdf"
        renderSVG.drawToFile(drawing, str(svg))
        svg.write_text(svg.read_text(encoding="utf-8").replace("font-family: StudyRegular", "font-family: Arial").replace("font-family: StudyBold", "font-family: Arial"), encoding="utf-8")
        renderPDF.drawToFile(drawing, str(pdf))
        subprocess.run([executable, "-r", "220", "-singlefile", "-png", str(pdf), str(output / name)], check=True, capture_output=True)
        cards.append(f'<section id="{name}"><h2>{html.escape(name)}</h2><img src="{name}.svg" alt="{html.escape(name)}"></section>')
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    gallery = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение масок HSTU</title>'
               '<style>body{font-family:Arial,sans-serif;background:#f8fafc;color:#17212b;margin:30px}main{max-width:1000px;margin:auto}section{background:white;border:1px solid #d8e0ea;padding:20px;margin:20px 0}img{width:100%;height:auto}h2{font-size:16px}</style>'
               '<main><h1>Динамическая и фиксированная маски</h1><p>Сравнение режимов, парные разности и изменение только маски. '
               'Числа у точек показывают средние значения метрик. '
               'JS умножена на 1000; Top-1 changed показан в процентах, разность долей - в процентных пунктах.</p>'
               + ''.join(cards) + '</main></html>')
    (output / "masking_plots.html").write_text(gallery, encoding="utf-8")
    print(f"Wrote {len(figures)} mask-study figures (SVG/PDF/PNG), {len(tables)} CSV tables and HTML gallery to {output}")


if __name__ == "__main__":
    main()
