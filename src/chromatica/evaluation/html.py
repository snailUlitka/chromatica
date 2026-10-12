"""Standalone HTML summaries of completed, comparable offline evaluations."""

from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from typing import TYPE_CHECKING, Any

from chromatica.evaluation.experiment import comparison_rows

if TYPE_CHECKING:
    from pathlib import Path


def _table(rows: list[dict[str, Any]]) -> str:
    """Escape primitive values in a horizontally scrollable results table."""
    columns = list(rows[0])
    header = "".join(f"<th scope='col'>{escape(key)}</th>" for key in columns)
    body = []
    for row in rows:
        cells = []
        for key in columns:
            value = row[key]
            label = "undefined" if value is None else str(value)
            if isinstance(value, float):
                label = f"{value:.6g}"
            cells.append(f"<td>{escape(label)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<div class='scroll'><table><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _hue_chart(counts: list[int], title: str) -> str:
    """Plot eligible-pixel percentages with explicit hue and percentage axes."""
    total = sum(counts)
    if not total:
        return f"<p>{escape(title)}: no eligible pixels.</p>"
    percentages = [100 * count / total for count in counts]
    peak = max(percentages)
    width, height = 720, 180
    points = " ".join(
        f"{index * width / len(counts):.2f},{height - value * height / peak:.2f}"
        for index, value in enumerate(percentages)
    )
    return (
        f"<figure><figcaption>{escape(title)} — {total:,} eligible pixels; "
        f"peak {peak:.3g}% per bin</figcaption>"
        f"<svg viewBox='-10 -10 {width + 20} {height + 35}' role='img' "
        f"aria-label='{escape(title, quote=True)} hue histogram'>"
        f"<polyline points='{points}' fill='none' stroke='#2563eb' "
        "stroke-width='2'/>"
        f"<path d='M0 0 V{height} H{width}' fill='none' stroke='#64748b'/>"
        f"<text x='0' y='{height + 20}'>0°</text>"
        f"<text x='{width / 2}' y='{height + 20}'>Hue (degrees)</text>"
        f"<text x='{width}' y='{height + 20}' text-anchor='end'>360°</text>"
        "</svg></figure>"
    )


def write_html_report(paths: list[Path], output: Path) -> None:
    """Read saved evaluations and history, without training or model inference.

    Comparisons are checked separately for each evaluated dataset. Raw reports
    retain full manifests; the self-contained HTML contains compact provenance.
    Existing HTML files are never overwritten.
    """
    if output.exists():
        msg = f"Refusing to overwrite an existing HTML report: {output}"
        raise FileExistsError(msg)
    if not paths:
        msg = "At least one completed evaluation is required."
        raise ValueError(msg)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        groups[report["config"]["data"]["name"]].append(path)
    sections = []
    for dataset, group in groups.items():
        rows = comparison_rows(group)
        for row in rows:
            row.pop("training_config")
        sections.append(f"<h2>{escape(dataset)}: architecture comparison</h2>")
        sections.append(_table(rows))
        for path in group:
            report = json.loads(path.read_text(encoding="utf-8"))
            config = report["config"]
            architecture = report["checkpoint"]["architecture"]
            sections.append(f"<h3>{escape(architecture)}</h3>")
            sections.append(
                f"<p>Available test images: {report['samples_available']}; "
                f"evaluated pixels: {report['metrics']['pixels']:,}. "
                f"Evaluation batch limit: {config['evaluation']['max_batches']}. "
                f"Training batch limits: {config['training']['max_train_batches']} / "
                f"{config['training']['max_val_batches']} (train / validation). "
                "None means no configured limit.</p>"
            )
            sections.append(_table([report["red_sector"]]))
            sections.append(_hue_chart(report["hue_gt"], "Held-out originals"))
            sections.append(_hue_chart(report["hue_pred"], "Model predictions"))
            history_path = path.parent / "history.json"
            if history_path.is_file():
                history = json.loads(history_path.read_text(encoding="utf-8"))
                if history:
                    sections.append("<h4>Training history (normalized-ab MSE)</h4>")
                    sections.append(_table(history))
            provenance = {
                key: report[key]
                for key in (
                    "config",
                    "environment",
                    "checkpoint",
                    "test_manifest_sha256",
                    "metric_version",
                    "lab_encoding",
                    "preprocessing_version",
                    "notes",
                )
            }
            sections.append(
                "<details><summary>Configuration and provenance</summary><pre>"
                + escape(json.dumps(provenance, ensure_ascii=False, indent=2))
                + "</pre></details>"
            )
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Chromatica experiment results</title><style>"
        "body{font:16px/1.5 system-ui,sans-serif;margin:32px auto;padding:0 24px;"
        "max-width:1200px;color:#172033;background:#f8fafc}"
        "h2{margin-top:48px}h3{margin-top:32px}.scroll,pre{overflow:auto}"
        "table{border-collapse:collapse;background:white;font-size:14px}"
        "th,td{padding:10px;border:1px solid #cbd5e1;text-align:left}"
        "svg{width:100%;max-width:800px}svg text{font-size:13px;fill:#475569}"
        "figure{margin:24px 0}pre{font-size:13px}details{margin-top:24px}"
        "</style><body><h1>Chromatica experiment results</h1>"
        "<p>Independent training and held-out evaluation. Lab errors use physical "
        "a*/b* units; normalized MSE uses the checkpoint encoding. "
        "Undefined values are labelled explicitly. Hue plots normalize each "
        "eligible population independently and use separate vertical scales.</p>"
        "<p>The architectures differ in Tanh/ReLU output activation as well as "
        "skip connections. These results cannot isolate a skip-connection effect. "
        "One seed and a three-epoch template do not reproduce the publications "
        "or establish statistical uncertainty. Batch-limited runs are exploratory. "
        "Source evaluation JSON files retain full test manifests.</p>"
        + "".join(sections)
        + "</body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document + "\n")
