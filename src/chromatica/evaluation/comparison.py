"""Export matched offline evaluations into a plain CSV research table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from chromatica.evaluation.experiment import comparison_rows


def main() -> None:
    """Export only existing evaluation reports; no model is run by this command."""
    parser = argparse.ArgumentParser(description="Compare existing evaluation reports.")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        msg = f"Refusing to overwrite {args.output}."
        raise FileExistsError(msg)
    rows = comparison_rows(args.reports)
    for row in rows:
        row.pop("training_config")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
