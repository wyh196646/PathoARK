#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import os
import statistics as stats
from typing import Dict, List, Optional, Tuple


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read_summary_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV into list of dict rows. Returns (fieldnames, rows)."""
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                # Skip fully empty lines
                if not any(v.strip() for v in row.values() if v is not None):
                    continue
                rows.append(row)
    except FileNotFoundError:
        pass
    return fieldnames, rows


def _select_row(rows: List[Dict[str, str]], selection: str, key_metric: str) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    sel = selection.lower()
    if sel == "last":
        return rows[-1]
    if sel == "mean":
        # Mean handled at metric aggregation time; return None to signal special handling
        return None
    # Default: best
    candidates = [(r, _to_float(r.get(key_metric))) for r in rows]
    candidates = [c for c in candidates if c[1] is not None]
    if not candidates:
        # Fallback to last if key metric missing
        return rows[-1]
    best_row, _ = max(candidates, key=lambda x: x[1])
    return best_row


def _aggregate_metrics(rows: List[Dict[str, str]], metrics: List[str], selection: str, key_metric: str) -> Dict[str, Optional[float]]:
    sel = selection.lower()
    if not rows:
        return {m: None for m in metrics}

    if sel == "mean":
        # Average each metric over available rows (ignoring non-numeric/missing)
        out: Dict[str, Optional[float]] = {}
        for m in metrics:
            vals = [_to_float(r.get(m)) for r in rows]
            vals = [v for v in vals if v is not None]
            out[m] = sum(vals) / len(vals) if vals else None
        return out

    # best or last: compute from selected row
    chosen = _select_row(rows, selection, key_metric)
    if chosen is None:  # shouldn't happen for best/last
        return {m: None for m in metrics}
    return {m: _to_float(chosen.get(m)) for m in metrics}


def _aggregate_cv(rows: List[Dict[str, str]], metrics: List[str]) -> Tuple[Dict[str, str], Dict[str, Optional[float]]]:
    """Aggregate metrics across CV folds (rows) and return formatted strings and means.

    Returns (formatted_map, mean_map) where formatted_map[m] = "mean+-std" string
    and mean_map[m] = mean float (or None when unavailable).
    """
    formatted: Dict[str, str] = {}
    means: Dict[str, Optional[float]] = {}
    for m in metrics:
        vals = [_to_float(r.get(m)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            formatted[m] = ""
            means[m] = None
            continue
        mean_v = sum(vals) / len(vals)
        std_v = stats.stdev(vals) if len(vals) >= 2 else 0.0
        means[m] = mean_v
        # Use ascii '+-' for portability
        formatted[m] = f"{mean_v:.4f}+-{std_v:.4f}"
    return formatted, means


def extract_experiment_results(
    base_dir: str,
    out_csv: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    selection: str = "cv",
    key_metric: str = "val_weighted_f1",
    mode: str = "simple",  # "simple" -> only metrics per model_lr, "full" -> original verbose format
) -> str:
    """Walk base_dir, find summary.csv files, and aggregate selected metrics to CSV.

    mode:
      - "simple": output one row per model+learning_rate (model_lr) with only the requested metrics.
                   If multiple entries map to the same model_lr, keep the one with highest key_metric.
      - "full":   original verbose output with metadata columns.

    Returns the output CSV path.
    """
    if metrics is None:
        metrics = ["val_bacc", "val_macro_auroc", "val_weighted_f1"]

    base_dir = os.path.abspath(base_dir)
    dataset_name = os.path.basename(base_dir.rstrip(os.sep))
    results: List[Dict[str, object]] = []
    simple_best: Dict[str, Dict[str, object]] = {}

    for dirpath, _, filenames in os.walk(base_dir):
        if "summary.csv" not in filenames:
            continue
        summary_path = os.path.join(dirpath, "summary.csv")

        # Derive meta fields from path
        rel_dir = os.path.relpath(dirpath, base_dir)
        parts = rel_dir.split(os.sep) if rel_dir != "." else []
        model = parts[0] if len(parts) >= 1 else None
        strategy = parts[1] if len(parts) >= 2 else None
        hyperparam = parts[2] if len(parts) >= 3 else None

        _, rows = _read_summary_csv(summary_path)
        if not rows:
            # still record an entry with Nones (full mode), or skip (simple mode)
            if mode == "full":
                entry: Dict[str, object] = {
                    "dataset": dataset_name,
                    "model": model,
                    "strategy": strategy,
                    "hyperparam": hyperparam,
                    "rows_in_summary": 0,
                    "summary_path": summary_path,
                    "selection": selection,
                    "key_metric": key_metric,
                }
                for m in metrics:
                    entry[m] = None
                results.append(entry)
            continue

        sel_lower = selection.lower()
        # When selection indicates CV aggregation, compute mean±std across rows
        if sel_lower in {"cv", "meanstd", "fold_meanstd"}:
            formatted_values, mean_values = _aggregate_cv(rows, metrics)
            metric_values = formatted_values
            key_for_compare = mean_values.get(key_metric)
        else:
            metric_values = _aggregate_metrics(rows, metrics, selection, key_metric)
            key_for_compare = _to_float(metric_values.get(key_metric))

        if mode == "full":
            entry = {
                "dataset": dataset_name,
                "model": model,
                "strategy": strategy,
                "hyperparam": hyperparam,
                "rows_in_summary": len(rows),
                "selection": selection,
                "key_metric": key_metric,
                "summary_path": summary_path,
                **metric_values,
            }
            # Provide selected key metric value for convenience when selection=best/last
            if selection.lower() != "mean":
                entry[f"{key_metric}_selected"] = metric_values.get(key_metric)
            results.append(entry)
        else:
            # simple mode: keep only metrics per model+lr; choose best by key_metric if duplicates
            model_lr = f"{model}_{hyperparam}" if model is not None and hyperparam is not None else None
            if model_lr is None:
                # If model or lr missing, skip this entry in simple mode
                continue
            cur_best = simple_best.get(model_lr)
            cur_key_val = key_for_compare
            if cur_best is None:
                # store hidden compare value to handle de-duplication across repeats
                simple_best[model_lr] = {"model_lr": model_lr, "__key_metric_mean": cur_key_val, **metric_values}
            else:
                prev_key_val = cur_best.get("__key_metric_mean")
                # Replace if current is better (or if previous is None and current is not None)
                if (prev_key_val is None and cur_key_val is not None) or (
                    prev_key_val is not None and cur_key_val is not None and cur_key_val > prev_key_val
                ):
                    simple_best[model_lr] = {"model_lr": model_lr, "__key_metric_mean": cur_key_val, **metric_values}

    # Ensure deterministic ordering
    if mode == "full":
        results.sort(key=lambda r: (
            str(r.get("model") or ""),
            str(r.get("strategy") or ""),
            str(r.get("hyperparam") or ""),
            str(r.get("summary_path") or ""),
        ))
    else:
        # Sort by model_lr
        results = [simple_best[k] for k in sorted(simple_best.keys())]

    if out_csv is None:
        out_csv = os.path.join(base_dir, "aggregated_summary.csv")

    # Create parent directories if needed
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Determine fieldnames
    if mode == "full":
        base_fields = [
            "dataset",
            "model",
            "strategy",
            "hyperparam",
            "rows_in_summary",
            "selection",
            "key_metric",
            "summary_path",
        ]
        extended = list(metrics)
        optional_fields = [f"{key_metric}_selected"] if selection.lower() != "mean" else []
        fieldnames = base_fields + extended + optional_fields
    else:
        # simple mode: only an identifier and requested metrics
        fieldnames = ["model_lr"] + list(metrics)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # only keep defined fields to avoid KeyError
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Found {len(results)} experiment(s). Wrote aggregated CSV to: {out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment summary.csv metrics.")
    parser.add_argument("--base-dir", default='/home/yuhaowang/project/PathARK/outputs/AIDPATH_CERB2', help="Base directory containing experiments (e.g., outputs/AIDPATH_CERB2)")
    parser.add_argument("--out-csv", help="Output CSV path; defaults to <base-dir>/aggregated_summary.csv")
    parser.add_argument("--metrics", default="val_bacc,val_macro_auroc,val_weighted_f1", help="Comma-separated list of metrics to extract")
    parser.add_argument("--selection", choices=["cv", "best", "mean", "last"], default="cv", help="How to select/aggregate rows from summary.csv (cv computes mean+-std across folds)")
    parser.add_argument("--key-metric", default="val_weighted_f1", help="Metric used to choose 'best' row")
    parser.add_argument("--mode", choices=["simple", "full"], default="simple", help="Output mode: simple (rows per model+lr with only metrics) or full (verbose)")

    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    extract_experiment_results(
        base_dir=args.base_dir,
        out_csv=args.out_csv,
        metrics=metrics,
        selection=args.selection,
        key_metric=args.key_metric,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
