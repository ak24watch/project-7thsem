"""Small utility to plot training metrics saved as CSV.

Reads a CSV with columns: epoch, avg_train_loss, avg_val_loss,
avg_train_relative_error, avg_val_relative_error

Produces and saves plots to the same folder by default.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Any


import plotly.graph_objects as go


def read_metrics(csv_path: str) -> Dict[str, Any]:
    """Read a metrics CSV and return separate epoch metrics and sample timing data.

    The repository's `metrics.csv` contains two kinds of rows:
    - Epoch rows with header: epoch,avg_train_loss,avg_val_loss,avg_train_relative_error,...
    - Sample timing rows like: samples,1,elapsed_time,0.1444835 or mom_samples,1,mom_elapsed_time,0.00355

    This function returns a dict with keys:
    - 'epoch': Dict[str, List[float]]  -- numeric columns from epoch table
    - 'samples': Dict[str, Dict[str, List[float]]] -- keyed by label ('samples','mom_samples') -> { 'x': [...], '<metric>': [...] }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metrics CSV not found: {csv_path}")

    epoch_metrics: Dict[str, List[float]] = {}
    sample_metrics: Dict[str, Dict[str, List[float]]] = {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header: List[str] | None = None
        for row in reader:
            # skip empty rows
            if not row or all((c is None or c.strip() == "") for c in row):
                continue
            first = row[0].strip()

            # detect header for epoch table
            if first.lower() == "epoch":
                header = [h.strip() for h in row]
                continue

            # if we have an epoch header and the row fits that layout, parse as epoch metrics
            if header is not None and len(row) >= len(header):
                for k, v in zip(header, row):
                    key = k.strip()
                    if key == "":
                        continue
                    try:
                        val = float(v)
                    except Exception:
                        # skip non-numeric values for epoch table
                        continue
                    epoch_metrics.setdefault(key.lower(), []).append(val)
                continue

            # parse sample timing rows like: samples,1,elapsed_time,0.144...
            if first in ("samples", "mom_samples"):
                label = first
                # expect at least: label, sample_count, key, value
                if len(row) >= 4:
                    try:
                        sample_count = float(row[1])
                    except Exception:
                        sample_count = None
                    key = row[2].strip().lower()
                    try:
                        value = float(row[3])
                    except Exception:
                        continue
                    sm = sample_metrics.setdefault(label, {})
                    # store sample counts as x-axis (only if parsed)
                    if sample_count is not None:
                        sm.setdefault("x", []).append(sample_count)
                    sm.setdefault(key, []).append(value)
                continue

            # otherwise ignore unknown rows

    return {"epoch": epoch_metrics, "samples": sample_metrics}


def plot_losses(
    epoch: List[float],
    train: List[float],
    val: List[float],
    outpath: str | None,
    show: bool = False,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epoch,
            y=train,
            mode="lines",
            name="Train Loss",
            line=dict(color="royalblue", width=3),
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epoch,
            y=val,
            mode="lines",
            name="Val Loss",
            line=dict(color="firebrick", width=3),
            connectgaps=False,
        )
    )
    fig.update_layout(
        title=dict(text="Loss vs Epoch", font=dict(size=20), x=0.5),
        xaxis=dict(title="Epoch", showgrid=True, gridcolor="lightgrey", zeroline=True),
        yaxis=dict(title="Loss", showgrid=True, gridcolor="lightgrey", zeroline=True),
        template="plotly_white",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        font=dict(size=16),
    )
    if outpath:
        fig.write_image(outpath)
    if show:
        fig.show()


def plot_rel_error(
    epoch: List[float],
    train: List[float],
    val: List[float],
    outpath: str | None,
    show: bool = False,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epoch,
            y=train,
            mode="lines",
            name="Train Rel Error",
            line=dict(color="seagreen", width=3),
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epoch,
            y=val,
            mode="lines",
            name="Val Rel Error",
            line=dict(color="orange", width=3),
            connectgaps=False,
        )
    )
    fig.update_layout(
        title=dict(text="Relative L2 Error vs Epoch", font=dict(size=20), x=0.5),
        xaxis=dict(title="Epoch", showgrid=True, gridcolor="lightgrey", zeroline=True),
        yaxis=dict(
            title="Relative Error", showgrid=True, gridcolor="lightgrey", zeroline=True
        ),
        template="plotly_white",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        font=dict(size=16),
    )
    if outpath:
        fig.write_image(outpath)
    if show:
        fig.show()


def plot_runtime_vs_samplesize(
    sample_size: List[float],
    runtime: List[float],
    outpath: str | None,
    show: bool = False,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample_size,
            y=runtime,
            mode="lines+markers",
            name="Runtime",
            line=dict(color="purple", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title=dict(
            text="Remaining CS ResNet: Model Runtime vs Sample Size",
            font=dict(size=20),
            x=0.5,
        ),
        xaxis=dict(
            title="Sample Size", showgrid=True, gridcolor="lightgrey", zeroline=True
        ),
        yaxis=dict(
            title="Model Runtime (s)",
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=True,
        ),
        template="plotly_white",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        font=dict(size=16),
    )
    if outpath:
        fig.write_image(outpath)
    if show:
        fig.show()


def plot_runtime_combined(
    sample_dict: Dict[str, Dict[str, List[float]]],
    outpath_list: List[str] | None,
    show: bool = False,
):
    """Plot runtime traces for multiple sample-labels on the same axes.

    sample_dict: {'samples': {'x': [...], 'elapsed_time': [...]}, 'mom_samples': {...}}
    outpath_list: list of file paths to write the same figure to (or None to not write)
    """
    fig = go.Figure()
    colors = {"samples": "purple", "mom_samples": "green"}
    for label, sm in sample_dict.items():
        x = sm.get("x", [])
        # choose runtime key
        runtime_key = None
        for candidate in ("elapsed_time", "mom_elapsed_time", "runtime"):
            if candidate in sm:
                runtime_key = candidate
                break
        if not x or runtime_key is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sm[runtime_key],
                mode="lines+markers",
                name=f"{label} ({runtime_key})",
                line=dict(color=colors.get(label, None), width=3),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title=dict(
            text="Model Runtime vs Sample Size (combined)", font=dict(size=20), x=0.5
        ),
        xaxis=dict(
            title="Sample Size", showgrid=True, gridcolor="lightgrey", zeroline=True
        ),
        yaxis=dict(
            title="Model Runtime (s)",
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=True,
        ),
        template="plotly_white",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        font=dict(size=14),
    )

    if outpath_list:
        for outpath in outpath_list:
            try:
                fig.write_image(outpath)
            except Exception:
                # fallback: try to write as png via kaleido or skip silently
                try:
                    fig.write_image(outpath)
                except Exception:
                    pass
    if show:
        fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics CSV.")
    parser.add_argument(
        "--csv", default="training_dir/metrics.csv", help="path to metrics CSV"
    )
    parser.add_argument("--outdir", default="training_dir", help="where to save plots")
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    args = parser.parse_args()

    csv_path = args.csv
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    parsed = read_metrics(csv_path)
    epoch_metrics: Dict[str, List[float]] = parsed.get("epoch", {})
    sample_metrics: Dict[str, Dict[str, List[float]]] = parsed.get("samples", {})

    # epoch x-axis
    epoch_x = epoch_metrics.get("epoch")
    if not epoch_x:
        # fallback: use range length of any epoch metric
        if epoch_metrics:
            any_values = next(iter(epoch_metrics.values()))
            epoch_x = list(range(1, len(any_values) + 1))
        else:
            epoch_x = []
    # ensure floats for plotting
    epoch_x = [float(v) for v in epoch_x]

    # Loss keys (lowercased in parsing)
    train_loss = (
        epoch_metrics.get("avg_train_loss") or epoch_metrics.get("train_loss") or []
    )
    val_loss = epoch_metrics.get("avg_val_loss") or epoch_metrics.get("val_loss") or []

    # Relative error
    train_rel = (
        epoch_metrics.get("avg_train_relative_error")
        or epoch_metrics.get("train_relative_error")
        or []
    )
    val_rel = (
        epoch_metrics.get("avg_val_relative_error")
        or epoch_metrics.get("val_relative_error")
        or []
    )

        # Plot epoch-based metrics when available
        if train_loss and val_loss:
            outpath = os.path.join(outdir, "metrics_loss.png")
            plot_losses(
                epoch_x,
                train_loss,
                val_loss,
                outpath if not args.show else None,
                show=args.show,
            )
            print(f"Saved loss plot to {outpath}")

        if train_rel and val_rel:
            outpath2 = os.path.join(outdir, "metrics_rel_error.png")
            plot_rel_error(
                epoch_x,
                train_rel,
                val_rel,
                outpath2 if not args.show else None,
                show=args.show,
            )
            print(f"Saved relative-error plot to {outpath2}")

        # Plot sample timing rows (samples, mom_samples)
        if "samples" in sample_metrics and "mom_samples" in sample_metrics:
            # create a combined plot containing both traces and save to both filenames
            out1 = os.path.join(outdir, "metrics_runtime_samples.png")
            out2 = os.path.join(outdir, "metrics_runtime_mom_samples.png")
            plot_runtime_combined(
                {"samples": sample_metrics["samples"], "mom_samples": sample_metrics["mom_samples"]},
                [out1, out2] if not args.show else None,
                show=args.show,
            )
            print(f"Saved runtime vs sample size plot to {out1}")
            print(f"Saved runtime vs sample size plot to {out2}")
        else:
            # fall back to plotting any available single label traces separately
            for label, sm in sample_metrics.items():
                x = sm.get("x", [])
                # determine runtime key (commonly 'elapsed_time' or 'mom_elapsed_time')
                runtime_key = None
                for candidate in ("elapsed_time", "mom_elapsed_time", "runtime"):
                    if candidate in sm:
                        runtime_key = candidate
                        break
                if not x or runtime_key is None:
                    continue
                outname = f"metrics_runtime_{label}.png"
                outpath3 = os.path.join(outdir, outname)
                plot_runtime_vs_samplesize(
                    x, sm[runtime_key], outpath3 if not args.show else None, show=args.show
                )
                print(f"Saved runtime vs sample size plot to {outpath3}")


if __name__ == "__main__":
    main()
