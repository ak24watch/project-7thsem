"""Small utility to plot training metrics saved as CSV.

Reads a CSV with columns: epoch, avg_train_loss, avg_val_loss,
avg_train_relative_error, avg_val_relative_error

Produces and saves plots to the same folder by default.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List


import plotly.graph_objects as go


def read_metrics(csv_path: str) -> Dict[str, List[float]]:
	metrics: Dict[str, List[float]] = {}
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"metrics CSV not found: {csv_path}")

	with open(csv_path, newline="") as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames
		for row in reader:
			for k in fieldnames:
				v = row.get(k, "")
				if k is None:
					continue
				k = k.strip()
				if k == "":
					continue
				metrics.setdefault(k, []).append(float(v) if v != "" else float("nan"))

	return metrics



def plot_losses(epoch: List[float], train: List[float], val: List[float], outpath: str | None, show: bool = False):
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=epoch, y=train, mode='lines', name='Train Loss',
							 line=dict(color='royalblue', width=3),
							 connectgaps=False))
	fig.add_trace(go.Scatter(x=epoch, y=val, mode='lines', name='Val Loss',
							 line=dict(color='firebrick', width=3),
							 connectgaps=False))
	fig.update_layout(
		title=dict(text='Loss vs Epoch', font=dict(size=20), x=0.5),
		xaxis=dict(title='Epoch', showgrid=True, gridcolor='lightgrey', zeroline=True),
		yaxis=dict(title='Loss', showgrid=True, gridcolor='lightgrey', zeroline=True),
		template='plotly_white',
		legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', bordercolor='lightgrey', borderwidth=1),
		font=dict(size=16)
	)
	if outpath:
		fig.write_image(outpath)
	if show:
		fig.show()



def plot_rel_error(epoch: List[float], train: List[float], val: List[float], outpath: str | None, show: bool = False):
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=epoch, y=train, mode='lines', name='Train Rel Error',
							 line=dict(color='seagreen', width=3),
							 connectgaps=False))
	fig.add_trace(go.Scatter(x=epoch, y=val, mode='lines', name='Val Rel Error',
							 line=dict(color='orange', width=3),
							 connectgaps=False))
	fig.update_layout(
		title=dict(text='Relative L2 Error vs Epoch', font=dict(size=20), x=0.5),
		xaxis=dict(title='Epoch', showgrid=True, gridcolor='lightgrey', zeroline=True),
		yaxis=dict(title='Relative Error', showgrid=True, gridcolor='lightgrey', zeroline=True),
		template='plotly_white',
		legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', bordercolor='lightgrey', borderwidth=1),
		font=dict(size=16)
	)
	if outpath:
		fig.write_image(outpath)
	if show:
		fig.show()


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot training metrics CSV.")
	parser.add_argument("--csv", default="training_dir/metrics.csv", help="path to metrics CSV")
	parser.add_argument("--outdir", default="training_dir", help="where to save plots")
	parser.add_argument("--show", action="store_true", help="show plots interactively")
	args = parser.parse_args()

	csv_path = args.csv
	outdir = args.outdir
	os.makedirs(outdir, exist_ok=True)

	metrics = read_metrics(csv_path)

	# prefer column names with/without spaces
	epoch = metrics.get("epoch") or metrics.get("Epoch") or list(range(1, len(next(iter(metrics.values()))) + 1))

	# Loss
	train_loss = metrics.get("avg_train_loss") or metrics.get("train_loss") or []
	val_loss = metrics.get("avg_val_loss") or metrics.get("val_loss") or []

	# Relative error
	train_rel = metrics.get("avg_train_relative_error") or metrics.get("train_relative_error") or []
	val_rel = metrics.get("avg_val_relative_error") or metrics.get("val_relative_error") or []


	if train_loss and val_loss:
		outpath = os.path.join(outdir, "metrics_loss.png")
		plot_losses(epoch, train_loss, val_loss, outpath if not args.show else None, show=args.show)
		print(f"Saved loss plot to {outpath}")

	if train_rel and val_rel:
		outpath2 = os.path.join(outdir, "metrics_rel_error.png")
		plot_rel_error(epoch, train_rel, val_rel, outpath2 if not args.show else None, show=args.show)
		print(f"Saved relative-error plot to {outpath2}")


if __name__ == "__main__":
	main()
