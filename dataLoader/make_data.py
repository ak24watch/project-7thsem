import os
import re
import numpy as np
import jax.numpy as jnp
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

def load_full_dataset(folder):
    er_pattern = re.compile(r"er_(\d+)\.npy$")
    et_pattern = re.compile(r"Et_(\d+)\.npy$")

    er_files = {}
    et_files = {}

    # Scan folder and collect file names
    for f in os.listdir(folder):
        mer = er_pattern.match(f)
        met = et_pattern.match(f)
        if mer:
            er_files[int(mer.group(1))] = f
        elif met:
            et_files[int(met.group(1))] = f

    # Only IDs that match
    ids = sorted(set(er_files) & set(et_files))

    ER_list = []
    ET_list = []

    # Load everything directly into RAM
    for i in ids:
        er = np.load(os.path.join(folder, er_files[i]))
        et = np.load(os.path.join(folder, et_files[i]))

        ER_list.append(er)
        ET_list.append(et)

    # Convert to JAX arrays
    ER = jnp.array(ER_list)
    ET = jnp.array(ET_list)

    print("Loaded dataset:")
    print("ER shape:", ER.shape)
    print("ET shape:", ET.shape)

    return ER, ET


def create_dataloader(ER, ET, batch_size, shuffle=True):
    N = ER.shape[0]
    ER = ER.reshape(N, 88, 88)
    ET = ET.reshape(N, 88, 88)

    def dataloader():
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, N, batch_size):
            b = idx[start : start + batch_size]
            yield ER[b], ET[b]

    return dataloader


def plot_sample(ER, ET, sample_idx=0):
    # Select sample and reshape to 88x88
    er_sample = ER[sample_idx].reshape(88, 88)
    et_sample = ET[sample_idx].reshape(88, 88)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "ER Real Part",
            "ER Imaginary Part",
            "ET Real Part",
            "ET Imaginary Part",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    heatmaps = [
        (jnp.real(er_sample), 1, 1),
        (jnp.imag(er_sample), 1, 2),
        (jnp.real(et_sample), 2, 1),
        (jnp.imag(et_sample), 2, 2),
    ]

    for z, row, col in heatmaps:
        fig.add_trace(
            go.Heatmap(z=z, colorbar=dict(len=0.5, y=0.25 if row == 1 else 0.75)),
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_xaxes(title_text="X", row=2, col=1)
    fig.update_xaxes(title_text="X", row=2, col=2)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=2)
    fig.update_yaxes(title_text="Y", row=2, col=1)
    fig.update_yaxes(title_text="Y", row=2, col=2)

    fig.update_layout(
        height=700,
        width=700,
        title_text="Sample ER & ET Real/Imaginary Parts",
        showlegend=False,
    )
    fig.show(renderer="browser")

def plot_e_forward(e_forward):
    # Reshape to (88, 88)
    e_forward_reshaped = e_forward.reshape(88, 88)
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["e_forward Real", "e_forward Imag"]
    )
    # Add real part (no colorbar)
    fig.add_trace(
        go.Heatmap(z=jnp.real(e_forward_reshaped), showscale=False), row=1, col=1
    )
    # Add imaginary part (colorbar shown)
    fig.add_trace(
        go.Heatmap(
            z=jnp.imag(e_forward_reshaped), colorbar=dict(title="Value"), showscale=True
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title_text="e_forward Real and Imaginary Parts", width=900, height=400
    )
    pio.show(fig, renderer="browser")


def prepare_dataloader(folder, batch_size):
    ER, ET = load_full_dataset(folder)
    dataloader = create_dataloader(ER, ET, batch_size)
    return dataloader


if __name__ == "__main__":
    # Example usage
    ER, ET = load_full_dataset("dataset")

    loader = create_dataloader(ER, ET, batch_size=32)

    fb_ER, fb_ET = next(iter(loader()))  # Example of getting the first batch

    # Plot one sample from ER and ET (first sample in batch)
    plot_sample(fb_ER, fb_ET, sample_idx=0)
