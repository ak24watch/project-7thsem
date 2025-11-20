import os
import re
import numpy as np
import jax.numpy as jnp
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio


def load_full_dataset(folder):
    er_pattern = re.compile(r"er_(\d+)\.npy$", re.IGNORECASE)
    et_pattern = re.compile(r"et_(\d+)\.npy$", re.IGNORECASE)

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

        # Check for NaN or Inf values in loaded arrays
        if np.isnan(er).any() or np.isinf(er).any():
            print(f"Warning: er_{i}.npy contains NaN or Inf values! Skipping.")
            continue
        if np.isnan(et).any() or np.isinf(et).any():
            print(f"Warning: Et_{i}.npy contains NaN or Inf values! Skipping.")
            continue

        ER_list.append(er)
        ET_list.append(et)

    # Convert to JAX arrays
    ER = jnp.array(ER_list)
    ET = jnp.array(ET_list)

    # print("Loaded dataset:")
    # print("ER shape:", ER.shape)
    # print("ET shape:", ET.shape)

    # Check for NaNs or Infs in ER and ET
    if jnp.isnan(ER).any() or jnp.isinf(ER).any():
        print("Warning: ER contains NaN or Inf values!")
    # if jnp.isnan(ET).any() or jnp.isinf(ET).any():
    #     print("Warning: ET contains NaN or Inf values!")

    return ER, ET


def create_dataloader(ER, ET, batch_size, shuffle=True):
    N = ER.shape[0]
    print("shape of ER:", ER.shape)
    ER = ER.reshape(N, 32, 32)
    ET = ET.reshape(N, 32, 32)
    # Check for NaNs or Infs in ER and ET
    if jnp.isnan(ER).any() or jnp.isinf(ER).any():
        print("Warning: ER contains NaN or Inf values!")
    if jnp.isnan(ET).any() or jnp.isinf(ET).any():
        print("Warning: ET contains NaN or Inf values!")

    def dataloader():
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, N, batch_size):
            b = idx[start : start + batch_size]
            yield ER[b], ET[b]

    return dataloader


def plot_sample(ER, ET, sample_idx=0):
    # Select sample and reshape to 32x32
    er_sample = ER[sample_idx].reshape(32, 32)
    et_sample = ET[sample_idx].reshape(32, 32)

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
    # Reshape to (32, 32)
    e_forward_reshaped = e_forward.reshape(32, 32)
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


def split_dataset(ER, ET, val_ratio=0.2, shuffle=True, seed=42):
    """
    Split ER, ET into train and validation sets.

    Args:
        ER: jnp.ndarray of shape (N, 32, 32) or (N, ...)
        ET: jnp.ndarray of shape (N, 32, 32) or (N, ...)
        val_ratio: fraction of data to use for validation
        shuffle: whether to shuffle before split
        seed: random seed for shuffling

    Returns:
        (ER_train, ET_train), (ER_val, ET_val)
    """
    import numpy as np

    N = ER.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_val = int(N * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    ER_train = ER[train_idx]
    ET_train = ET[train_idx]
    ER_val = ER[val_idx]
    ET_val = ET[val_idx]

    return (ER_train, ET_train), (ER_val, ET_val)


def prepare_dataloader(folder, batch_size, val_ratio=0.2, seed=42):
    """
    Load the full dataset, split into train/val, and create dataloaders for each.

    Returns:
        train_loader, val_loader, sizes_dict
    where sizes_dict = {"train": N_train, "val": N_val}
    """
    ER, ET = load_full_dataset(folder)
    (ER_tr, ET_tr), (ER_va, ET_va) = split_dataset(
        ER, ET, val_ratio=val_ratio, shuffle=True, seed=seed
    )
    train_loader = create_dataloader(ER_tr, ET_tr, batch_size, shuffle=True)
    val_loader = create_dataloader(ER_va, ET_va, batch_size, shuffle=False)
    sizes = {"train": int(ER_tr.shape[0]), "val": int(ER_va.shape[0])}
    return train_loader, val_loader, sizes


if __name__ == "__main__":
    # Example usage
    ER, ET = load_full_dataset("dataset")

    loader = create_dataloader(ER, ET, batch_size=32)

    # fb_ER, fb_ET = next(iter(loader()))  # Example of getting the first batch

    # # Plot one sample from ER and ET (first sample in batch)
    # plot_sample(fb_ER, fb_ET, sample_idx=0)
