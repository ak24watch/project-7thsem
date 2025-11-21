import os
import orbax.checkpoint as ocp
import flax.nnx as nnx
from training_dir.train_cno import build_cno_model
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
from training_dir.train_cno import get_config
from training_dir.train_cno import preprocess_batch
from dataLoader.make_data import prepare_dataloader  # not used here


def load_cno_model(config: dict, load_path: str):
    """
    Load a saved Flax NNX CNO_2D model using orbax.

    Args:
        config: Same architecture config used for training.
        load_path: Path to the checkpoint directory (e.g., config["save_path"]).

    Returns:
        The restored model.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {load_path}")

    abs_load_path = os.path.abspath(load_path)

    rngs = nnx.Rngs(config.get("random_seed", 0))
    model = build_cno_model(config, rngs)

    # Create a template state to guide restoration
    abstract_state = nnx.state(model)

    checkpointer = ocp.PyTreeCheckpointer()
    # Restore directly into the abstract template using item=; avoids incorrect args usage.
    restored_state = checkpointer.restore(abs_load_path, item=abstract_state)

    nnx.update(model, restored_state)
    return model


if __name__ == "__main__":
    config = get_config()
    # Example: load the model from the path specified in the training config
    loaded_model = load_cno_model(config, config["save_path"])
    print("Model loaded successfully!")
    loaded_model.eval()  # Set to evaluation mode

    train_loader, val_loader, sizes = prepare_dataloader(
        "dataset", batch_size=1
    )
    print("DataLoader prepared successfully")
    ER, ET = next(iter(val_loader()))
    # ET = jnp.load("triangle_ET.npy").reshape(1, 32, 32)
    # ER = jnp.load("traiangle_image.npy").reshape(1, 32, 32)
    print("shape of ER and ET:", ER.shape, ET.shape)
    EI = jnp.load("EI.npy").reshape(32, 32)
    print("Data batch and EI loaded successfully")
    input, target = jax.vmap(preprocess_batch, in_axes=(0, None, None, 0))(
        ER, EI, config["K0"], ET
    )
    print("Preprocessing successful")
    print(f"Input shape: {input.shape}, Target shape: {target.shape}")
    # Forward pass through the loaded model
    output = loaded_model(input)
    print("Forward pass successful")
    print(f"Output shape: {output.shape}")

    eps_real = ER.real
    eps_imag = ER.imag

    eps_complex = jnp.stack([eps_real, eps_imag], axis=-1)
    print("stacked eps_complex shape:", eps_complex.shape)

    ET_target = target[0, ..., 0] + 1j * target[0, ..., 1]
    ET_predicted = output[0, ..., 0] + 1j * output[0, ..., 1]
    relative_error = jnp.linalg.norm(ET_predicted - ET_target) / jnp.linalg.norm(
        ET_target
    )
    print(f"Relative error between output and target: {relative_error:.6f}")

    fig, axs = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle("Model Evaluation: ER, ET (Target & Predicted)")

    extent = [-1, 1, -1, 1]  # domain from -1 to 1 in both axes

    # ER real
    im0 = axs[0, 0].imshow(eps_real[0], cmap="viridis", extent=extent)
    axs[0, 0].set_title("ER Real")
    plt.colorbar(im0, ax=axs[0, 0])

    # ER imaginary
    im1 = axs[0, 1].imshow(eps_imag[0], cmap="viridis", extent=extent)
    axs[0, 1].set_title("ER Imaginary")
    plt.colorbar(im1, ax=axs[0, 1])

    # ET target real
    im2 = axs[0, 2].imshow(target[0, ..., 0], cmap="viridis", extent=extent)
    axs[0, 2].set_title("ET Target Real")
    plt.colorbar(im2, ax=axs[0, 2])

    # ET target imaginary
    im3 = axs[0, 3].imshow(target[0, ..., 1], cmap="viridis", extent=extent)
    axs[0, 3].set_title("ET Target Imaginary")
    plt.colorbar(im3, ax=axs[0, 3])

    # ET predicted real
    im4 = axs[1, 0].imshow(output[0, ..., 0], cmap="viridis", extent=extent)
    axs[1, 0].set_title("ET Predicted Real")
    plt.colorbar(im4, ax=axs[1, 0])

    # ET predicted imaginary
    im5 = axs[1, 1].imshow(output[0, ..., 1], cmap="viridis", extent=extent)
    axs[1, 1].set_title("ET Predicted Imaginary")
    plt.colorbar(im5, ax=axs[1, 1])

    # ET target absolute
    im6 = axs[1, 2].imshow(np.abs(ET_target), cmap="magma", extent=extent)
    axs[1, 2].set_title("ET Target |Abs|")
    plt.colorbar(im6, ax=axs[1, 2])

    # ET predicted absolute
    im7 = axs[1, 3].imshow(np.abs(ET_predicted), cmap="magma", extent=extent)
    axs[1, 3].set_title("ET Predicted |Abs|")
    plt.colorbar(im7, ax=axs[1, 3])

    # ET target phase
    im8 = axs[2, 0].imshow(np.angle(ET_target), cmap="twilight", extent=extent)
    axs[2, 0].set_title("ET Target Phase")
    plt.colorbar(im8, ax=axs[2, 0])

    # ET predicted phase
    im9 = axs[2, 1].imshow(np.angle(ET_predicted), cmap="twilight", extent=extent)
    axs[2, 1].set_title("ET Predicted Phase")
    plt.colorbar(im9, ax=axs[2, 1])

    # Relative error values
    rel_err_map = np.abs(ET_predicted - ET_target) / (np.abs(ET_target) + 1e-8)
    im10 = axs[2, 2].imshow(rel_err_map, cmap="inferno", extent=extent)
    axs[2, 2].set_title("Relative Error Map")
    plt.colorbar(im10, ax=axs[2, 2])

    # Relative error percentage plot
    rel_err_percent = rel_err_map * 100
    im11 = axs[2, 3].imshow(rel_err_percent, cmap="inferno", extent=extent)
    axs[2, 3].set_title("Relative Error (%)")
    plt.colorbar(im11, ax=axs[2, 3])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("train_complete.png")
