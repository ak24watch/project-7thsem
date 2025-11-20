import os
import orbax.checkpoint as ocp
import flax.nnx as nnx
from training_dir.train_cno import build_cno_model
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from training_dir.train_cno import get_config
from training_dir.train_cno import preprocess_batch
from dataLoader.make_data import prepare_dataloader


def plot_complex_fields(output, target, ER, save_path=None):
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    fields = [(output, "Output"), (target, "Target"), (ER, "ER")]
    for i, (field, name) in enumerate(fields):
        # field shape: (32, 32, 2), where last dim is (real, imag)
        real = field[..., 0]
        imag = field[..., 1]
        complex_field = real + 1j * imag
        abs_val = jnp.abs(complex_field)
        phase = jnp.angle(complex_field)
        axs[i, 0].imshow(real, cmap="viridis")
        axs[i, 0].set_title(f"{name} Real")
        axs[i, 1].imshow(imag, cmap="viridis")
        axs[i, 1].set_title(f"{name} Imaginary")
        axs[i, 2].imshow(abs_val, cmap="viridis")
        axs[i, 2].set_title(f"{name} Absolute")
        axs[i, 3].imshow(phase, cmap="twilight")
        axs[i, 3].set_title(f"{name} Phase")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


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

    # train_loader, val_loader, sizes = prepare_dataloader(
    #     "dataset", batch_size=1
    # )
    print("DataLoader prepared successfully")
    # ER, ET = next(iter(train_loader()))
    ET = jnp.load("triangle_ET.npy").reshape(1, 32, 32)
    ER = jnp.load("traiangle_image.npy").reshape(1, 32, 32)
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

    plot_complex_fields(output[0], target[0], eps_complex[0], save_path="test.png")
    ET_target = target[0, ..., 0] + 1j * target[0, ..., 1]
    ET_predicted = output[0, ..., 0] + 1j * output[0, ..., 1]
    relative_error = jnp.linalg.norm(ET_predicted - ET_target) / jnp.linalg.norm(
        ET_target
    )
    print(f"Relative error between output and target: {relative_error:.6f}")
