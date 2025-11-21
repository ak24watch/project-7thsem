import os
import time
import orbax.checkpoint as ocp
import flax.nnx as nnx
from training_dir.train_cno import build_cno_model
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from training_dir.train_cno import get_config
from training_dir.train_cno import preprocess_batch
from dataLoader.make_data import prepare_dataloader
import csv


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


def test_model_forward(inputs, model_state):
    model = nnx.merge(model_graphdef, model_state)
    model.eval()
    _ = model(inputs)
    return model_state


if __name__ == "__main__":
    config = get_config()
    # Example: load the model from the path specified in the training config
    loaded_model = load_cno_model(config, config["save_path"])
    print("Model loaded successfully!")
    # loaded_model.eval()  # Set to evaluation mode

    # Prepare dataloader using training data only (no validation)
    train_loader, val_loader, sizes = prepare_dataloader(
        config["data_folder"],
        batch_size=32000,
        val_ratio=0.0,
        seed=config.get("data_seed", 42),
    )
    print(f"DataLoader prepared: sizes={sizes}")
    # Use the first batch from the train_loader (batch_size=1)
    ER, ET = next(iter(train_loader()))
    print("shape of ER and ET:", ER.shape, ET.shape)
    EI = jnp.load("EI.npy").reshape(32, 32)
    print("Data batch and EI loaded successfully")
    # Preprocess the batch exactly as in training (vmap over the batch dim)
    inputs, targets = jax.vmap(preprocess_batch, in_axes=(0, None, None, 0))(
        ER, EI, config["K0"], ET
    )
    print("Preprocessing successful")
    print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    # Forward pass through the loaded model
    time_dict = {}
    model_graphdef, model_state = nnx.split(loaded_model)
    # output = loaded_model(input)
    # Run nnx.scan using batch_size=1. Use dataset size as number of scan steps.
  
    for num_sample in range(1, 32000, 100):
        inputs_subset = inputs[:num_sample, :, :, :]
        inputs_subset = inputs_subset.reshape(
            num_sample, 1, inputs.shape[1], inputs.shape[2], inputs.shape[3]
        )
        start_time = time.perf_counter()
        output = nnx.scan(
            test_model_forward,
            in_axes=(0, nnx.Carry),
            out_axes=(nnx.Carry),
        )(inputs_subset, model_state)
        output = jax.block_until_ready(output)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_dict[num_sample] = elapsed_time
        print(
            f"Time taken for scanning {num_sample} steps: {elapsed_time:.6f} seconds"
        )

    # Write time_dict to metrics.csv
    metrics_csv_path = "/home/dell/project-7thsem/training_dir/metrics.csv"
    with open(metrics_csv_path, mode="a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for samples, elapsed_time in time_dict.items():
            csv_writer.writerow(["samples", samples, "elapsed_time", elapsed_time])

    # eps_real = ER.real
    # eps_imag = ER.imag

    # eps_complex = jnp.stack([eps_real, eps_imag], axis=-1)
    # print("stacked eps_complex shape:", eps_complex.shape)

    # plot_complex_fields(output[0], target[0], eps_complex[0], save_path="test.png")
    # ET_target = target[0, ..., 0] + 1j * target[0, ..., 1]
    # ET_predicted = output[0, ..., 0] + 1j * output[0, ..., 1]
    # relative_error = jnp.linalg.norm(ET_predicted - ET_target) / jnp.linalg.norm(
    #     ET_target
    # )
    # print(f"Relative error between output and target: {relative_error:.6f}")
