import optax
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from dataLoader.make_data import prepare_dataloader
from model_dir.Cno_2d_model import CNO_2D, ActivationFilter
from functools import partial
from tqdm import tqdm
import orbax.checkpoint as ocp
import os


# import finitediffx as fdx  # not used currently; keep commented for potential physics loss


def preprocess_batch(ER, EI, k0, ET):
    K_ = k0 * jnp.sqrt(ER)

    input_batch = jnp.stack(
        [ER.real, ER.imag, EI.real, EI.imag, K_.real, K_.imag], axis=-1
    )
    output_batch = jnp.stack([ET.real, ET.imag], axis=-1)
    return input_batch, output_batch


def compute_loss(cno_model, targets, inputs):
    predicted = cno_model(inputs)
    ET_target_real = targets[..., 0]
    ET_target_imag = targets[..., 1]
    ET_output_real = predicted[..., 0]
    ET_output_imag = predicted[..., 1]
    loss_real = jnp.mean((ET_output_real - ET_target_real) ** 2)
    loss_imag = jnp.mean((ET_output_imag - ET_target_imag) ** 2)
    data_loss = loss_real + loss_imag

    # # physics loss
    # EI_real = inputs[..., 2]
    # EI_imag = inputs[..., 3]

    # ER = inputs[..., 0] + 1j * inputs[..., 1]
    # ES_real = ET_output_real - EI_real
    # ES_imag = ET_output_imag - EI_imag
    # lap_fn = partial(fdx.laplacian, step_size=(dx, dy), accuracy=fdx_accuracy)

    # laplacian_ES_real = jax.vmap(lap_fn, in_axes=0)(ES_real)
    # laplacian_ES_imag = jax.vmap(lap_fn, in_axes=0)(ES_imag)

    # laplacian_ES = laplacian_ES_real + 1j * laplacian_ES_imag

    # ES = ES_real + 1j * ES_imag
    # ET = ET_output_real + 1j * ET_output_imag

    # physics_residual = laplacian_ES + k0**2 * ES + k0**2 * ER * ET
    # physics_loss = jnp.mean(jnp.abs(physics_residual) ** 2)

    # total_loss = data_loss + config["physics_loss_weight"] * physics_loss

    # return total_loss
    return data_loss


def relative_l2_error(predicted, targets, eps: float = 1e-8):
    """
    Compute mean relative L2 error over a batch for 2-channel (real, imag) fields.

    predicted, targets: shape (B, H, W, 2)
    returns scalar jnp.float32
    """
    # Flatten spatial and channel dims per sample
    num = jnp.sqrt(jnp.sum((predicted - targets) ** 2, axis=(1, 2, 3)))
    den = jnp.sqrt(jnp.sum((targets) ** 2, axis=(1, 2, 3)))
    rel = num / (den + eps)
    return jnp.mean(rel)


@nnx.jit
def update_cno(cno_model, inputs, targets, optimizer):
    loss, grads = nnx.value_and_grad(compute_loss)(
        cno_model,
        targets,
        inputs,
    )
    optimizer.update(cno_model, grads)
    return loss


def build_cno_model(config, rngs):
    """
    Create the CNO_2D model from config and rngs. Reused for training and loading.
    """
    return CNO_2D(
        encoder_in_channels=config["encoder_in_channels"],
        encoder_out_channels=config["encoder_out_channels"],
        encoder_in_size=config["encoder_in_size"],
        encoder_out_size=config["encoder_out_size"],
        decoder_in_channels=config["decoder_in_channels"],
        decoder_out_channels=config["decoder_out_channels"],
        decoder_in_size=config["decoder_in_size"],
        decoder_out_size=config["decoder_out_size"],
        lp_latent_channels=config["lp_latent_channels"],
        lp_in_channels=config["lp_in_channels"],
        lp_out_channels=config["lp_out_channels"],
        lp_in_size=config["lp_in_size"],
        lp_out_size=config["lp_out_size"],
        activation=partial(ActivationFilter, activation=config["activation"]),
        use_bn=config["use_bn"],
        num_residual_blocks=config["num_residual_blocks"],
        rngs=rngs,
        kernel_size=config["kernel_size"],
    )


def save_nnx_model(model, save_path: str):
    """
    Serialize and save an NNX model state to disk using orbax.
    """
    # Collect model state (parameters, batchnorm stats, etc.) as a PyTree of jax.Array leaves.
    state = nnx.state(model)
    # Convert any jax.Array leaves to host numpy arrays so no sharding reconstruction is needed on restore.
    state_host = jax.tree_util.tree_map(
        lambda x: jax.device_get(x) if isinstance(x, (jax.Array, jnp.ndarray)) else x,
        state,
    )
    checkpointer = ocp.PyTreeCheckpointer()
    abs_save_path = os.path.abspath(save_path)
    # Save host materialized state; loader can restore without sharding inference warning.
    checkpointer.save(abs_save_path, state_host, force=True)


def train_cno_model(config, metrics=None):
    # prepare data: train/val loaders
    train_loader, val_loader, sizes = prepare_dataloader(
        config["data_folder"],
        config["batch_size"],
        val_ratio=config["val_ratio"],
        seed=config["data_seed"],
    )
    EI = jnp.load("EI.npy")
    EI = EI.reshape(32, 32)
    # check EI for NaN or infinite values
    if jnp.any(jnp.isnan(EI)) or jnp.any(jnp.isinf(EI)):
        print("EI contains NaN or infinite values.")

    # create model
    rngs = nnx.Rngs(config["random_seed"])
    model = build_cno_model(config, rngs)

    # create optimizer
    optimizer = nnx.Optimizer(
        model, optax.adamw(config["learning_rate"]), wrt=nnx.Param
    )

    # training loop
    with tqdm(total=config["num_epochs"], desc="Training Epochs") as pbar:
        for epoch in range(config["num_epochs"]):
            train_epoch_loss = 0.0
            train_epoch_rel = 0.0
            train_batches = 0

            # Training phase
            for ER, ET in train_loader():
                # # check ER for NaN or infinite values
                # if jnp.any(jnp.isnan(ER)) or jnp.any(jnp.isinf(ER)):
                #     print("ER contains NaN or infinite values.")

                # # check ET for NaN or infinite values
                # if jnp.any(jnp.isnan(ET)) or jnp.any(jnp.isinf(ET)):
                #     print("ET contains NaN or infinite values.")

                inputs, targets = nnx.vmap(
                    preprocess_batch, in_axes=(0, None, None, 0)
                )(ER, EI, config["K0"], ET)
                model.train()
                loss = update_cno(
                    model,
                    inputs,
                    targets,
                    optimizer,
                )
                train_epoch_loss += float(loss)
                # compute train relative error (no grad)
                preds = model(inputs)
                batch_rel = float(relative_l2_error(preds, targets))
                train_epoch_rel += batch_rel
                train_batches += 1

            avg_train_loss = (
                train_epoch_loss / train_batches if train_batches > 0 else 0.0
            )
            avg_train_rel = (
                train_epoch_rel / train_batches if train_batches > 0 else 0.0
            )

            # Validation phase
            val_epoch_loss = 0.0
            val_epoch_rel = 0.0
            val_batches = 0
            for ER, ET in val_loader():
                inputs, targets = nnx.vmap(
                    preprocess_batch, in_axes=(0, None, None, 0)
                )(ER, EI, config["K0"], ET)
                model.eval()
                v_loss = float(compute_loss(model, targets, inputs))
                preds = model(inputs)
                v_rel = float(relative_l2_error(preds, targets))
                val_epoch_loss += v_loss
                val_epoch_rel += v_rel
                val_batches += 1

            avg_val_loss = val_epoch_loss / val_batches if val_batches > 0 else 0.0
            avg_val_rel = val_epoch_rel / val_batches if val_batches > 0 else 0.0

            # Track metrics if provided
            if metrics is not None:
                metrics.setdefault("avg_train_loss", []).append(avg_train_loss)
                metrics.setdefault("avg_val_loss", []).append(avg_val_loss)
                metrics.setdefault("avg_train_relative_error", []).append(avg_train_rel)
                metrics.setdefault("avg_val_relative_error", []).append(avg_val_rel)

            pbar.set_postfix(
                {
                    "train_loss": f"{avg_train_loss:.5f}",
                    "val_loss": f"{avg_val_loss:.5f}",
                }
            )
            pbar.update(1)
        # Optionally, update tqdm bar with loss
        # tqdm.set_postfix({"loss": avg_loss})
    # save after training (optional)
    save_path = config.get("save_path")
    if save_path:
        save_nnx_model(model, save_path)
    return model


def get_config():
    frequency = 400e6  # 400 MHz
    c0 = 3e8  # Speed of light in m/s
    wavelength = c0 / frequency
    return {
        "data_folder": "dataset/",
        "batch_size": 80,
        "encoder_in_channels": [32, 64, 128],
        "encoder_out_channels": [64, 128, 256],
        "encoder_in_size": [32, 16, 8],
        "encoder_out_size": [16, 8, 4],
        "decoder_in_channels": [256, 128, 64],
        "decoder_out_channels": [128, 64, 32],
        "decoder_in_size": [8, 16, 32],
        "decoder_out_size": [16, 32, 64],
        "lp_latent_channels": [
            32,
            32,
        ],
        "lp_in_channels": [6, 32],
        "lp_out_channels": [32, 2],
        "lp_in_size": [32, 32],
        "lp_out_size": [32, 32],
        "activation": nnx.swish,
        "use_bn": True,
        "num_residual_blocks": 8,
        "learning_rate": 1e-3,
        "num_epochs": 100,
        "random_seed": 42,
        "data_seed": 432,
        "val_ratio": 0.2,
        "kernel_size": 3,
        "save_path": "checkpoints/cno_2d/",
        "metrics_csv": "training_dir/metrics.csv",
        "wavelength": wavelength,
        "K0": 2 * jnp.pi / wavelength,
    }


if __name__ == "__main__":
    config = get_config()
    print("wavelength is", config["wavelength"])
    metrics = {
        "avg_train_loss": [],
        "avg_val_loss": [],
        "avg_train_relative_error": [],
        "avg_val_relative_error": [],
    }

    model = train_cno_model(config, metrics=metrics)

    # Optionally save metrics to CSV for external tracking
    try:
        import csv

        csv_path = config.get("metrics_csv")
        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "epoch",
                        "avg_train_loss",
                        "avg_val_loss",
                        "avg_train_relative_error",
                        "avg_val_relative_error",
                    ]
                )
                for i in range(len(metrics["avg_train_loss"])):
                    writer.writerow(
                        [
                            i + 1,
                            metrics["avg_train_loss"][i],
                            metrics["avg_val_loss"][i],
                            metrics["avg_train_relative_error"][i],
                            metrics["avg_val_relative_error"][i],
                        ]
                    )
            print(f"Saved metrics to {csv_path}")
    except Exception as e:
        print(f"Could not write metrics CSV: {e}")
    # Example: load the model later (or immediately) from disk
    # from check_model import load_cno_model
    # loaded_model = load_cno_model(config, config["save_path"])
