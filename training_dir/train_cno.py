import optax
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from dataLoader.make_data import prepare_dataloader
from model_dir.Cno_2d_model import CNO_2D, ActivationFilter
from functools import partial
from tqdm import tqdm
import flax.serialization
import msgpack
import os


import finitediffx as fdx


def preprocess_batch(ER, EI, k0, ET):
  
    K_ = k0* jnp.sqrt(ER)


    input_batch = jnp.stack(
        [ER.real, ER.imag, EI.real, EI.imag, K_.real, K_.imag], axis=-1
    )
    output_batch = jnp.stack([ET.real, ET.imag], axis=-1)
    return input_batch, output_batch


def compute_loss(cno_model, targets, inputs, dx, dy, fdx_accuracy, k0):
    predicted = cno_model(inputs)
    ET_target_real = targets[..., 0]
    ET_target_imag = targets[..., 1]
    ET_output_real = predicted[..., 0]
    ET_output_imag = predicted[..., 1]
    loss_real = jnp.mean((ET_output_real.ravel() - ET_target_real.ravel()) ** 2)
    loss_imag = jnp.mean((ET_output_imag.ravel() - ET_target_imag.ravel()) ** 2)
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

    # physics_residual = laplacian_ES + k0**2 * ES + k0**2 * (ER - 1) * ET
    # physics_loss = jnp.mean(jnp.abs(physics_residual.ravel()) ** 2)

    # total_loss = data_loss + config["physics_loss_weight"] * physics_loss

    # return total_loss
    return data_loss


@nnx.jit(static_argnames=["dx", "dy", "fdx_accuracy", "k0"])
def update_cno(cno_model, inputs, targets, dx, dy, fdx_accuracy, k0, optimizer):
    loss, grads = nnx.value_and_grad(compute_loss)(
        cno_model,
        targets,
        inputs,
        dx=dx,
        dy=dy,
        fdx_accuracy=fdx_accuracy,
        k0=k0,
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
    Serialize and save an NNX model state to disk using flax.serialization.
    """
    state = nnx.state(model)
    payload = flax.serialization.to_bytes(state)
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(payload)


def train_cno_model(config):
    # prepare data
    data_loader = prepare_dataloader(config["data_folder"], config["batch_size"])
    EI = jnp.load("EI.npy")
    EI = EI.reshape(32, 32)
     # check EI for NaN or infinite values
    if jnp.any(jnp.isnan(EI)) or jnp.any(jnp.isinf(EI)):
        print("EI contains NaN or infinite values.")
    if jnp.any(jnp.isnan(ET)) or jnp.any(jnp.isinf(ET)):
        print("ET contains NaN or infinite values.")

    # create model
    rngs = nnx.Rngs(config["random_seed"])
    model = build_cno_model(config, rngs)

    # create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(config["learning_rate"]), wrt=nnx.Param)

    # training loop
    with tqdm(total=config["num_epochs"], desc="Training Epochs") as pbar:
        for epoch in range(config["num_epochs"]):
            epoch_loss = 0
            batch_count = 0

            for ER, ET in data_loader():
                # check ER for NaN or infinite values
                if jnp.any(jnp.isnan(ER)) or jnp.any(jnp.isinf(ER)):
                    print("ER contains NaN or infinite values.")

                # check ET for NaN or infinite values
                if jnp.any(jnp.isnan(ET)) or jnp.any(jnp.isinf(ET)):
                    print("ET contains NaN or infinite values.")

                inputs, targets = nnx.vmap(preprocess_batch, in_axes=(0, None, None, 0))(
                    ER, EI, config["K0"], ET
                )
                loss = update_cno(
                    model,
                    inputs,
                    targets,
                    config["dx"],
                    config["dy"],
                    config["fdx_accuracy"],
                    config["K0"],
                    optimizer,
                )
                epoch_loss += float(loss)
                batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            pbar.set_postfix({"loss": avg_loss})
            pbar.update(1)
        # Optionally, update tqdm bar with loss
        # tqdm.set_postfix({"loss": avg_loss})
    # save after training (optional)
    save_path = config.get("save_path")
    if save_path:
        save_nnx_model(model, save_path)
    return model


if __name__ == "__main__":
    # Example configuration
    frequency = 400e6  # 400 MHz
    c0 = 3e8  # Speed of light in m/s
    wavelength = c0 / frequency  # Wavelength in meters
    print("wavelength is", wavelength)
    print("delta is ", wavelength / 20)
    config = {
        "data_folder": "dataset/",
        "batch_size": 32,
        "K0": 2 * jnp.pi / wavelength,
        "dx": wavelength / 20,
        "dy": wavelength / 20,
        "fdx_accuracy": 10,
        "physics_loss_weight": 0.3,
        "encoder_in_channels": [32, 64, 128],
        "encoder_out_channels": [64, 128, 256],
        "encoder_in_size": [32, 16, 8],
        "encoder_out_size": [16, 8, 4],
        "decoder_in_channels": [256, 128, 64],
        "decoder_out_channels": [128, 64, 32],
        "decoder_in_size": [8, 16, 32],
        "decoder_out_size": [16, 32, 64],
        "lp_latent_channels": [
            16,
            16,
        ],  # comes in b/w lp_in_channels and lp_out_channels
        "lp_in_channels": [6, 32],  # may be 8 in_channels  for lift if added something
        "lp_out_channels": [32, 2],
        "lp_in_size": [32, 32],  # change lp
        "lp_out_size": [32, 32],  # change lp_out_size of lift to 176  afterwards
        "activation": nnx.leaky_relu,
        "use_bn": True,
        "num_residual_blocks": 4,
        "learning_rate": 1e-3,
        "num_epochs": 100,
        "random_seed": 42,
        "kernel_size": 3,
        # new: where to save the trained model
        "save_path": "checkpoints/cno_2d.msgpack",
    }
   

    model = train_cno_model(config)
    # Example: load the model later (or immediately) from disk
    # from check_model import load_cno_model
    # loaded_model = load_cno_model(config, config["save_path"])

