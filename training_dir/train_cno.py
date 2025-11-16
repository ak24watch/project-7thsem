import optax
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from dataLoader.make_data import prepare_dataloader
from model_dir.Cno_2d_model import CNO_2D
import plotly.graph_objects as go

from tqdm import tqdm

import finitediffx as fdx


@nnx.jit
def preprocess_batch(ER, EI, config, ET):
    K_ = config["K0"] * jnp.sqrt(ER)
    input_batch = jnp.stack(
        [ER.real, ER.imag, EI.real, EI.imag, K_.real, K_.imag], axis=-1
    )
    output_batch = jnp.stack([ET.real, ET.imag], axis=-1)
    return input_batch, output_batch


def compute_loss(cno_model, targets, inputs, config):
    predicted = cno_model(inputs)
    ET_target_real = targets[..., 0]
    ET_target_imag = targets[..., 1]
    ET_output_real = predicted[..., 0]
    ET_output_imag = predicted[..., 1]
    loss_real = jnp.mean((ET_output_real.ravel() - ET_target_real.ravel()) ** 2)
    loss_imag = jnp.mean((ET_output_imag.ravel() - ET_target_imag.ravel()) ** 2)
    data_loss = loss_real + loss_imag

    # physics loss
    EI_real = inputs[..., 2]
    EI_imag = inputs[..., 3]

    ER = inputs[..., 0] + 1j * inputs[..., 1]
    ES_real = ET_output_real - EI_real
    ES_imag = ET_output_imag - EI_imag

    laplcian_ES_real = fdx.laplacian(
        ES_real, step_size=(config["dx"], config["dy"]), accuracy=config["fdx_accuracy"]
    )
    laplcian_ES_imag = fdx.laplacian(
        ES_imag, step_size=(config["dx"], config["dy"]), accuracy=config["fdx_accuracy"]
    )

    laplacian_ES = laplcian_ES_real + 1j * laplcian_ES_imag

    ES = ES_real + 1j * ES_imag
    ET = ET_output_real + 1j * ET_output_imag

    physics_residual = laplacian_ES + config.k0**2 * ES + config.k0**2 * (ER - 1) * ET
    physics_loss = jnp.mean(jnp.abs(physics_residual.ravel()) ** 2)

    total_loss = data_loss + config["physics_loss_weight"] * physics_loss

    return total_loss


@nnx.jit
def update_cno(cno_model, inputs, targets, config, optimizer):
    loss, grads = nnx.value_and_grad(compute_loss)(cno_model, targets, inputs, config)
    optimizer.update(cno_model, grads)
    return loss


def train_cno_model(config):
    # prepare data
    data_loader = prepare_dataloader(config["data_folder"], config["batch_size"])
    EI = jnp.load("dataset/EI.npy")


    # create model
    rngs = nnx.Rngs(config["random_seed"])
    model = CNO_2D(
        encoder_in_channels=config["encoder_in_channels"],
        encoder_out_channels=config["encoder_out_channels"],
        encoder_in_size=config["encoder_in_size"],
        encoder_out_size=config["encoder_out_size"],
        decoder_in_channels=config["decoder_in_channels"],
        decoder_out_channels=config["decoder_out_channels"],
        decoder_in_size=config["decoder_in_size"],
        decoder_out_size=config["decoder_out_size"],
        lp_latent_dim=config["lp_latent_dim"],
        lp_in_channels=config["lp_in_channels"],
        lp_out_channels=config["lp_out_channels"],
        lp_in_size=config["lp_in_size"],
        lp_out_size=config["lp_out_size"],
        activation=config["activation"],
        use_bn=config["use_bn"],
        num_residual_blocks=config["num_residual_blocks"],
        rngs=rngs,
    )

    # create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(config["learning_rate"]), wrt=nnx.Param)

    # training loop
    for epoch in tqdm(range(config["num_epochs"]), desc="Training Epochs"):
        epoch_loss = 0
        batch_count = 0

        for ER, ET in data_loader():
            inputs, targets = preprocess_batch(ER, EI, config, ET)
            loss = update_cno(model, inputs, targets, config, optimizer)
            epoch_loss += float(loss)
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        tqdm.write(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        # Optionally, update tqdm bar with loss
        # tqdm.set_postfix({"loss": avg_loss})

    

if __name__ == "__main__":
    # Example configuration
    config = {
        "data_folder": "dataset/",
        "batch_size": 16,
        "K0": 2 * jnp.pi / 0.3,
        "dx": 0.01,
        "dy": 0.01,
        "fdx_accuracy": 2,
        "physics_loss_weight": 0.3,
        "encoder_in_channels": [34, 64, 128],
        "encoder_out_channels": [64, 128, 256],
        "encoder_in_size": [64, 32, 16],
        "encoder_out_size": [32, 16, 8],
        "decoder_in_channels": [256, 64, 16, 8],
        "decoder_out_channels": [64, 16, 8, 2],
        "decoder_in_size": [8, 16, 32, 64],
        "decoder_out_size": [16, 32, 64, 128],
        "lp_latent_dim": [34, 2],
        "lp_in_channels": [34, 2],
        "lp_out_channels": [34, 2],
        "lp_in_size": [64, 128],
        "lp_out_size": [64, 128],
        "activation": nnx.leaky_relu,
        "use_bn": True,
        "num_residual_blocks": 4,
        "learning_rate": 1e-3,
        "num_epochs": 100,
        "random_seed": 42,
    }

    train_cno_model(config)
