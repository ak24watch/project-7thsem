import optax

import jax.numpy as jnp
import flax.nnx as nnx
from dataLoader.make_data import prepare_dataloader
from model_dir.Cno_2d_model import CNO_2D
import equinox as eqx

from tqdm import tqdm

import finitediffx as fdx


def preprocess_batch(ER, EI, k0, ET):
    print("EI shape is", EI.shape)
    print("ER shape is", ER.shape)
    print("ET shape is", ET.shape)
    K_ = k0 * jnp.sqrt(ER)
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

    # physics loss
    EI_real = inputs[..., 2]
    EI_imag = inputs[..., 3]

    ER = inputs[..., 0] + 1j * inputs[..., 1]
    ES_real = ET_output_real - EI_real
    ES_imag = ET_output_imag - EI_imag

    laplcian_ES_real = fdx.laplacian(ES_real, step_size=(dx, dy), accuracy=fdx_accuracy)
    laplcian_ES_imag = fdx.laplacian(ES_imag, step_size=(dx, dy), accuracy=fdx_accuracy)

    laplacian_ES = laplcian_ES_real + 1j * laplcian_ES_imag

    ES = ES_real + 1j * ES_imag
    ET = ET_output_real + 1j * ET_output_imag

    physics_residual = laplacian_ES + k0**2 * ES + k0**2 * (ER - 1) * ET
    physics_loss = jnp.mean(jnp.abs(physics_residual.ravel()) ** 2)

    total_loss = data_loss + config["physics_loss_weight"] * physics_loss

    return total_loss


@nnx.jit
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


def train_cno_model(config):
    # prepare data
    data_loader = prepare_dataloader(config["data_folder"], config["batch_size"])
    EI = jnp.load("e_forward.npy")
    EI = EI.reshape(88, 88)

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
        tqdm.write(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        # Optionally, update tqdm bar with loss
        # tqdm.set_postfix({"loss": avg_loss})


if __name__ == "__main__":
    # Example configuration
    config = {
        "data_folder": "dataset/",
        "batch_size": 88,
        "K0": 2 * jnp.pi / 0.3,
        "dx": 0.0375,
        "dy": 0.0375,
        "fdx_accuracy": 2,
        "physics_loss_weight": 0.3,
        "encoder_in_channels": [32, 64, 128],
        "encoder_out_channels": [64, 128, 256],
        "encoder_in_size": [88, 44, 22],
        "encoder_out_size": [44, 22, 11],
        "decoder_in_channels": [256, 128, 64, 32],
        "decoder_out_channels": [128, 64, 32, 2],
        "decoder_in_size": [11, 22, 44, 88],
        "decoder_out_size": [22, 44, 88, 88],
        "lp_latent_dim": [256, 256],
        "lp_in_channels": [6, 2],
        "lp_out_channels": [32, 2],
        "lp_in_size": [88, 88],
        "lp_out_size": [88, 88],
        "activation": nnx.leaky_relu,
        "use_bn": True,
        "num_residual_blocks": 4,
        "learning_rate": 1e-3,
        "num_epochs": 100,
        "random_seed": 42,
    }

    train_cno_model(config)
