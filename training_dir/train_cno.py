import optax
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from dataLoader.make_data import prepare_dataloader
from model_dir.Cno_2d_model import CNO_2D
import plotly.graph_objects as go
import plotly.io as pio

import finitediffx as fdx


@jax.jit
def preprocess_batch(ER, EI, config, ET):
    K_ = config["K0"] * jnp.sqrt(ER)
    input_batch = jnp.stack([ER.real, ER.imag, EI.real, EI.imag, K_.real, K_.imag], axis=-1)
    output_batch = jnp.stack([ET.real, ET.imag], axis=-1)
    # return input_batch, output_batch


def compute_loss(outputs, targets, inputs, config):
    ET_target_real = targets[..., 0]
    ET_target_imag = targets[..., 1]
    ET_output_real = outputs[..., 0]
    ET_output_imag = outputs[..., 1]
    loss_real = jnp.mean((ET_output_real.ravel() - ET_target_real.ravel()) ** 2)
    loss_imag = jnp.mean((ET_output_imag.ravel() - ET_target_imag.ravel()) ** 2)
    data_loss = loss_real + loss_imag

    # physics loss
    EI_real = inputs[..., 2]
    EI_imag = inputs[..., 3]
    K_real = inputs[..., 4]
    K_imag = inputs[..., 5]
    ER_real = inputs[..., 0]
    ER_imag = inputs[..., 1]

    


