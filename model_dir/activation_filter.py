import jax

import flax.nnx as nnx


class ActivationFilter(nnx.Module):
    def __init__(self, in_size, out_size, activation=nnx.leaky_relu):
        self._in_size = in_size
        self._out_size = out_size
        # comsider mish instead of leaky relu
        self._activation = activation

    def __call__(self, x):
        # bicubic interpolation in jax
        x = jax.image.resize(
            x,
            shape=(x.shape[0], 2 * self._in_size, 2 * self._in_size, x.shape[-1]),
            method="bilinear",
            antialias=False,
        )

        #
        x = self._activation(x)

        x = jax.image.resize(
            x,
            shape=(x.shape[0], self._out_size, self._out_size, x.shape[-1]),
            method="bilinear",
            antialias=False,
        )

        return x


# x = jnp.ones((1, 32, 32, 3))  # Example input
# in_size = 32
# out_size = 64
# resized_x = ActivationFilter(in_size, out_size)(x)
# print("Resized shape:", resized_x.shape)  # Should be (1, 64, 64, 3)
if __name__ == "__main__":
    # Example usage
    import jax.numpy as jnp
    in_size = 64
    out_size = 88
    x = jnp.ones((2, in_size, in_size, 34))  # Example input
    resized_x = ActivationFilter(in_size, out_size)(x)
    print("Resized shape:", resized_x.shape)  # Should be (2, 88, 88, 34)