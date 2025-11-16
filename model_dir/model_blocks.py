import jax
import flax.nnx as nnx


class CNOBlock(nnx.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation, # class not function
        rngs: nnx.Rngs = None,
        use_bn=True,
        kernel_size=3,
        in_size=None,
        out_size=None,
    ):
        pad = (kernel_size - 1) // 2
        
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            padding=pad,
            rngs=rngs,
            kernel_init=jax.nn.initializers.he_normal(),  # Use JAX's built-in He normal initializer
        )

        if use_bn:
            self.bn = nnx.BatchNorm(
                num_features=out_channels,
                rngs=rngs,
                use_fast_variance=False,
            )

        else:
            self.bn = jax.nn.identity

        self.activation = activation(in_size, out_size)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class LiftProjectBlock(nnx.Module):
    def __init__(
        self,
        in_channels,
        latent_dim,
        out_channels,
        activation,
        in_size=None,
        out_size=None,
        rngs=None,
        kernel_size=3,
    ):
        pad = (kernel_size - 1) // 2

        self._inter_cno_block = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            use_bn=True,
            activation=activation,
            rngs=rngs,
            in_size=in_size,
            out_size=out_size,
            kernel_size=kernel_size,
        )
        self.conv = nnx.Conv(
            in_features=latent_dim,
            out_features=out_channels,
            kernel_size=kernel_size,
            padding=pad,
            rngs=rngs,
            kernel_init=jax.nn.initializers.he_normal(),
        )

    def __call__(self, x):
        x = self._inter_cno_block(x)
        x = self.conv(x)
        return x


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        channels,
        activation,
        size=None,
        use_bn=True,
        rngs=None,
        kernel_size=3,
    ):
        pad = (kernel_size - 1) // 2

        self.conv1 = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=kernel_size,
            padding=pad,
            rngs=rngs,
            kernel_init=jax.nn.initializers.he_normal()
        )
        self.conv2 = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=kernel_size,
            padding=pad,
            rngs=rngs,
            kernel_init=jax.nn.initializers.he_normal()
        )
        if use_bn:
            self.bn1 = nnx.BatchNorm(
                num_features=channels, rngs=rngs, use_fast_variance=False
            )
            self.bn2 = nnx.BatchNorm(
                num_features=channels, rngs=rngs, use_fast_variance=False
            )
        else:
            self.bn1 = jax.nn.identity
            self.bn2 = jax.nn.identity

        self.activation = activation(size, size)  # size used in activation function

    def __call__(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual

        return x
