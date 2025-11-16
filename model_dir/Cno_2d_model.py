import flax.nnx as nnx
from model_dir.model_main_blocks import Resnet, EncoderBlock, DecoderBlock
from model_dir.model_blocks import LiftProjectBlock
import jax.numpy as jnp
from collections.abc import Sequence
from model_dir.activation_filter import ActivationFilter


class CNO_2D(nnx.Module):
    # Mark python containers that may hold Arrays as static data so Flax's
    # pytree checker doesn't treat them as mutable pytree fields.
    # Treat these attributes as static data for the Module pytree.
    encoder_list: nnx.Data[list]
    decoder_list: nnx.Data[list]

    def __init__(
        self,
        encoder_in_channels: Sequence[int],
        encoder_out_channels: Sequence[int],
        encoder_in_size: Sequence[int],
        encoder_out_size: Sequence[int],
        decoder_in_channels: Sequence[int],
        decoder_out_channels: Sequence[int],
        decoder_in_size: Sequence[int],
        decoder_out_size: Sequence[int],
        lp_latent_dim: Sequence[int],
        lp_in_channels: Sequence[int],
        lp_out_channels: Sequence[int],
        lp_in_size: Sequence[int],
        lp_out_size: Sequence[int],
        activation: nnx.leaky_relu, # <-- here is the mistake ,as this a class not function
        use_bn=True,
        num_residual_blocks=4,
        rngs: nnx.Rngs = None,
        kernel_size=3,
    ):
        self.encoder_list = []
        self.decoder_list = []

        for i in range(len(encoder_in_channels)):
            self.encoder_list.append(
                EncoderBlock(
                    in_channels=encoder_in_channels[i],
                    out_channels=encoder_out_channels[i],
                    in_size=encoder_in_size[i],
                    out_size=encoder_out_size[i],
                    use_bn=use_bn,
                    rngs=rngs,
                    activation=activation,
                    kernel_size=kernel_size,
                    num_residual_blocks=num_residual_blocks,
                )
            )
        # print("rngs count before decoder:", rngs.count)
        for i in range(len(decoder_in_channels)):
            self.decoder_list.append(
                DecoderBlock(
                    in_channels=decoder_in_channels[i],
                    out_channels=decoder_out_channels[i],
                    in_size=decoder_in_size[i],
                    out_size=decoder_out_size[i],
                    use_bn=use_bn,
                    rngs=rngs,
                    activation=activation,
                    kernel_size=kernel_size,
                    num_residual_blocks=num_residual_blocks,
                )
            )

        self.encoders = nnx.Sequential(*self.encoder_list)
        self.decoders = nnx.Sequential(*self.decoder_list)

        self.resnet_bolttleneck = Resnet(
            channels=encoder_out_channels[-1],
            size=encoder_out_size[-1],
            num_blocks=num_residual_blocks,
            us_bn=use_bn,
            rngs=rngs,
            activation=activation,
            kernel_size=kernel_size,
        )

        self.lift = LiftProjectBlock(
            in_channels=lp_in_channels[0],
            latent_dim=lp_latent_dim[0],
            out_channels=lp_out_channels[0],
            in_size=lp_in_size[0],
            out_size=lp_out_channels[0],
            rngs=rngs,
            kernel_size=kernel_size,
            activation=activation,
        )

        self.project = LiftProjectBlock(
            in_channels=lp_out_channels[-1],
            latent_dim=lp_latent_dim[-1],
            out_channels=lp_out_channels[-1],
            in_size=lp_in_size[-1],
            out_size=lp_out_size[-1],
            rngs=rngs,
            kernel_size=kernel_size,
            activation=activation,
        )

   
    def __call__(self, x):
        # lift
        x = self.lift(x)

        # encoder

        x = self.encoders(x)

        # resnet bottleneck
        x = self.resnet_bolttleneck(x)

        # decoder
        x = self.decoders(x)

        # project
        x = self.project(x)

        return x

    # test CnO_2D


if __name__ == "__main__":
    # Example usage
    # lift/project parameters
    lp_latent_dim = [34, 2]  # latent dimension for lift and project blocks
    lp_in_channels = [34, 2]
    lp_out_channels = [34, 2]  # output channels for lift and project blocks
    lp_in_size = [64, 128]  # input size for lift and project blocks
    lp_out_size = [64, 128]  # output size for lift and project

    # encder parameters
    encoder_in_channels = [34, 64, 128]
    encoder_out_channels = [64, 128, 256]
    encoder_in_size = [64, 32, 16]
    encoder_out_size = [32, 16, 8]

    # decoder parameters
    decoder_in_channels = [256, 64, 16, 8]
    decoder_out_channels = [64, 16, 8, 2]
    decoder_in_size = [8, 16, 32, 64]
    decoder_out_size = [16, 32, 64, 128]

    # rngs for random number generation
    rngs = nnx.Rngs(45)
    import time

    start_time = time.time()
    # create the model
    model = CNO_2D(
        encoder_in_channels=encoder_in_channels,
        encoder_out_channels=encoder_out_channels,
        encoder_in_size=encoder_in_size,
        encoder_out_size=encoder_out_size,
        decoder_in_channels=decoder_in_channels,
        decoder_out_channels=decoder_out_channels,
        decoder_in_size=decoder_in_size,
        decoder_out_size=decoder_out_size,
        lp_latent_dim=lp_latent_dim,
        lp_in_channels=lp_in_channels,
        lp_out_channels=lp_out_channels,
        lp_in_size=lp_in_size,
        lp_out_size=lp_out_size,
        activation=ActivationFilter,
        use_bn=True,
        num_residual_blocks=4,
        rngs=rngs,
    )
    end_time = time.time()
    print("Model created in:", end_time - start_time, "seconds")

    # print(model)
    # time it

    start_time = time.time()
    # Example input shape model in must always be (batch_size, height, width, channels)
    # as activation filter expects batch dimension
    out = model(jnp.ones((1, 64, 64, 34)))

    end_time = time.time()
    print("Output shape:", out.shape)  # Should be (1, 128, 128, 2)
    print("Time taken:", end_time - start_time, "seconds")
