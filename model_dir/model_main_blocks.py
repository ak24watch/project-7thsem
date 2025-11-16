import flax.nnx as nnx
from model_dir.model_blocks import ResidualBlock, CNOBlock



class Resnet(nnx.Module):
    def __init__(
        self,
        channels,
        size,
        us_bn,
        rngs,
        activation,
        num_blocks=4,
        kernel_size=3,
    ):
        @nnx.split_rngs(splits=num_blocks)
        @nnx.vmap(axis_size=num_blocks)
        def create_resnet_block(rngs):
            return ResidualBlock(
                channels=channels,
                size=size,
                use_bn=us_bn,
                rngs=rngs,
                kernel_size=kernel_size,
                activation=activation, # class not function
            )

        self.resnet = create_resnet_block(rngs)

        self.num_blocks = num_blocks

    def __call__(self, x):
        # @nnx.split_rngs(splits=self.num_blocks)  # only use if block contains droput
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(x, block):
            x = block(x)
            return x

        return forward(x, self.resnet)


class EncoderBlock(nnx.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in_size,
        out_size,
        use_bn,
        rngs,
        activation,
        kernel_size=3,
        num_residual_blocks=4,
    ):
        self.down_block = CNOBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            in_size=in_size,
            out_size=out_size,
            use_bn=use_bn,
            rngs=rngs,
            activation=activation, # class not function
            kernel_size=kernel_size,
        )

        # self.resnet = Resnet(
        #     channels=out_channels,
        #     size=out_size,
        #     num_blocks=num_residual_blocks,
        #     us_bn=use_bn,
        #     rngs=rngs,
        #     activation=activation,
        #     kernel_size=kernel_size,
        # )

        self.invariant_block = CNOBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            in_size=out_size,
            out_size=out_size,
            use_bn=use_bn,
            rngs=rngs,
            activation=activation, # class not function
            kernel_size=kernel_size,
        )

    def __call__(self, x):
        x = self.down_block(x)
        # x = self.resnet(x)
        x = self.invariant_block(x)
        return x


class DecoderBlock(nnx.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in_size,
        out_size,
        use_bn,
        rngs,
        activation,
        kernel_size=3,
        num_residual_blocks=4,
    ):
        self.up_block = CNOBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            in_size=in_size,
            out_size=out_size,
            use_bn=use_bn,
            rngs=rngs,
            activation=activation, # class not function
            kernel_size=kernel_size,
        )

        # self.resnet = Resnet(
        #     channels=out_channels,
        #     size=out_size,
        #     num_blocks=num_residual_blocks,
        #     us_bn=use_bn,
        #     rngs=rngs,
        #     activation=activation,
        #     kernel_size=kernel_size,
        # )

        self.invariant_block = CNOBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            in_size=out_size,
            out_size=out_size,
            use_bn=use_bn,
            rngs=rngs,
            activation=activation, # class not function
            kernel_size=kernel_size,
        )

    def __call__(self, x):
        x = self.up_block(x)
        # x = self.resnet(x)
        x = self.invariant_block(x)
        return x
