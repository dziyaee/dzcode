import numpy as np
import torch
import torch.nn as nn
from dzlib.utils.helper import info, modules, children, forwardpass, params, shapes
from collections import namedtuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def convblock(in_channels, out_channels, conv_params, batchnorm, activation):
    activations = nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leakyrelu', nn.LeakyReLU(0.2)],
        ['sigmoid', nn.Sigmoid()]
        ])

    block = []
    block.append(nn.Conv2d(in_channels, out_channels, *conv_params))

    if batchnorm is True:
        block.append(nn.BatchNorm2d(out_channels))

    if is not None:
        assert activation in activations
        block.append(activations[activation])

    return block


def concat(main, skip):
    main_height, main_width = main.shape[2], main.shape[3]
    skip_height, skip_width = skip.shape[2], skip.shape[3]

    if (main_height == skip_height and main_width == skip_width):
        main = torch.cat((main, skip), dim=1)

    else:
        new_height, new_width = min(main_height, skip_height), min(main_width, skip_width)

        main_diff_height = (main_height - new_height) // 2
        main_diff_width = (main_width - new_width) // 2
        main = main[:, :, main_diff_height: main_diff_height + new_height, main_diff_width: main_diff_width + new_width]

        skip_diff_height = (skip_height - new_height) // 2
        skip_diff_width = (skip_width - new_width) // 2
        skip = skip[:, :, skip_diff_height: skip_diff_height + new_height, skip_diff_width: skip_diff_width + new_width]

        main = torch.cat((main, skip), dim=1)

    return main


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, up_convs, batchnorm, activation, upsampling):
        super().__init__()
        # 1
        self.upsample = None
        if upsampling is not None:
            self.upsample = nn.Upsample(*upsampling)
        # 2
        self.skip_function = None
        if skip_channels != 0:
            self.skip_function = concat
        # add skip_channels to in_channels to account for concatenation function. If skip = 0, in_channels won't change, no need to change formula
        in_channels += skip_channels
        # 3
        batchnorm = nn.BatchNorm2d(in_channels)
        main_branch1 = convblock(in_channels,  out_channels, up_convs[0], batchnorm, activation)
        main_branch2 = convblock(out_channels, out_channels, up_convs[1], batchnorm, activation)
        self.main_branch = nn.Sequential(batchnorm, *main_branch1, *main_branch2)

    def forward(self, x, sx):
        # 1
        if self.upsample is not None:
            x = self.upsample(x)
        # 2
        print("-" * 100)
        info(x, 'main after upsampling')
        info(sx, 'skip prior to concating')
        if self.skip_function is not None:
            x = self.skip_function(x, sx)
        info(x, 'main after concating')
        # 3
        x = self.main_branch(x)
        info(x, 'main_up output')
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, down_convs, skip_convs, batchnorm, activation):
        super().__init__()
        # 1
        self.skip_branch = None
        if skip_channels != 0:
            skip_branch = convblock(in_channels, skip_channels, skip_convs[0], batchnorm, activation)
            self.skip_branch = nn.Sequential(*skip_branch)
        # 2
        main_branch1 = convblock(in_channels,  out_channels, down_convs[0], batchnorm, activation)
        main_branch2 = convblock(out_channels, out_channels, down_convs[1], batchnorm, activation)
        self.main_branch = nn.Sequential(*main_branch1, *main_branch2)

    def forward(self, x):
        # 1
        sx = None
        if self.skip_branch is not None:
            sx = self.skip_branch(x)
        # 2
        x = self.main_branch(x)

        print("-" * 100)
        info(x, 'main_down output')
        info(sx, 'skip_down output')

        return (x, sx)


class UNet(nn.Module):
    ''''''

    def __init__(self, in_channels, out_channels, down_channels, skip_channels, up_channels, down_convs, skip_convs, up_convs, batchnorm, activation, upsampling):
        super().__init__()

        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        assert isinstance(down_channels, list)
        assert isinstance(skip_channels, list)
        assert isinstance(up_channels, list)
        assert isinstance(down_convs, list)
        assert isinstance(skip_convs, list)
        assert isinstance(up_convs, list)
        if batchnorm is not None:
            assert isinstance(batchnorm, bool)
        if activation is not None:
            assert isinstance(activation, str)
        if upsampling is not None:
            assert isinstance(upsampling, Upsample)

        # Encoder Channels: Combine in_channels and down_channels into a main_channels list. Combine main_channels list, main_channels list offset by 1, and skip_channels list into a tuple. When unpacked into a zip object, this will return in_channels, out_channels, skip_channels when iterated over
        main_channels = [in_channels, *down_channels]
        encoder_channels = (main_channels, main_channels[1:], skip_channels)
        self.encoder = nn.ModuleList([EncoderBlock(inc, outc, skipc, down_convs, skip_convs, batchnorm, activation) for inc, outc, skipc in zip(*encoder_channels)])

        # Decoder Channels: Update in_channels to last down_channel. Note: List[::-1] is used to return a reversed list. Combine in_channels and reversed up_channels into a main_channels list. Combine main_channels list, main_channels list offset by 1, and reversed skip_channels list into a tuple. When unpacked into a zip object, this will return in_channels, out_channels, skip_channels when iterated over
        in_channels = down_channels[-1]
        main_channels = [in_channels, *up_channels[::-1]]
        decoder_channels = (main_channels, main_channels[1:], skip_channels[::-1])
        self.decoder = nn.ModuleList([DecoderBlock(inc, outc, skipc, up_convs, batchnorm, activation, upsampling) for inc, outc, skipc in zip(*decoder_channels)])

        # Last Block Channels: Update in_channels to first up_channel. Note: conv_params  set to up_convs[1], batchnorm set to None, activation set to sigmoid.
        in_channels = up_channels[0]
        self.last = nn.Sequential(*convblock(in_channels, out_channels, conv_params=up_convs[1], batchnorm=None, activation='sigmoid'))

    def forward(self, x):
        skips = []
        for encoder_block in self.encoder:
            x, sx = encoder_block(x)
            skips.append(sx)

        for sx, decoder_block in zip(skips[::-1], self.decoder):
            x = decoder_block(x, sx)

        x = self.last(x)
        return x


# Starting In Channels
in_channels = 3

# Out Channels for Last Block (out), Encoder Main (down), Encoder Skip (skip), and Decoder (up)
out_channels = 3
down_channels = [8, 16, 32, 64, 128]
skip_channels = [0, 0,  0,  4,  4]
up_channels =   [8, 16, 32, 64, 128]

# Conv Named Tuple, takes in nn.Conv2d input arguments with corresponding names
Conv = namedtuple('Conv', 'kernel_sizes strides paddings padding_mode')

# Encoder Main Convs 1 & 2 Params, pass both in a list
down_conv1 = Conv(kernel_sizes=3, strides=2, paddings=1, padding_mode='reflect')
down_conv2 = Conv(kernel_sizes=3, strides=1, paddings=1, padding_mode='reflect')
down_convs = [down_conv1, down_conv2]

# Encoder Skip Conv Params, pass in a list
skip_conv1 = Conv(kernel_sizes=1, strides=1, paddings=0, padding_mode='reflect')
skip_convs = [skip_conv1]

# Decoder Convs 1 & 2 Params, pass both in a list
up_conv1 =   Conv(kernel_sizes=3, strides=1, paddings=1, padding_mode='reflect')
up_conv2 =   Conv(kernel_sizes=1, strides=1, paddings=0, padding_mode='reflect')
up_convs = [up_conv1, up_conv2]

# Unet Batchnorms (True / False / None) and Activation (see ModuleDict defined in UNet or ConvBlock for valid keys)
batchnorm = True
activation = 'leakyrelu'

# Upsample NamedTuple, takes in nn.Upsample input arguments with corresponding names
Upsample = namedtuple('Upsample', 'size scale_factor mode align_corners')
upsampling = Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=None)

# Starting In Channels, Ending Out Channels, Down/Up/Skip Out Channels
model = UNet(
    in_channels =   in_channels,
    out_channels =  out_channels,
    down_channels = down_channels,
    skip_channels = skip_channels,
    up_channels =   up_channels,
    down_convs =    down_convs,
    skip_convs =    skip_convs,
    up_convs =      up_convs,
    batchnorm =     batchnorm,
    activation =   activation,
    upsampling =    upsampling
    )

print(model)
# forwardpass(model)
# children(model)
# modules(model)
# params(model)
# shapes(model)

# print(torch.__version__)



















