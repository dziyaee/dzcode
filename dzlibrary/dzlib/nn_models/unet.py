import numpy as np
import torch
import torch.nn as nn
from dzlib.utils.helper import info, modules, children, forwardpass, params, shapes
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def convblock(in_channels, out_channels, **kwargs):
    convblock = []
    convblock.append(nn.Conv2d(in_channels, out_channels, **kwargs))
    convblock.append(nn.BatchNorm2d(out_channels))
    convblock.append(nn.LeakyReLU(0.2))
    return convblock


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

    # assert main.shape[2] == skip.shape[2], f"main shape: {main.shape}, skip shape: {skip.shape}"
    # assert main.shape[3] == skip.shape[3], f"main shape: {main.shape}, skip shape: {skip.shape}"
    # main = torch.cat((main, skip), dim=1)
    return main


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        # 1
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=None)
        # 2
        self.skip_function = None
        if skip_channels != 0:
            self.skip_function = concat
        # add skip_channels to in_channels to account for concatenation function. If skip = 0, in_channels won't change, no need to change formula
        in_channels += skip_channels
        # 3
        batchnorm = nn.BatchNorm2d(in_channels)
        main_branch1 = convblock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        main_branch2 = convblock(out_channels, out_channels, kernel_size=1, stride=1, padding=0, padding_mode='reflect')
        self.main_branch = nn.Sequential(batchnorm, *main_branch1, *main_branch2)

    def forward(self, x, sx):
        # 1
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
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        # 1
        self.skip_branch = None
        if skip_channels != 0:
            skip_branch = convblock(in_channels, skip_channels, kernel_size=1, stride=1, padding=0)
            self.skip_branch = nn.Sequential(*skip_branch)
        # 2
        main_branch1 = convblock(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        main_branch2 = convblock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
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


# class Decoder(nn.Module):
#     def __init__(self, channels, activation, padding_mode, skip_function):
#         super().__init__()

#         self.decoder_branch = nn.ModuleList([DecoderBlock(in_channels, out_channels, skip_channels, activation, padding_mode, skip_function) for in_channels, out_channels, skip_channels in channels])

#     def forward(self, x, skips):
#         # Reverse skips list?
#         for i, decoder_block in enumerate(self.decoder_branch):
#             x = decoder_block(x, skips[i])

#         return x





# class Encoder(nn.Module):
#     def __init__(self, channels, activation, padding_mode):
#         super().__init__()

#         self.encoder_branch = nn.ModuleList([EncoderBlock(in_channels, out_channels, skip_channels, activation, padding_mode) for in_channels, out_channels, skip_channels in channels])

#     def forward(self, x):
#         skips = []
#         for encoder_block in self.encoder_branch:
#             x, sx = encoder_block(x)
#             skips.append(sx)

#         return x, skips


class UNet(nn.Module):

    # Valid activations strings and corresponding nn.modules for activation input argument
    activations = nn.ModuleDict([
        ['relu',        nn.ReLU()],
        ['leakyrelu',   nn.LeakyReLU(0.2)],
        ['sigmoid',     nn.Sigmoid()]
        ])

    padding_modes = ['reflect']

    def __init__(self, in_channels, out_channels, down_channels, up_channels, skip_channels):
        super().__init__()

        # Check in_channels, out_channels are ints. Check down_, up_, skip_channels are lists of same length
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        assert isinstance(down_channels, list)
        assert isinstance(up_channels, list)
        assert isinstance(skip_channels, list)
        assert len(down_channels) == len(up_channels) == len(skip_channels)

        main_encoder_channels = [in_channels, *down_channels]
        encoder_channels = (main_encoder_channels, main_encoder_channels[1:], skip_channels)
        self.encoder = nn.ModuleList([EncoderBlock(inc, outc, skipc) for inc, outc, skipc in zip(*encoder_channels)])

        in_channels = down_channels[-1]
        main_decoder_channels = [in_channels, *up_channels[::-1]]
        decoder_channels = (main_decoder_channels, main_decoder_channels[1:], skip_channels[::-1])
        self.decoder = nn.ModuleList([DecoderBlock(inc, outc, skipc) for inc, outc, skipc in zip(*decoder_channels)])

        in_channels = up_channels[0]
        self.last = nn.Sequential(*convblock(in_channels, out_channels, kernel_size=1, stride=1, padding=0))


    def forward(self, x):

        skips = []
        for encoder_block in self.encoder:
            x, sx = encoder_block(x)
            skips.append(sx)

        for sx, decoder_block in zip(skips[::-1], self.decoder):
            x = decoder_block(x, sx)

        x = self.last(x)
        return x


model = UNet(3, 3, [8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [0, 0, 0, 4, 4])
print(model)
forwardpass(model)
# children(model)
# modules(model)
# params(model)
# shapes(model)



















