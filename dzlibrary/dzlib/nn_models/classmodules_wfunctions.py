import numpy as np
import torch
import torch.nn as nn
from dzlib.utils.helper import info, modules, children, forward, params, shapes
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


global depth
depth = 1


def convblock(in_channels, out_channels, activation, **kwargs):
    activations = nn.ModuleDict([
        ['relu',        nn.ReLU()],
        ['leakyrelu',   nn.LeakyReLU(0.2)],
        ['sigmoid',     nn.Sigmoid()]
        ])

    assert activation in activations, f"{activation} not found in {activations}"

    convblock = []
    convblock.append(nn.Conv2d(in_channels, out_channels, **kwargs))
    convblock.append(nn.BatchNorm2d(out_channels))
    convblock.append(activations[activation])

    return convblock



def concat(main, skip):
    assert main.shape[2] == skip.shape[2]
    assert main.shape[3] == skip.shape[3]

    main = torch.cat((main, skip), dim=1)
    return main


class DecoderBlock(nn.Module):
    skip_functions = {'concat': concat()}

    def __init__(self, in_channels, out_channels, skip_channels, activation, padding_mode, skip_function):
        super().__init__()

        assert skip_function in self.skip_functions, f'{skip_function} not found in {self.skip_functions}'
        self.skip_channels = skip_channels
        self.skip_function = self.skip_functions[skip_function]

        self.upsample = nn.Upsample(scale_factor=upfactor, mode=upmode, align_corners=None)

        batchnorm = nn.BatchNorm2d(in_channels)
        main_branch1 = convblock(in_channels, out_channels, activation, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        main_branch2 = convblock(out_channels, out_channels, activation, kernel_size=1, stride=1, padding=1, padding_mode=padding_mode)
        self.main_branch = nn.Sequential(batchnorm, *main_branch1, *main_branch2)

    def forward(self, x, sx):
        x = self.upsample(x)
        if self.skip_channels != 0:
            x = self.skip_function(x, sx)

        x = self.main_branch(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, activation, padding_mode):
        super().__init__()

        self.skip_channels = skip_channels

        if self.skip_channels != 0:
            skip_branch = convblock(in_channels, skip_channels, activation, kernel_size=1, stride=1, padding=0)
            self.skip_branch = nn.Sequential(*skip_branch)

        main_branch1 = convblock(in_channels, out_channels, activation, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
        main_branch2 = convblock(out_channels, out_channels, activation, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.main_branch = nn.Sequential(*main_branch1, *main_branch2)

    def forward(self, x):
        global depth
        if self.skip_channels != 0:
            sx = self.skip_branch(x)
            info(sx, f'skip_branch {depth} output')

        else:
            sx = None

        x = self.main_branch(x)
        info(x, f'main_branch {depth} output')
        depth += 1

        return (x, sx)




# def encoderblock(in_channels, out_channels, skip_channels, activation, padding_mode):

#     convblock1 = convblock(in_channels, out_channels, activation, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
#     convblock2 = convblock(out_channels, out_channels, activation, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

#     if skip_channels != 0:
#         skipconvblock1 = convblock(in_channels, skip_channels, activation, kernel_size=1, stride=1, padding=0)



#     encoderblock = nn.Sequential(*convblock1, *convblock2)
#     return encoderblock


class Decoder(nn.Module):
    def __init__(self, channels, activation, padding_mode, skip_function):
        super().__init__()

        self.decoder_branch = nn.ModuleList([DecoderBlock(in_channels, out_channels, skip_channels, activation, padding_mode, skip_function) for in_channels, out_channels, skip_channels in channels])

    def forward(self, x, skips):
        # Reverse skips list?
        for i, decoder_block in enumerate(self.decoder_branch):
            x = decoder_block(x, skips[i])

        return x





class Encoder(nn.Module):
    def __init__(self, channels, activation, padding_mode):
        super().__init__()

        self.encoder_branch = nn.ModuleList([EncoderBlock(in_channels, out_channels, skip_channels, activation, padding_mode) for in_channels, out_channels, skip_channels in channels])

    def forward(self, x):
        skips = []
        for encoder_block in self.encoder_branch:
            x, sx = encoder_block(x)
            skips.append(sx)

        return x, skips


class UNet(nn.Module):

    # Valid activations strings and corresponding nn.modules for activation input argument
    activations = nn.ModuleDict([
        ['relu',        nn.ReLU()],
        ['leakyrelu',   nn.LeakyReLU(0.2)],
        ['sigmoid',     nn.Sigmoid()]
        ])

    padding_modes = ['reflect']

    def __init__(self, input_channels, enc_channels, skip_channels, activation='leakyrelu', padding_mode='reflect'):
        super().__init__()

        assert isinstance(input_channels, int)
        assert isinstance(enc_channels, (list, tuple))
        # assert isinstance(dec_channels, (list, tuple))
        assert isinstance(skip_channels, (list, tuple))
        assert activation in self.activations, f"activation: {activation} not found in {self.activations}"
        assert padding_mode in self.padding_modes, f"padding mode: {padding_mode} not found in {self.padding_modes}"

        self.enc_channels = [input_channels, *enc_channels]
        self.skip_channels = skip_channels
        # self.dec_channels = [enc_channels[-1], *dec_channels] # may have to add skip channels here? # also consider adding skip indices!
        channels = zip(self.enc_channels, self.enc_channels[1:], skip_channels)

        # self.encoder = EncoderBlock(self.enc_channels[0], self.enc_channels[1], activation, padding_mode)
        self.encoder = Encoder(channels, activation, padding_mode)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x



        # /Encoder: Contains all Encoder Blocks
        # //Block: Contains Skip Blocks
        # ///Skip: Contains Conv Blocks
        # ////Convblocks: Contains nn.Modules
        # ///Main: Contains Conv Blocks
        # ////ConvBlocks: Contains nn.Modules


model = UNet(3, [8, 16, 32], [4, 0, 7])
# forward(model)
# children(model)
# modules(model)
params(model)
shapes(model)
# print(model)


