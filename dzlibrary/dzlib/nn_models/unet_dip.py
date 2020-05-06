import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from dzlib.utils.helper import calc_padding
import numpy as np


def convblock(in_channels, out_channels, kernel_size, stride):
    """Function to create and return a normal list containing an nn.Conv2d() and a nn.ReLU() module. This 'convblock' is the basic building block of the UNet architecture.

    Args:
        in_channels (int): number of input channels for the nn.Conv2d() module
        out_channels (int): number of output channels for the nn.Conv2d() module
        kernel_size (int): kernel size for the nn.Conv2d() module
        stride (int): kernel stride for the nn.Conv2d() module

    Returns:
        layers (list): a normal list containing the nn.Conv2d() and the nn.ReLU() module
    """
    padding = int((kernel_size - 1) / 2)
    padding_mode='reflect'
    conv_kwargs = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'padding_mode': padding_mode}
    layers = []
    layers.append(nn.Conv2d(**conv_kwargs))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    return layers


class EncoderBlock(nn.Module):

    """Class to create a single encoder block of the UNet architecture. This block normally consists of two 'convblocks' followed by a nn.MaxPool2d module.

    Attributes:
        convs (nn.Sequential): Contains an unpacked list of 'convblocks'
        pool (nn.MaxPool2d): The max pool layer responsible for down-sampling the image height and image width
    """

    def __init__(self, in_channels, out_channels, kernel_sizes, strides, maxpool_size, maxpool_stride):
        """EncoderBlock is initialized by creating two separate modules. The 'conv' module consists of one or more 'convblocks' unpacked into a nn.Sequential module. The 'pool' module consists of a nn.MaxPool2d module. These modules are separated because the output of the 'conv' module is used as both the input to the 'pool' module and as the input to the skip connection.

        Args:
            in_channels (int): number of input channels for the first 'convblock'
            out_channels (int): number of output channels for the all 'convblocks', and the number of input channels for all but the first 'convblocks'
            kernel_sizes (list): list of kernel sizes for each convblock
            strides (list): list of kernel strides for each convblock
            maxpool_size (int): kernel size of nn.MaxPool2d
            maxpool_stride (int): stride of nn.MaxPool2d
        """
        super().__init__()

        layers = []
        for kernel_size, stride in zip(kernel_sizes, strides):
            layers.extend([*convblock(in_channels, out_channels, kernel_size, stride)])
            in_channels = out_channels

        self.convs = nn.Sequential(*layers)
        # self.pool = nn.MaxPool2d(kernel_size=maxpool_size, stride=maxpool_stride)
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=maxpool_size, stride=maxpool_stride)

    def forward(self, x):
        """Forward pass for the EncoderBlock class. The input x passes through the convs module and outputs sx. The output sx passes through the pool module and outputs x. Both x (main branch) and sx (skip connection) are returned.

        Args:
            x (4d Tensor): Main branch UNet input. Shape: (N x C x H x W)

        Returns:
            x (4d Tensor): Main branch UNet output. Shape: (N x C' x H' x W')
            sx (4d Tensor): Skip connection UNet output. Shape: (N x C' x H x W)
        """
        sx = self.convs(x)
        # print(f'\nx before pool: {sx.shape}')
        x = self.pool(sx)
        # print(f'x after pool:  {x.shape}')
        return x, sx


class DecoderBlock(nn.Module):

    """Class to create a single decoder block of the UNet architecture. This block normally consists of one nn.ConvTranspose2d module followed by two 'convblocks'.

    Attributes:
        deconv (nn.ConvTranspose2d): The conv transpose layer responsible for up-sampling the image height and image width
        convs (nn.Sequential): Contains an unpacked list of 'convblocks'
    """

    def __init__(self, in_channels, out_channels, kernel_sizes, strides, convtranspose_size, convtranspose_stride):
        """DecoderBlock is initialized by creating two separate modules. The 'deconv' module consists of a nn.ConvTranspose2d module. The 'conv' module consists of one or more 'convblocks' unpacked into a nn.Sequential module. These modules are separated because the output of the 'deconv' module is concatenated with the skip connection input to form the input to the 'conv' module.

        Args:
            in_channels (int): number of input channels for the 'deconv' module and the first 'convblock' in the 'conv' module
            out_channels (int): number of output channels for the 'deconv' module, all 'convblocks', and the number of input channels for all but the first 'convblocks' in the 'conv' module
            kernel_sizes (list): list of kernel sizes for each 'convblock' in the 'conv' module
            strides (list): list of kernel strides for each 'convblock' in the 'conv' module
            convtranspose_size (int): kernel size of nn.ConvTranspose2d
            convtranspose_stride (int): stride of nn.ConvTranspose2d
        """
        super().__init__()

        # self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=convtranspose_size, stride=convtranspose_stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        layers = []
        for kernel_size, stride in zip(kernel_sizes, strides):
            layers.extend([*convblock(in_channels, out_channels, kernel_size, stride)])
            in_channels = out_channels

        self.convs = nn.Sequential(*layers)

    def forward(self, x, sx):
        """Forward Pass of the DecoderBlock class. The inputs x and sx form the main branch and skip connection inputs, respectively. Input x has shape (N x C' x H' x W'). Input sx has shape (N x C x H x W).
        First, x is passed through the 'deconv' module, which results in an output x of shape (N x C x H x W).
        Second, x and sx are concatenated along the channels axis to form an output x of shape (N x 2C x H x W).
        Finally, x is passed through the 'convs' module which results in an output of shape (N x C x H x W).

        Args:
            x (4d Tensor): Main branch UNet input. Shape: (N x C' x H' x W')
            sx (4d Tensor): Skip connection UNet input. Shape: (N x C x H x W)

        Returns:
            x (4d Tensor): Main branch UNet output. Shape: (N x C x H x W)
        """
        print(f'\nx before deconv: {x.shape}')
        x = self.conv(x)
        # x = self.deconv(x)
        size = (sx.shape[2], sx.shape[3])
        x = F.interpolate(input=x, size=size, mode='nearest')
        print(f'x after deconv:  {x.shape}')
        print(f'sx before cat:   {sx.shape}')
        x = torch.cat((x, sx), dim=1)
        print(f'x after cat:     {x.shape}')
        x = self.convs(x)
        return x


class UNet_DIP(nn.Module):

    """Class to create a neural net with a UNet architecture consisting of a main branch and one skip connection per encoder / decoder depth. The main branch travels through all encoder modules, a middle module, all decoder modules, and a last module.

    Attributes:
        encoder (nn.ModuleList): A nn.ModuleList into which EncoderBlocks are unpacked. ModuleList is used instead of Sequential because the skip connection inputs need to be collected for each EncoderBlock.
        mid (nn.Sequential): A nn.Sequential into which a list containing unpacked convblocks are unpacked. Sequential can be used here as there are no branching paths for the input.
        decoder (nn.ModuleList): A nn.ModuleList into which DecoderBlocks are unpacked. Similar to the encoder, ModuleList is used instead of Sequential because the skip connection outputs need to be passed alongside the main branch inputs to each DecoderBlock.
        last (nn.Sequential): A nn.Sequential into which a list containing a nn.Conv2d module is unpacked. Similar to mid, Sequential can be used here as there are no branching paths for the input.
        Used by _calc_padding() method to calculate padding necessary for downsampled image sizes to be equal to upsampled image sizes such that no cropping is needed during forward pass:
            maxpool_size (int): kernel size used by nn.MaxPool2d and nn.ConvTranspose2d modules to down or up sample the image.
            maxpool_stride (int): stride used by nn.MaxPool2d and nn.ConvTranspose2d modules to down or up sample the image.
            depth (int): number of nn.MaxPool2d layers
    """

    def __init__(self, input_channels, output_channels, encoder_channels, kernel_sizes, kernel_strides, maxpool_size, maxpool_stride):
        """UNet is initialized by first creating a channels_list by combining the input_channels and encoder_channels. From this list, in_channels and out_channels can be derived for each of the four modules (encoder, mid, decoder, last). All nn.Conv2d modules in the Encoder, Mid, and Decoder Layers use the kernel_sizes and kernel_strides lists as their kernel_size and stride input arguments. All nn.MaxPool2d and nn.ConvTranspose2d modules use the maxpool_size and maxpool_stride ints for their kernel_size and stride arguments. Depth is defined as the number of downsample or upsample operations, which corresponds to the number of nn.MaxPool2d or nn.ConvTranspose2d modules in the net. This is calculated by subtracting 1 from the length of the encoder_channels list. Depth, along with maxpool_size and maxpool_stride are stored as instance attributes and are used in the _calc_padding() method when the pad2d() method is called. pad2d() is intended to be used after initializing the net to calculate and apply sufficient padding for the net input. The crop2d() method can be called to undo the padding by center-cropping the output to the input's original size.

        Encoder Layers:
            in_channels: (first) to (third-last) element
            out_channels: (second) to (second-last) element

        Mid Layers:
            in_channels: (second-last) element
            out_channels: (last) element

        Decoder Layers:
            in_channels: (last) to (third) element
            out_channels: (second-last) to (second) element

        Last Layers:
            in_channels: (second) element
            out_channels: output_channels

        Args:
            input_channels (int): number of channels of net input
            output_channels (int): number of channels of net output
            encoder_channels (list): list of output channels of the 'contracting path', ending with the output channels of the 'middle path'
            kernel_sizes (list): list of kernel sizes of nn.Conv2d modules in any given encoder, mid, and decoder convblock
            kernel_strides (list): list of kernel strides of nn.Conv2d modules in any given encoder, mid, and decoder convblock
            maxpool_size (int): kernel size of nn.MaxPool2d in EncoderBlocks and nn.ConvTranspose2d in DecoderBlocks
            maxpool_stride (int): stride of nn.MaxPool2d in EncoderBlocks and nn.ConvTranspose2d in DecoderBlocks
        """
        super().__init__()

        # Asserts
        assert isinstance(encoder_channels, list)
        assert isinstance(kernel_sizes, list)
        assert isinstance(kernel_strides, list)
        assert len(kernel_sizes) == len(kernel_strides)
        for element in [input_channels, output_channels, *encoder_channels, *kernel_sizes, *kernel_strides, maxpool_size, maxpool_stride]:
            assert isinstance(element, int)

        # Create Layers
        in_channels_list = [input_channels, *encoder_channels, *encoder_channels[-2: : -1]]
        out_channels_list = [*encoder_channels, *encoder_channels[-2: : -1], output_channels]
        d = len(encoder_channels) - 1

        encoder_channels =  (in_channels_list[:d],      out_channels_list[:d])
        mid_channels =      (in_channels_list[d],       out_channels_list[d])
        decoder_channels =  (in_channels_list[d+1: -1], out_channels_list[d+1: -1])
        last_channels =     (in_channels_list[-1],      out_channels_list[-1])

        encoder_blocks = []
        for in_channels, out_channels in zip(*encoder_channels):
            encoder_blocks.append(EncoderBlock(in_channels, out_channels, kernel_sizes, kernel_strides, maxpool_size, maxpool_stride))

        mid_block = []
        in_channels, out_channels = mid_channels
        for kernel_size, kernel_stride in zip(kernel_sizes, kernel_strides):
            mid_block.extend([*convblock(in_channels, out_channels, kernel_size, kernel_stride)])
            in_channels = out_channels

        decoder_blocks = []
        for in_channels, out_channels in zip(*decoder_channels):
            decoder_blocks.append(DecoderBlock(in_channels, out_channels, kernel_sizes, kernel_strides, maxpool_size, maxpool_stride))

        last_block = []
        in_channels, out_channels = last_channels
        last_block.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()])

        # UNet Layers
        self.encoder = nn.ModuleList([*encoder_blocks])
        self.mid = nn.Sequential(*mid_block)
        self.decoder = nn.ModuleList([*decoder_blocks])
        self.last = nn.Sequential(*last_block)

        # _calc_padding variables
        self.maxpool_size = maxpool_size
        self.maxpool_stride = maxpool_stride
        self.depth = d

    def pad2d(self, ninput):
        """Function to calculate and apply any necessary padding to a net input such that the height and width sizes of the downsampled and upsampled images are equal so that no cropping is necessary during the forward pass. If no padding is needed along both dimensions, the original input is returned. If an odd number of padding is necessary, the input is biased towards the top and left, such that if there is 'p' padding on the top/left, there will be 'p+1' padding on the bottom/right. The padding variables are stored as instance attributes which are used to reverse this process when the crop2d() method is called.

        Args:
            ninput (4d Tensor): Net Input of shape (N x C x H x W)

        Returns:
            ninput (4d Tensor): Unchanged Net Input of shape (N x C x H x W)
            padded_ninput (4d Tensor): Padded Net Input of shape (N x C x H' x W'), where H' = H + padding_height, and W' = W + padding_width
        """
        self._top =  self._left =  self._height =  self._width = None
        num, channels, height, width = ninput.shape
        padding_height, padding_width = self._calc_padding(height, width)
        if (padding_height == 0 and padding_width == 0):
            return ninput

        else:
            padded_ninput = torch.zeros(num, channels, height + padding_height, width + padding_width)
            top = int(np.floor(padding_height / 2))
            left = int(np.floor(padding_width / 2))
            padded_ninput[:, :, top: height + top, left: width + left] = ninput[:, :, :, :]

            self._top, self._left, self._height, self._width = top, left, height, width
            return padded_ninput

    def crop2d(self, noutput):
        if self._top == self._left == self._height == self._width == None:
            return noutput

        else:
            top, left, height, width = self._top, self._left, self._height, self._width
            cropped_noutput = noutput[:, :, top: height + top, left: width + left]
            return cropped_noutput

    def _calc_padding(self, height, width):
        """Function used by the pad2d() method to calculate any necessary padding to a net input such that the height and width sizes of the downsampled and upsampled images are equal so that no cropping is necessary during the forward pass.
        Assumptions made for padding calculation:
            1) For every downsampling operation, there is an upsampling operation
            2) All downsampling operations use the same kernel size and stride
            3) All upsampling operations use the same kernel size and stride, which are equal to the downsampling kernel size and stride
            4)
        The padding calculation is based on the following variables:


        Args:
            height (int): original height of net input
            width (int): original width of net input

        Returns:
            padding (list): list of 2 ints: necessary padding for height and width, respectively
        """
        k = self.maxpool_size
        s = self.maxpool_stride
        d = self.depth

        padding = []
        for x in [height, width]:
            xmin_step = (s ** d - 1) / (s - 1)
            xmin = 1 + (k - 1) * xmin_step

            n = np.ceil((x - xmin) / s ** d)
            xvalid = xmin + n * s ** d

            xnew = max(xmin, xvalid)
            padding.append(int(xnew - x))

        return padding

    def forward(self, x):
        """Forward pass of the UNet. UNet takes an input x of shape (N x C x H x W).
        1) Input x passes through each successive 'EncoderBlock' in the 'Encoder' ModuleList which outputs x. Additionally, after each 'EncoderBlock' the skip connection output sx is also appended to a list 'skips'.
        2) Output x passes through the 'Mid' Sequential Module which outputs x.
        3) Output x passes through each successive 'DecoderBlock' in the 'Decoder' ModuleList along with each skip output sx in the reversed skips list. This outputs x.
        4) Output x passes through the 'Last' Sequential Module which results in the final output x.

        Args:
            x (4d Tensor): Initial UNet input. Shape: (N x C x H x W)

        Returns:
            x (4d Tensor): Final UNet output. Shape: (N x C' x H x W)
        """
        # Encoder
        skips = []
        for encoderblock in self.encoder:
            x, sx = encoderblock(x)
            skips.append(sx)

        # Mid
        x = self.mid(x)

        # Decoder
        for decoderblock, sx in zip(self.decoder, skips[::-1]):
            x = decoderblock(x, sx)

        # Last
        x = self.last(x)
        return x


if __name__ == '__main__':
    n, c, h, w = (1, 3, 400, 415)
    ninput = torch.randn(n, c, h, w)
    print(f"Original Height = {h}, Original Width = {w}")

    encoder_channels = [8, 16, 32, 64, 128]
    k, s = 2, 2

    net = UNet_DIP(input_channels=c, output_channels=c, encoder_channels=encoder_channels, kernel_sizes=[3, 3], kernel_strides=[1, 1], maxpool_size=k, maxpool_stride=s)
    ninput = net.pad2d(ninput)
    n, c, h, w = ninput.shape
    print(f"Padded Height =   {h}, Padded Width =   {w}")

    noutput = net(ninput)
    print(f"Noutput:         {noutput.shape}")
