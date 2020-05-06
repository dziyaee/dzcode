import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple


def convblock(in_channels, out_channels, kernel_size, stride):
    """Function to create and return a normal list containing an nn.Conv2d() and a nn.ReLU() module. This 'convblock' is the basic building block of the UNet. The nn.Conv2d module uses padding

    Args:
        in_channels (int): number of input channels for the nn.Conv2d() module
        out_channels (int): number of output channels for the nn.Conv2d() module
        kernel_size (int): kernel size for the nn.Conv2d() module
        stride (int): kernel stride for the nn.Conv2d() module

    Returns:
        layers (list): a normal list containing the nn.Conv2d() and the nn.ReLU() module
    """

    # Padding to keep output size exactly equal to input size for odd kernel size with stride of 1
    padding = int(np.floor((kernel_size - 1) / 2))
    padding_mode = 'reflect'

    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode))
    layers.append(nn.ReLU())
    return layers


class EncoderBlock(nn.Module):

    """Class to create a single encoder block of the UNet. This block normally consists of one or more 'convblocks' followed by a nn.MaxPool2d module.

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

        # Iterate through the kernel sizes and kernel strides lists to create each convblock
        layers = []
        for kernel_size, stride in zip(kernel_sizes, strides):
            layers.extend([*convblock(in_channels, out_channels, kernel_size, stride)])

            # After the first convblock, out_channels is used as both the in and out channels of each subsequent convblock
            in_channels = out_channels

        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=maxpool_size, stride=maxpool_stride)

    def forward(self, x):
        """Forward pass for the EncoderBlock class. The input x passes through the convs module and outputs sx. The output sx passes through the pool module and outputs x. Both x (main branch) and sx (skip connection) are returned.

        Args:
            x (4d Tensor): Main branch UNet input. Shape: (N x C x H x W)

        Returns:
            x (4d Tensor): Main branch UNet output. Shape: (N x C' x H' x W')
            sx (4d Tensor): Skip connection UNet output. Shape: (N x C' x H x W)
        """
        sx = self.convs(x)
        x = self.pool(sx)
        return x, sx


class DecoderBlock(nn.Module):

    """Class to create a single decoder block of the UNet. This block normally consists of one nn.ConvTranspose2d module followed by one or more 'convblocks'

    Attributes:
        convs (nn.Sequential): Contains an unpacked list of 'convblocks'
        deconv (nn.ConvTranspose2d): The conv transpose layer responsible for up-sampling the image height and image width
    """

    # Namedtuple to store tensor dimensions data. Used in the _concat method
    _TensorDims = namedtuple('_TensorDims', 'num channels height width')

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

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=convtranspose_size, stride=convtranspose_stride)

        # Iterate through the kernel sizes and kernel strides lists to create each convblock
        layers = []
        for kernel_size, stride in zip(kernel_sizes, strides):
            layers.extend([*convblock(in_channels, out_channels, kernel_size, stride)])

            # After the first convblock, out_channels is used as both the in and out channels of each subsequent convblock
            in_channels = out_channels

        self.convs = nn.Sequential(*layers)

    def _concat(self, tensor1, tensor2):
        """Method to concatenate two 4d tensors along axis/dimension 1. If both tensors have equal heights and equal widths, they will be concatenated without any cropping. If either heights or widths are not equal, both tensors will be center-cropped using the smallest height and smallest width.

        Example:
            tensor1 shape (1 x 8 x 12 x 17)
            tensor2 shape (1 x 5 x 15 x 10)

            new height = min(12, 15) = 12
            new width = min(17, 10) = 10

            new tensor1 shape (1 x 8 x 12 x 10)
            new tensor2 shape (1 x 5 x 12 x 10)

            output shape (1 x 13 x 12 x 10)

        Args:
            tensor1 (4d Tensor): Shape (N x C1 x H1 x W1)
            tensor2 (4d Tensor): Shape (N x C2 x H2 x W2)

        Returns:
            (4d Tensor): Shape (N x (C1 + C2), min(H1, H2), min(W1, W2))
        """

        # tensor1 and 2 dimensions (N x C x H x W)
        tn1 = self._TensorDims(*tensor1.shape)
        tn2 = self._TensorDims(*tensor2.shape)

        # no cropping necessary if height and width are equal
        if ((tn1.height == tn2.height) and (tn1.width == tn2.width)):
            return torch.cat((tensor1, tensor2), dim=1)

        else:
            # use minimum height and width as new dimensions for center cropping
            height, width = min(tn1.height, tn2.height), min(tn1.width, tn2.width)
            diff = lambda old, new: (old - new) // 2

            # crop tensor1
            diff_height, diff_width = diff(tn1.height, height), diff(tn1.width, width)
            tensor1 = tensor1[:, :, diff_height: height + diff_height, diff_width: width + diff_width]

            # crop tensor2
            diff_height, diff_width = diff(tn2.height, height), diff(tn2.width, width)
            tensor2 = tensor2[:, :, diff_height: height + diff_height, diff_width: width + diff_width]
            return torch.cat((tensor1, tensor2), dim=1)

    def forward(self, x, sx):
        """Forward Pass of the DecoderBlock class. The inputs x and sx form the main branch and skip connection inputs, respectively. Input x has shape (N x C1 x H1' x W1'). Input sx has shape (N x C1/2 x H2 x W2).
        First, x is passed through the 'deconv' module, which results in an output x of shape (N x C1/2 x H1 x W1).
        Second, x and sx are concatenated along the channels axis to form an output x of shape (N x C1 x min(H1, H2) x min(W1, W2)).
        Finally, x is passed through the 'convs' module which results in an output of the same shape.

        Args:
            x (4d Tensor): Main branch UNet input. Shape: (N x C1 x H1' x W1')
            sx (4d Tensor): Skip connection UNet input. Shape: (N x C1/2 x H2 x W2)

        Returns:
            x (4d Tensor): Main branch UNet output. Shape: (N x C1 x min(H1, H2) x min(W1, W2))
        """
        x = self.deconv(x)
        x = self._concat(x, sx)
        x = self.convs(x)
        return x


class UNet(nn.Module):
    """Class to create a neural net with a UNet architecture consisting of a main branch and one skip connection per encoder / decoder block pair. The main branch travels through all encoder blocks, a middle block, all decoder blocks, and a last block.

    Attributes:
        decoder (nn.ModuleList): A nn.ModuleList into which DecoderBlocks are unpacked. Similar to the encoder, ModuleList is used instead of Sequential because the skip connection outputs need to be passed alongside the main branch inputs to each DecoderBlock.
        encoder (nn.ModuleList): A nn.ModuleList into which EncoderBlocks are unpacked. ModuleList is used instead of Sequential because the skip connection inputs need to be collected for each EncoderBlock.
        last (nn.Sequential): A nn.Sequential into which a list containing a nn.Conv2d module is unpacked. Similar to mid, Sequential can be used here as there are no branching paths for the input.
        mid (nn.Sequential): A nn.Sequential into which a list containing unpacked convblocks are unpacked. Sequential can be used here as there are no branching paths for the input.
    """
    def __init__(self, input_channels, output_channels, encoder_channels, kernel_sizes, kernel_strides, maxpool_size, maxpool_stride):
        """UNet is initialized by first creating a channels_list by combining the input_channels and encoder_channels. From this list, in_channels and out_channels can be derived for each of the four modules (encoder, mid, decoder, last). All nn.Conv2d modules in the Encoder, Mid, and Decoder Layers use the kernel_sizes and kernel_strides lists as their kernel_size and stride input arguments. All nn.MaxPool2d and nn.ConvTranspose2d modules use the maxpool_size and maxpool_stride ints for their kernel_size and stride arguments.

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
        assert isinstance(input_channels,   int), f"input_channels must be an int, got {type(input_channels)} instead"
        assert isinstance(output_channels,  int), f"output_channels must be an int, got {type(output_channels)} instead"
        assert isinstance(maxpool_size,     int), f"maxpool_size must be an int, got {type(maxpool_size)} instead"
        assert isinstance(maxpool_stride,   int), f"maxpool_stride must be an int, got {type(maxpool_stride)} instead"
        assert isinstance(encoder_channels, (list, tuple)), f"encoder_channels must be a list or tuple, got {type(encoder_channels)} instead"
        assert isinstance(kernel_sizes,     (list, tuple)), f"kernel_sizes must be a list or tuple, got {type(kernel_sizes)} instead"
        assert isinstance(kernel_strides,   (list, tuple)), f"kernel_strides must be a list or tuple, got {type(kernel_strides)} instead"
        assert len(kernel_sizes) == len(kernel_strides), f"kernel_sizes and kernel_strides must be same length, got {len(kernel_sizes), len(kernel_strides)} instead"
        for element in (*encoder_channels, *kernel_sizes, *kernel_strides):
            assert isinstance(element, int), f"{element} must be an int, got {type(element)} instead"

        # Create lists of input channels and output channels of the first conv module in each block
        in_channels_list = [input_channels, *encoder_channels, *encoder_channels[-2: : -1]]
        out_channels_list = [*encoder_channels, *encoder_channels[-2: : -1], output_channels]

        # Number of downsamples
        d = len(encoder_channels) - 1

        # Assign tuples of input / output channel list pairs for the encoder, mid, decoder, and last blocks
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

        # Unpack lists of blocks into final blocks
        self.encoder = nn.ModuleList([*encoder_blocks])
        self.mid = nn.Sequential(*mid_block)
        self.decoder = nn.ModuleList([*decoder_blocks])
        self.last = nn.Sequential(*last_block)

    def forward(self, x):
        """Forward pass of the UNet. UNet takes an input x of shape (N x C x H x W).
        1) Encoder: Input x passes through each successive 'EncoderBlock' in the 'Encoder' ModuleList which outputs x. Additionally, after each 'EncoderBlock' the skip connection output sx is also appended to a list 'skips'.
        2) Mid: Output x passes through the 'Mid' Sequential Module which outputs x.
        3) Decoder: Output x passes through each successive 'DecoderBlock' in the 'Decoder' ModuleList along with each skip output sx in the reversed skips list. This outputs x.
        4) Last: Output x passes through the 'Last' Sequential Module which results in the final output x.

        Args:
            x (4d Tensor): Initial UNet input. Shape: (N x C x H x W)

        Returns:
            x (4d Tensor): Final UNet output. Shape: (N x C' x H' x W')
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
