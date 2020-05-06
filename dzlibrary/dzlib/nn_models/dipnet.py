import torch
import torch.nn as nn
from collections import namedtuple
import warnings
warnings.simplefilter("ignore", UserWarning)


def convblock(in_channels, out_channels, conv2d_args, batchnorm, activation):
    ''' Function to return a list of nn.Modules consisting of a Conv2d followed by optional BatchNorm2d and activation modules.
    The nn.Conv2d module input arguments consist of in_channels, out_channels, and conv_params. in_channels and out_channels are ints. conv_params is a namedtuple which is converted into a dict and passed as a **kwargs.
    The optional nn.BatchNorm2d module takes the out_channels as the sole input argument.
    The optional nn.[activation] module uses the activation string as a key for the activations ModuleDict defined within the DIPNet class

        Args:
            in_channels, out_channels: ints
            conv2d_args: namedtuple, contains kernel_size, stride, padding, padding_mode, intended to be unpacked into nn.Conv2d() with **
            batchnorm: bool or None, set to True to include a BatchNorm2d module
            activation: string or None, string corresponding to the key of the desired activation module

        Returns:
            block: list of nn.modules. Intended to be unpacked into an nn.Sequential() or nn.ModuleList with *.
    '''
    # ModuleDict containing valid activation key and module pairs. This is defined as a DIPNet class variable
    activations = DIPNet.activations
    pad = int((conv2d_args.kernel_size - 1) / 2)

    # Convert named tuple to a dict which can be passed as a **kwargs to nn.Conv2d()
    conv2d_args = conv2d_args._asdict()


    block = []
    block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=pad, **conv2d_args))

    if batchnorm is True:
        block.append(nn.BatchNorm2d(out_channels))

    if activation is not None:
        block.append(activations[activation])

    return block


def concat(main, skip):
    ''' Function to concatenate the main-branch and skip-branch outputs in the decoder block of the DIPNet along the channels dimension (dim=1). If the heights and widths of the main and skip images are not the same, the larger image will be center-cropped to the same size as the smaller image

        Args:
            main, skip: 4d tensors of shapes (1, C1, H1, W1) and (1, C2, H2, W2)

        Returns:
            new_main: 4d tensor of shape (1, C1+C2, min(H1, H2), min(W1, W2))
    '''
    main_height, main_width = main.shape[2], main.shape[3]
    skip_height, skip_width = skip.shape[2], skip.shape[3]

    # Only if both H1 = H2 and W1 = W2 will this be skipped
    if not ((main_height == skip_height) and (main_width == skip_width)):

        # New Dim = min(Dim1, Dim2)
        new_height, new_width = min(main_height, skip_height), min(main_width, skip_width)

        # Diff = (Old Dim - New Dim) // 2
        diff = lambda old, new: (old - new) // 2

        # Center Crop Main using Diffs and New Dims
        diff_height, diff_width = diff(main_height, new_height), diff(main_width, new_width)
        main = main[:, :, diff_height: diff_height + new_height, diff_width: diff_width + new_width]

        # Center Crop Skip using Diffs and New Dims
        diff_height, diff_width = diff(skip_height, new_height), diff(skip_width, new_width)
        skip = skip[:, :, diff_height: diff_height + new_height, diff_width: diff_width + new_width]

    new_main = torch.cat((main, skip), dim=1)

    return new_main


class EncoderBlock(nn.Module):
    ''' Class defining a single encoder block of the DIPNet. This block contains a Main Branch to downsample the input, and an optional Skip Branch to act as a skip connection.

        Attributes:
            main_branch and skip_branch: Each branch is a separate nn.Sequential module containing convblocks created by the convblock function. Note: The convblock function returns a list of modules which is then unpacked into the nn.Sequential module using *.

                main_branch: a nn.Sequential module made up of two convblocks: main_branch1, main_branch2.
                skip_branch: a nn.Sequential module made up of one convblock: skip_branch1.
    '''

    def __init__(self, in_channels, out_channels, skip_channels, down_convs, skip_convs, batchnorm, activation):
        ''' Args:
                args passed to convblock function:
                    in_channels, out_channels, skip_channels: ints to be used as Conv2d input arguments for in_channels and out_channels
                    down_convs, skip_convs: lists of namedtuples to be used as Conv2d input arguments for kernel_size, stride, padding and padding_mode
                    batchnorm: bool or None to indicate existence of batchnorm module in convblock
                    activation: string or None to indicate existence and type of activation module in convblock
        '''
        super().__init__()

        # skip_branch convblock is only created if skip_channels is not zero, else skip_branch is None.
        self.skip_branch = None
        if skip_channels != 0:
            skip_branch = []
            skip_in_channels = in_channels
            for skip_conv in skip_convs:
                skip_branch.extend(convblock(skip_in_channels, skip_channels, skip_conv, batchnorm, activation))
                skip_in_channels = skip_channels
            self.skip_branch = nn.Sequential(*skip_branch)

        main_branch = []
        for down_conv in down_convs:
            main_branch.extend(convblock(in_channels, out_channels, down_conv, batchnorm, activation))
            in_channels = out_channels

        self.main_branch = nn.Sequential(*main_branch)

    def forward(self, x):
        ''' Forward pass of Encoder Block. Takes 1 input x, and returns 2 outputs (x, sx) corresponding to the main_branch and skip_branch respectively.
        If skip_branch is None, sx is also None.

            Args:
                x: 4d tensor of shape (1, C, H, W), main_branch and skip_branch input

            Returns:
                x: 4d tensor of shape (1, C', H', W'), main_branch output
                sx: 4d tensor of shape (1, C'', H'', W''), skip_branch output
        '''

        # skip connection, if skip_branch is None, sx is also None
        sx = None
        if self.skip_branch is not None:
            sx = self.skip_branch(x)

        # main connection
        x = self.main_branch(x)

        return (x, sx)


class DecoderBlock(nn.Module):
    ''' Class defining a single decoder block of the DIPNet. This block contains an Upsample module, and Skip Function to merge merge main and skip connections, and a Main Branch.

        Attributes:
            upsample: a nn.Upsample module.
            skip_function: a function used to concatenate main and skip connection images
            main_branch: nn.Sequential module containing a batchnorm module and two convblocks (main_branch1, main_branch2) created by the convblock function. Note: The convblock function returns a list of modules which is then unpacked into the nn.Sequential module using *.
    '''

    def __init__(self, in_channels, out_channels, skip_channels, up_convs, batchnorm, activation, upsample_args):
        ''' Args:
                args passed to convblock function:
                    in_channels, out_channels, skip_channels: ints to be used as Conv2d input arguments for in_channels and out_channels
                    down_convs, skip_convs: lists of namedtuples to be used as Conv2d input arguments for kernel_size, stride, padding and padding_mode
                    batchnorm: bool or None to indicate existence of batchnorm module in convblock
                    activation: string or None to indicate existence and type of activation module in convblock
                upsample_args: namedtuple, contains size, scale_factor, mode, align_corners
        '''
        super().__init__()

        # Convert named tuple to a dict which can be passed as a **kwargs to nn.Upsample()
        upsample_args = upsample_args._asdict()
        self.upsample = None
        if upsample_args is not None:
            self.upsample = nn.Upsample(**upsample_args)

        # skip_function is only set if skip_channels is not zero, else skip_function is None
        self.skip_function = None
        if skip_channels != 0:
            self.skip_function = concat

        # skip_channels is added to in_channels to account for concatenation
        in_channels += skip_channels
        main_branch = [nn.BatchNorm2d(in_channels)]
        for up_conv in up_convs:
            main_branch.extend(convblock(in_channels, out_channels, up_conv, batchnorm, activation))
            in_channels = out_channels

        self.main_branch = nn.Sequential(*main_branch)

    def forward(self, x, sx):
        ''' Forward pass of Decoder Block. Takes 2 inputs (x, sx), corresponding to the main and skip connections respectively, and returns 1 output x.
        If skip_function is None, sx is not used.

            Args:
                x: 4d tensor of shape (1, C, H, W), upsample and skip_function input
                sx: 4d tensor of shape (1, C, H', W'), skip_function input

            Returns:
                x: 4d tensor of shape (1, C'', H'', W''), main_branch output
        '''

        # Upsample main branch
        if self.upsample is not None:
            x = self.upsample(x)

        # Concatenate main and skip branches
        if self.skip_function is not None:
            x = self.skip_function(x, sx)

        # main branch
        x = self.main_branch(x)

        return x


class DIPNet(nn.Module):
    ''' Class defining a DIPNet architecture consisting of Encoder Blocks, Decoder Blocks, and a Last Block. There is a Main Branch running through all these blocks, and there are optional Skip Branches which branch off the input x during any given Encoder Block, and merge back during a corresponding Decoder Block. Each Block consists of smaller blocks of modules called a convblock.
    - convblock structure: (Conv2d, BatchNorm2d, Activation), where BatchNorm2d and Activation are optional

    The EncoderBlocks are used to downsample the image as well as branch it off as a skip connection if needed.
    - EncoderBlock structure: (MainBranch: (convblock1, convblock2), SkipBranch: (convblock1)), where SkipBranch is optional.

    The DecoderBlocks are used to upsample the image as well as merge the skip connection if needed.
    - DecoderBlock structure: (Upsample, Concat(Main, Skip), BatchNorm2d, convblock1, convblock2), where Concat is optional.

    The LastBlock is used to obtain the desired out_channels for the final output image.
    - LastBlock structure: (Conv2d, BatchNorm2d, Activation), where BatchNorm2d and Activation are optional

        Attributes:
            activations: nn.ModuleDict containing key, module pairs for valid activation functions. This is a class variable
            encoder: nn.ModuleList containing all Encoder Blocks in sequential order
            decoder: nn.ModuleList containing all Decoder Blocks in sequential order
            last: nn.Sequential containing 1 convblock
    '''

    # ModuleDict containing valid activation key and module pairs
    activations = nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leakyrelu', nn.LeakyReLU(0.2)],
        ['sigmoid', nn.Sigmoid()]])

    Conv = namedtuple('Conv', 'kernel_size stride padding_mode')
    Upsample = namedtuple('Upsample', 'size scale_factor mode align_corners')

    def __init__(self, in_channels, out_channels, down_channels, skip_channels, up_channels, down_convs, skip_convs, up_convs, batchnorm, last_batchnorm, activation, last_activation, upsample_args):
        ''' Args:
                ints:
                in_channels: number of input channels of DIPNet input (used as in_channels in first encoder block)
                out_channels: number of output channels of DIPNet output (used as out_channels in last block)

                lists of ints:
                down_channels: number of output channels within each successive depth of the DIPNet Encoder main branch
                skip_channels number of output channels within each successive depth of the DIPNet Encoder skip branch
                up_channels: number of output channels within each successive depth of the DIPNet Decoder main branch

                lists of namedtuples of structure (kernel_size, stride, padding, padding_mode), passed as input args for nn.Conv2d()
                down_convs: convblock1 and convblock2 args of the DIPNet Encoder main branch convblocks
                skip_convs: convblock1 args of the DIPNet Encoder skip branch convblock
                up_convs: convblock1 and convblock2 args of the DIPNet Decoder main branch convblocks

                batchnorm, last_batchnorm: bool or None to toggle existence of batchnorm module within each convblock and last convblock respectively
                activation, last_activation: string or None to toggle existence and type of activation module within each convblock and last convblock respectively. string must correspond to a key within the activations ModuleDict defined as a DIPNet class variable

                upsample_args: named tuple of structure (size, scale_factor, mode, align_corners), passed as input args for nn.Upsample() within the DIPNet Decoder Blocks
        '''
        super().__init__()

        # Assert Ints
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        # Assert Lists of Ints
        assert isinstance(down_channels, list)
        assert isinstance(skip_channels, list)
        assert isinstance(up_channels, list)
        for channels in [*down_channels, *skip_channels, *up_channels]:
            assert isinstance(channels, int)

        # Assert Lists of NamedTuples of name Conv
        assert isinstance(down_convs, list)
        assert isinstance(skip_convs, list)
        assert isinstance(up_convs, list)
        for conv_args in [*down_convs, *skip_convs, *up_convs]:
            assert isinstance(conv_args, self.Conv)

        # Assert Bool or None
        assert (isinstance(batchnorm, bool) or batchnorm is None)
        assert (isinstance(last_batchnorm, bool) or last_batchnorm is None)

        # Assert in self.activations ModuleDict or None
        assert (activation in self.activations or activation is None)
        assert (last_activation in self.activations or last_activation is None)

        # Assert NamedTuple of name Upsample or None
        assert (isinstance(upsample_args, self.Upsample) or upsample_args is None)

        # main_channels: list of input channels of encoder main branch and skip branch
        # main_channels[1:]: list of output channels of encoder main branch
        # skip_channels: list of output channels of encoder skip branch
        # These are combined into a tuple which is then unpacked into a zip object. When iterated over, this returns a tuple of 3 elements, 1 from each list
        main_channels = [in_channels, *down_channels]
        encoder_channels = (main_channels, main_channels[1:], skip_channels)
        self.encoder = nn.ModuleList([EncoderBlock(inc, outc, skipc, down_convs, skip_convs, batchnorm, activation) for inc, outc, skipc in zip(*encoder_channels)])

        # in_channels: update starting input channels to last output channels of encoder block
        # main_channels: list of input channels of decoder main branch if no corresponding skip connection
        # main_channels[1:]: list of output channels of decoder main branch
        # skip_channels[::-1]: reversed list of skip_channels, to be added to input channels to account for possible skip connections
        # These are combined into a tuple which is then unpacked into a zip object. When iterated over, this returns a tuple of 3 elements, 1 from each list
        in_channels = down_channels[-1]
        main_channels = [in_channels, *up_channels[::-1]]
        decoder_channels = (main_channels, main_channels[1:], skip_channels[::-1])
        self.decoder = nn.ModuleList([DecoderBlock(inc, outc, skipc, up_convs, batchnorm, activation, upsample_args) for inc, outc, skipc in zip(*decoder_channels)])

        # in_channels: update starting input channels to last output channels of decoder block, contained as the first element in the list
        in_channels = up_channels[0]
        self.last = nn.Sequential(*convblock(in_channels, out_channels, conv2d_args=up_convs[-1], batchnorm=last_batchnorm, activation=last_activation))

    def forward(self, x):
        ''' Forward Pass of DIPNet:
        Takes 1 input x which is passed sequentially to each EncoderBlock in the encoder ModuleList. For each input x to an EncoderBlock, two outputs are returned (x, sx). The sx outputs are appended to a list (skips). These outputs represent the sequential outputs of the skip connections, while the single output x represents the output of the main connection.

        Next, a zip object is created with the reversed skips list and the decoder ModuleList. The main branch output x, and the skip branch outputs sx are passed sequentially to each DecoderBlock which returns a new main branch output x.

        Finally, the main branch output x is passed to the last Sequential block
        '''

        # Iterate over encoder blocks, passing x and returned (x, sx). Append each sx to skips list
        skips = []
        for encoder_block in self.encoder:
            x, sx = encoder_block(x)
            skips.append(sx)

        # Iterate over decoder blocks with reversed skips list, passing (x, sx).
        for sx, decoder_block in zip(skips[::-1], self.decoder):
            x = decoder_block(x, sx)

        # Pass x to last block
        x = self.last(x)
        return x


if __name__ == '__main__':
    # Starting Input Channels for first Encoder Block
    in_channels = 3

    # Final Output Channels for Last Block
    out_channels = 3

    # Output Channels for Encoder Main (down), Encoder Skip (skip), and Decoder (up) Blocks
    down_channels = [8, 16, 32, 64, 128]
    skip_channels = [0, 0,  0,  4,  4]
    up_channels =   [8, 16, 32, 64, 128]

    # DIPNet.Conv, NamedTuple, takes in nn.Conv2d input arguments with corresponding names. Converted to a dict and passed as a **kwargs
    # Encoder Main Convs 1 & 2 Params, pass both in a list
    down_conv1 = DIPNet.Conv(kernel_size=3, stride=2, padding_mode='reflect')
    down_conv2 = DIPNet.Conv(kernel_size=3, stride=1, padding_mode='reflect')
    down_convs = [down_conv1, down_conv2]

    # Encoder Skip Conv 1 Params, pass in a list
    skip_conv1 = DIPNet.Conv(kernel_size=1, stride=1, padding_mode='reflect')
    skip_convs = [skip_conv1]

    # Decoder Main Convs 1 & 2 Params, pass both in a list
    up_conv1 = DIPNet.Conv(kernel_size=3, stride=1, padding_mode='reflect')
    up_conv2 = DIPNet.Conv(kernel_size=1, stride=1, padding_mode='reflect')
    up_convs = [up_conv1, up_conv2]

    # Unet Batchnorms (True / False / None) and Activations (see ModuleDict defined in as DIPNet class variable)
    batchnorm = True
    last_batchnorm = None
    activation = 'leakyrelu'
    last_activation = 'sigmoid'

    # DIPNet.Upsample NamedTuple, takes in nn.Upsample input arguments with corresponding names. Converted to a dict and passed as a **kwargs
    upsample_args = DIPNet.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=None)

    # pass params to DIPNet class
    model = DIPNet(
        in_channels =       in_channels,
        out_channels =      out_channels,
        down_channels =     down_channels,
        skip_channels =     skip_channels,
        up_channels =       up_channels,
        down_convs =        down_convs,
        skip_convs =        skip_convs,
        up_convs =          up_convs,
        batchnorm =         batchnorm,
        last_batchnorm =    last_batchnorm,
        activation =        activation,
        last_activation =   last_activation,
        upsample_args =     upsample_args
        )

    # print(model)
    x = torch.randn(1, 3, 256, 384)
    y = model(x)
    print(x.shape)
    print(y.shape)

