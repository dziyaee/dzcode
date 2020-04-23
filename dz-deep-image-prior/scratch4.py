import torch
import torch.nn as nn
import numpy as np
from utils.helper import info
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Concat(nn.Module):

    def __init__(self, dim=1, *modules):
        super(Concat, self).__init__()

        self.dim = dim
        for i, module in enumerate(modules):
            self.addmod(module)

    def forward(self, input):

        outputs = []
        for module in self._modules.values():
            outputs.append(module(input))

        heights = []
        widths = []
        for output in outputs:
            print(output.shape)
            heights.append(output.shape[2])
            widths.append(output.shape[3])

        # Check if all elements in heights are equal, and if all elements in widths are equal
        if (len(set(heights)) == 1) and (len(set(widths)) == 1):
            outputs_ = outputs

        else:

            min_height, min_width = min(heights), min(widths)
            outputs_ = []

            for output in outputs:
                height_edge = (output.shape[2] - min_height) // 2
                width_edge = (output.shape[3] - min_width) // 2

                outputs_.append(output[:, :, height_edge: height_edge + min_height, width_edge: width_edge + min_width])

        return torch.cat(outputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)
# ----------------------------------------------------------------------------


def addmod(self, module):

    n = len(self) + 1
    self.add_module(str(n), module)


torch.nn.Module.addmod = addmod
# -----------------------------------------------------------------------------


def addblock(self, outs, ins, size, stride, pad, bn=True, act=True):

    self.addmod(nn.ReflectionPad2d(pad))
    self.addmod(nn.Conv2d(in_channels=ins, out_channels=outs, kernel_size=size, stride=stride))

    if bn == True:
        self.addmod(nn.BatchNorm2d(outs))

    if act == True:
        self.addmod(nn.LeakyReLU(0.2))


torch.nn.Module.addblock = addblock
# -------------------------------------------------------------------------

# net = nn.Sequential()
# temp = net
# deeper = nn.Sequential()

x = torch.randn(1, 3, 10, 10)

input_depth = x.shape[1]
Nd = [8]
Nu = [8]
Ns = [4]

for i in range(len(Nd)):

    temp = nn.Sequential()
    skiplayer = nn.Sequential()
    deeplayer = nn.Sequential()

    assert len(Nd) == len(Nu) == len(Ns), "Length of Number of down/up/skip kernel lists must be equal"

    if i == 0:
        first = True
    else:
        first = False

    if i == len(Nd) - 1:
        last = True
    else:
        last = False

    if Ns[i] == 0:
        skip = False
    else:
        skip = True

    if first:
        ins = input_depth

    if skip:
        skiplayer.addblock(outs=Ns[i], ins=ins, size=1, stride=1, pad=0, bn=False, act=False)
        deeplayer.addblock(outs=Nd[i], ins=ins, size=3, stride=2, pad=1, bn=True, act=True)
        deeplayer.addblock(outs=Nd[i], ins=Nd[i], size=3, stride=1, pad=1, bn=True, act=True)
        deeplayer.addmod(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        temp.addmod(Concat(1, skiplayer, deeplayer))

    else:
        deeplayer.addblock(outs=Nd[i], ins=ins, size=3, stride=2, pad=1, bn=True, act=True)
        deeplayer.addblock(outs=Nd[i], ins=Nd[i], size=3, stride=1, pad=1, bn=True, act=True)
        deeplayer.addmod(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        temp.addmod(deeplayer)


print(temp)
print("-" * 100)
info(x, "Net Input")
print("-" * 30)
y = temp(x)
print("-" * 30)
info(y, "Net Output")
print("-" * 100)

# # --------------------------------------------------------------------------------
# # Downsample Block
# # 2D, H/2, W/2
# temp.addblock(outs=8, ins=3, size=3, stride=2, pad=1)
# # D, H, W
# temp.addblock(outs=8, ins=8, size=3, stride=1, pad=1)

# # If not last Architecture Depth, add 'deeper' Sequential Layer
# if last == False:
#     temp.addmod(deeper)

# # Upsample Block:
# # D, 2H, 2W
# temp.addmod(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
# # Takes current IN channels
# temp.addmod(nn.BatchNorm2d(16))
# # D/2, H, W
# temp.addblock(outs=8, ins=16, size=3, stride=1, pad=1)
# # D, H, W
# temp.addblock(outs=8, ins=8, size=1, stride=1, pad=0)

# # Final block (outside of loop) to match net input channels + final activation
# net.addblock(outs=3, ins=8, size=1, stride=1, pad=0, bn=False, act=False)
# net.addmod(nn.Sigmoid())

# # Test, add something to 'deeper' layer
# deeper.addmod(nn.Sigmoid())
# # print(net)

