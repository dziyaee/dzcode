import torch
import torch.nn as nn
import numpy as np


def modules(net):
    [(print(module), print("-" * 100)) for module in net.modules()]
    return None

def children(net):
    [(print(name, child), print("-" * 100)) for name, child in net.named_children()]
    return None

def forwardpass(net):
    ninput = torch.randn(1, 3, 257, 384)
    noutput = net(ninput)
    print(f"Input Shape:  {ninput.shape}")
    print(f"Output Shape: {noutput.shape}")
    return None

def params(net):
    n_params = np.sum([np.prod(params.size()) for params in net.parameters()])
    return n_params

def shapes(net):
    [print(params.size()) for params in net.parameters()]
    return None


# Examples
if __name__ == "__main__":
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.BatchNorm2d(16),
        nn.ReLU()
        )

    print(f"\nmodules output:")
    modules(net)
    print(f"\nchildren output:")
    children(net)
    print(f"\nforwardpass output:")
    forwardpass(net)
    print(f"\nparams output:")
    n_params = params(net)
    print(n_params)
    print(f"\nshapes output:")
    shapes(net)





