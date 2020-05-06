import torch
import torch.nn as nn
import numpy as np


# Quick Info for nn.Modules
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

# Auto Padding for 'Valid' Convolution-Like Operations to ensure 'same' output size
# Method 1: p = calc_padding(x, k, s, d)
def calc_padding(x, k, s, d):
    """Function to calculate the amount of 'padding' needed to ensure the same final output size as the starting input size 'x' after 'd' downsampling convolutions and a subsequent 'd' upsampling convolutions of kernel size 'k' and stride 's'.

    Note: This operation can be achieved by using the final_dim function, taking the ceiling of the output, and then using the valid_dim function. The other functions are easier to understand.

    Args:
        x (int): Initial input size
        k (int): kernel size
        s (int): stride
        d (int): number of downsampling convolutions, which is equal to the number of upsampling convolutions

    Returns:
        padding (int): amount of padding to add to the initial input size

    """

    xmin_spacing = (s ** d - 1) / (s - 1)
    xmin = 1 + (k - 1) * xmin_spacing

    n = np.ceil((x - xmin) / s ** d)
    validx = xmin + n * s ** d

    newx = max(xmin, validx)
    padding = int(newx - x)
    return padding

# Method 2: y = final_dim(x, k, s, n), y = np.ceil(y), x = valid_dim(y, k, s, n)
def final_dim(x, k, s, n):
    """Formula to calculate the final output size "y" after "n" valid convolution operations defined by a kernel size "k" and a stride "s" on an input of size "x":

    y = (x + (s - k) * Sum from (d = 0) to (d = n-1) of s^d) / s^n

    Args:
        x (int): Initial input size
        k (int): kernel size
        s (int): stride
        n (int): number of convolution operations

    Returns:
        y (float): Final output size

        Note:
            floor(y) will give the actual output size after valid convolutions
            ceil(y) will give the next valid output size and can be used to calculate the next valid input size
    """
    # Formula to calculate the sum of powers, defined as sum from i=0 to i=n-1 of s^i
    sum_of_powers = lambda s, n: np.sum(s ** np.arange(0, n))
    sop = sum_of_powers(s, n)
    y = (x + (s - k) * (sop)) / s ** n

    return y

def valid_dim(y, k, s, n):
    """Formula to calculate the initial input size "x" that will result in a final output size "y" after "n" valid convolution operations defined by a kernel size "k" and a stride "s". A valid convolution operation is one where no padding is added to the input along any dimension.

    For n = 1:
        x = (s^n * (y - 1)) + (k * Sum from (d = 0) to (d = n-1) of s^d)

    For n > 1:
        x = (s^n * (y - 1)) + (k * Sum from (d = 0) to (d = n-1) of s^d) - (s * Sum from (d = 0) to (d = n-2) of s^d)

    Args:
        y (int): final dimension y. If not a whole number, ceiling(y) will result in the next valid integer y being used for the calculation. This will give the next valid x
        k (int): kernel size
        s (int): stride
        n (int): number of valid convolution operations

    Returns:
        x (int): first valid initial input size x
    """

    # Formula to calculate the sum of powers, defined as sum from i=0 to i=n-1 of s^i
    sum_of_powers = lambda s, n: np.sum(s ** np.arange(0, n))
    sop1 = sum_of_powers(s, n)

    if n == 1:
        x = ((s ** n) * (y - 1)) + (k * sop1) - (s*0)
    else:
        sop2 = sum_of_powers(s, n-1)
        x = ((s**n) * (y - 1)) + (k * sop1) - (s * sop2)

    return x
