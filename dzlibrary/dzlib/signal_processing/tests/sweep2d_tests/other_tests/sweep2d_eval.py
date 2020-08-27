import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal as sp
from dzlib.signal_processing.sweep2d import Sweep2d


def generate_common(settings):
    # generate limits of each input argument
    unpadded, window, padding, stride = settings['input_arg_limits'].values()
    xlims = tuple((min, max+1) for min, max in unpadded.values())
    klims = tuple((min, max+1) for min, max in window.values())
    plims = tuple((min, max+1) for min, max in padding.values())
    slims = tuple((min, max+1) for min, max in stride.values())

    # generate dimensions from limits
    n = settings['n_tests']
    xdims = list(list(np.random.randint(min, max, n, dtype=np.int32)) for min, max in xlims)
    kdims = list(list(np.random.randint(min, max, n, dtype=np.int32)) for min, max in klims)
    pdims = list(list(np.random.randint(min, max, n, dtype=np.int32)) for min, max in plims)
    sdims = list(list(np.random.randint(min, max, n, dtype=np.int32)) for min, max in slims)

    return xdims, kdims, pdims, sdims


def generate_vs_torch(settings):
    xdims, kdims, pdims, sdims = generate_common(settings)

    # vs torch args shapes
    kdims[1] = xdims[1]  # overwrite: set kernel depth equal to unpadded depth
    xshapes = [dims for dims in zip(*xdims)]
    kshapes = [dims for dims in zip(*kdims)]
    pshapes = [dims for dims in zip(*pdims)]
    sshapes = [dims for dims in zip(*sdims)]

    return xshapes, kshapes, pshapes, sshapes


def generate_vs_scipy(settings):
    xdims, kdims, pdims, sdims = generate_common(settings)
    n = settings['n_tests']

    # vs scipy args shapes
    xdims[0] = kdims[0] = [1] * n  # overwrite: set unpadded kernel and unpadded nums to 1
    xdims[1] = kdims[1] = [1] * n  # overwrite: set unpadded kernel and unpadded depths to 1
    pdims[0] = pdims[1] = [0] * n  # overwrite: padding is 0, but doesn't actually matter what gets passed here
    sdims[0] = sdims[1] = [1] * n  # overwrite: stride is 1, but doesn't actually matter what gets passed here
    xshapes = [dims for dims in zip(*xdims)]
    kshapes = [dims for dims in zip(*kdims)]
    pshapes = [dims for dims in zip(*pdims)]
    sshapes = [dims for dims in zip(*sdims)]

    return xshapes, kshapes, pshapes, sshapes


def errors(func):
    def wrap(precision, *args, **kwargs):
        actual, desired = func(*args, **kwargs)
        errors = np.abs(desired - actual)
        indices = np.where(errors > 1.5 * 10 ** (-precision))
        return actual[indices], desired[indices]

    func.errors = wrap
    return func


@errors
def eval_vs_torch(data, mode, seed=None):
    # mode argument doesn't get used here at all
    inputs_shape, kernels_shape, padding, stride = data

    # generate arrays
    if seed:
        np.random.seed(seed)
    inputs_np = np.random.randn(*inputs_shape).astype(np.float32)
    kernels_np = np.random.randn(*kernels_shape).astype(np.float32)
    inputs_pt = torch.from_numpy(inputs_np).type(torch.FloatTensor)
    kernels_pt = torch.from_numpy(kernels_np).type(torch.FloatTensor)

    # sweeper
    sweep2d = Sweep2d(inputs_shape, kernels_shape, padding, stride, mode)
    sweep_outputs_np = sweep2d.correlate2d(inputs_np, kernels_np)

    # pytorch
    torch_outputs_pt = F.conv2d(inputs_pt, kernels_pt, None, stride, padding)
    torch_outputs_np = np.asarray(torch_outputs_pt)

    return sweep_outputs_np, torch_outputs_np


@errors
def eval_vs_scipy(data, mode, seed=None):
    inputs_shape, kernels_shape, padding, stride = data

    # generate arrays
    if seed:
        np.random.seed(seed)
    inputs_np = np.random.randn(*inputs_shape).astype(np.float32)
    kernels_np = np.random.randn(*kernels_shape).astype(np.float32)

    # sweeper
    sweep2d = Sweep2d(inputs_shape, kernels_shape, padding, stride, mode)
    sweep_outputs_np = sweep2d.convolve2d(inputs_np, kernels_np)

    # scipy
    i = [0] * (inputs_np.ndim - 2) + [...]  # index into last two dimensions for scipy inputs and sweep output
    i = tuple(i)  # convert indices list to tuple due to list indices behavior changing in the future
    scipy_outputs_np = sp.convolve2d(inputs_np[i], kernels_np[i], mode)

    return sweep_outputs_np[i], scipy_outputs_np
