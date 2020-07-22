import numpy as np
from numpy import log10
from scipy.fft import ifft, ifft2


def window2d(window1d):
    """Function to create a 2d window from a 1d window by computing the outer product of the 1d window with itself

    Args:
        window1d (Numpy Array): 1d window

    Returns:
        window (Numpy Array): 2d window
    """
    window = np.outer(window1d, window1d)
    return window


def phase_correlation(X1, X2, eps=1e-6):
    """Function to compute the phase correlation between two DFT input arrays:
    1) Cross-Power Spectrum is calculated by the element-wise multiplication (Hadamard Product) of the first array and the conjugate of the second.
    2) Cross-Power Spectrum is normalized by dividing by it's own absolute value. This normalizes the magnitude to 1
    3) The Normalized Cross-Correlation is computed by taking the Inverse Fourier Transform of the Normalized Cross-Power Spectrum

    Args:
        X1, X2 (Numpy Array): two input arrays representing the DFTs of two signals

    Returns:
        x3: Normalized Cross-Correlation of X1 and X2
    """
    ifftfunc = (ifft, ifft2)
    X3 = np.multiply(X1, np.conj(X2))
    X3 = X3 / (np.abs(X3) + eps)
    x3 = ifftfunc[X3.ndim-1](X3)
    return x3


def add_gaussian_noise(x, noise_mean=0, noise_std=1, return_snr=False):
    """Function to add Gaussian noise with a specified mean and std to an array. Option to return snr along with noisy array

    Args:
        x (Numpy Array): input array
        noise_mean (float): Gaussian noise mean
        noise_std (float): Gaussian noise std
        return_snr (bool, optional): option to calculate and return SNR along with noisy array

    Returns:
        noisy_x (Numpy Array): input array with added Gaussian noise
        snr (float, optional): SNR of input array and noise
    """
    if noise_std > 0:
        noise = noise_mean + noise_std * np.random.randn(*x.shape)
        noisy_x = x + noise
        if return_snr:
            snr = 10 * log10(np.mean(x ** 2) / np.mean(noise ** 2))
            return noisy_x, snr
        return noisy_x


def im2col(xdims, kdims, sdims):
    # standardize input shape sizes to 3 to represent channel, height, width dims
    dims = [xdims, kdims, sdims]
    size = 3
    new = []
    for dim in dims:
        dim = np.asarray(dim)
        diff = size - dim.size
        value = [1] * diff
        dim = np.insert(dim, 0, value)
        new.append(dim)

    xdim, kdim, sdim = new

    # channel, height, width dimensions of input, window, stride
    xc, xh, xw = xdim
    kc, kh, kw = kdim
    _, sh, sw = sdim

    # first window index vector
    deps = np.array(np.arange(kc), ndmin=2) * xh * xw
    rows = np.array(np.arange(kh), ndmin=2) * xw
    cols = np.array(np.arange(kw), ndmin=2)
    window = np.array((deps.T + rows).ravel(), ndmin=2)
    window = np.array((window.T + cols).ravel(), ndmin=2)

    # number of windows along rows and cols
    nh = int(np.floor((xh - kh) / sh) + 1)
    nw = int(np.floor((xw - kw) / sw) + 1)

    # index offset vector
    rows = np.array(np.arange(nh), ndmin=2) * sh * xw
    cols = np.array(np.arange(nw), ndmin=2) * sw
    offset = np.array((rows.T + cols).ravel(), ndmin=2)

    # add offset to window via broadcasting to create final indices
    indices = window.T + offset
    return indices

# inputs
input_ = np.random.randint(0, 9, (2, 6, 6))
kernel = np.random.randint(0, 1, (2, 3, 3))
stride = (2, 2)

indices = im2col(input_.shape, kernel.shape, stride)
output = np.take(input_, indices)
print(input_)
print(output)

