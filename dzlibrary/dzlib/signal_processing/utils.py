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

