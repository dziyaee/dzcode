import numpy as np
from numpy import log10
from scipy.fftpack import fft2
import matplotlib as mpl
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
from dzlib.common.utils import info, stats
mpl.use("Qt5Agg")


eps = 1e-6
def phase_corr(X1, X2):
    """
    Function to calculate the phase correlation between two 2d signals
    The power spectrum of a signal can be calculated by multiplying the signal DFT by it's own complex conjugate. In doing so, the phases of each component cancel out, while the magnitudes multiply together to give the power.
    As a matrix operation, the above multiplication is an element-wise multiplication, otherwise known as the Hadamard Product.
    In order to obtain the phase shift resulting from a time shift of a signal, the cross-power spectrum (cps) is used. In this case, there are two signals, the original signal DFT (X1), and the time-shifted signal DFT (X2). The cross-power spectrum is given by the Hadamard Product of the original signal DFT and the complex conjugate of the time-shifted signal DFT. The resulting phase component of the output is due to the time shift of the signal, and the magnitudes once again multiply together to give the power.
        Note: An epsilon value (eps) is used here to deal with divide-by-zero errors. The magnitude of the eps may need to be adjusted to keep it very small in comparison to the signal magnitudes

    Args:
        X1 (TYPE): Description
        X2 (TYPE): Description

    Returns:
        TYPE: Description
    """
    N1, N2 = X1.shape

    # cross-power spectrum
    cps = np.multiply(X1, np.conj(X2)) + eps

    # normalized cross-power spectrum
    ncps = cps / np.abs(cps)

    # normalized cross-correlation
    ncc = (1 / N1 * N2) * fft2(ncps)
    return ncc


def add_gnoise2d(x, *args):
    """Summary

    Args:
        x (TYPE): Description
        *args: Description

    Returns:
        TYPE: Description
    """
    N1, N2 = x.shape
    mean, std = 0, 1
    if args:
        mean, std = args
    noise = mean + std * np.random.randn(N1, N2)
    snr = np.mean(x ** 2) / np.mean(noise ** 2)
    x += noise
    return x, snr


# amplitude, span, and initial coords of non-zero values in image
A = 5
h, w = 5, 5
global x1n1, x1n2, x2n1, x2n2
x1n1, x1n2 = 0, 0
x2n1, x2n2 = 0, 0

# full image span
global N1, N2
N1, N2 = 50, 50

# noise characteristics
noise_mean = 0
noise_std = 0.4

# init images with zeros, non-zeros, then add noise
x1 = np.zeros((N1, N2))
x1[0: 0+h, 0: 0+w] = A
x1, _ = add_gnoise2d(x1, noise_mean, noise_std)

x2 = np.zeros((N1, N2))
x2[0: 0+h, 0: 0+w] = A
x2, _ = add_gnoise2d(x2, noise_mean, noise_std)

# DFT of images
X1, X2 = fft2(x1), fft2(x2)

# phase correlation, time-domain impulse peaks
x3 = phase_corr(X1, X2)
x3 = np.abs(x3)
e1, e2 = np.where(x3 == np.max(x3))

# Initial Plot
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)
[ax1, ax2, ax3] = axes

vmin = min(np.min(x1), np.min(x2))
vmax = max(np.max(x1), np.max(x2))

img1 = ax1.imshow(x1, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='binary', vmin=vmin, vmax=vmax)
img2 = ax2.imshow(x2, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='binary', vmin=vmin, vmax=vmax)
img3 = ax3.imshow(x3, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='Reds')

fig.suptitle("Motion Estimation via Phase Correlation")
ax1.set_title("Original Image", fontsize=10)
ax2.set_title(f"Shifted Image\n Actual Shift: {x2n1-x1n1, x2n2-x1n2}", fontsize=10)
ax3.set_title(f"Motion Estimate\n Estimated Shift: {e1[0], e2[0]}", fontsize=10)

ax2.tick_params(left=False)
ax3.tick_params(left=False, right=True, labelright=True)

# Data for plot updates
axes_dict = {ax1: 0, ax2: 1}
X_list = [X1, X2]
img_list = [img1, img2]

# Plot updates
global clicking
clicking = False

def mouse_click(event):
    global clicking
    global ax
    if event.inaxes is None:
        return
    if event.button != 1:
        return
    clicking = True
    # ax = event.inaxes


def mouse_release(event):
    global clicking
    if event.inaxes is None:
        return
    if event.button != 1:
        return
    clicking = False


def mouse_move(event):
    global x1n1, x1n2
    global x2n1, x2n2

    x, y = event.xdata, event.ydata

    if (clicking and event.inaxes in axes_dict):

        # get axis index
        i = axes_dict[event.inaxes]

        # get new image coords
        n1, n2 = int(y), int(x)

        if i == 0:
            x1n1, x1n2 = n1, n2
        elif i ==1:
            x2n1, x2n2 = n1, n2

        # create image and compute DFT
        x = np.zeros((N1, N2))
        x[n1: n1+h, n2: n2+w] = A

        x, _ = add_gnoise2d(x, noise_mean, noise_std)
        X = fft2(x)

        # update DFT list and compute phase correlation time-domain impulse
        X_list[i] = X
        x3 = phase_corr(*X_list)
        x3 = np.abs(x3)
        e1, e2 = np.where(x3 == np.max(x3))

        # update image plots
        img_list[i].set_data(x)
        img3.set_data(x3)

        # Other plot stuff
        ax1.set_title("Original Image", fontsize=10)
        ax2.set_title(f"Shifted Image\n Actual Shift: {x2n1-x1n1, x2n2-x1n2}", fontsize=10)
        ax3.set_title(f"Motion Estimate\n Estimated Shift: {e1[0], e2[0]}", fontsize=10)

        # update axis info
        # ax3.set_title(f"k1 = {k1[0]}, k2 = {k2[0]}")
        fig.canvas.draw_idle()


plt.connect('motion_notify_event', mouse_move)
plt.connect('button_press_event', mouse_click)
plt.connect('button_release_event', mouse_release)
plt.show(block=True)
