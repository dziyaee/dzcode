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
    # Calculate the cross-power spectrum, which is given by the Hadamard (or element-wise) product of x1 and the complex conjugate of x2
    # Note: An epsilon value is used here to deal with divide-by-zero
    cps = np.multiply(X1, np.conj(X2)) + eps

    # Normalize the cross-power spectrum by dividing it with it's own magnitude
    ncps = cps / np.abs(cps)

    # Calculate the normalized cross-correlation by computing the Inverse DFT
    ncc = (1 / N1 * N2) * fft2(ncps)
    x3 = np.abs(ncc)
    return x3


def add_gnoise2d(x, *args):
    N1, N2 = x.shape
    mean, std = 0, 1
    if args:
        mean, std = args
    noise = mean + std * np.random.randn(N1, N2)
    snr = np.mean(x ** 2) / np.mean(noise ** 2)
    x += noise
    return x, snr


# Image information
# A: non-zero amplitude
# N1, N2: number of rows, cols
# n1, n2 non-zero starting indices for row, col
# h, w: non-zero height, width
# k1, k2: time-shift values for row, col
A = 5
N1, N2 = 50, 50
n1, n2 = 0, 0
x1n1, x1n2 = n1, n2
h, w = 5, 5
k1, k2 = 0, 0
x2n1, x2n2 = n1 + k1, n2 + k2
zeros = np.zeros((N1, N2))
noise_mean = 0
noise_std = 0.4

# Image init; starting image, x1; translated image, x2
x1 = np.zeros((N1, N2))
x1[n1: n1+h, n2: n2+w] = A
x1, _ = add_gnoise2d(x1, noise_mean, noise_std)

x2 = np.zeros((N1, N2))
x2[n1+k1: n1+k1+h, n2+k2: n2+k2+w] = A
x2, _ = add_gnoise2d(x2, noise_mean, noise_std)

# DFT of Initial Image x1, X1; DFT of Translated Image x2, X2
X1, X2 = fft2(x1), fft2(x2)

# Phase Correlation Time-Domain impulse
x3 = phase_corr(X1, X2)
e1, e2 = np.where(x3 == np.max(x3))

# Initial Plot
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)
[ax1, ax2, ax3] = axes

vmin = min(np.min(x1), np.min(x2))
vmax = max(np.max(x1), np.max(x2))

img1 = ax1.imshow(x1, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='binary', vmin = vmin, vmax = vmax)
img2 = ax2.imshow(x2, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='binary', vmin = vmin, vmax = vmax)
img3 = ax3.imshow(x3, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='Reds')

ax1.set_title("Original Image", fontsize=10)
ax2.set_title(f"Shifted Image\n Actual Shift: {k1, k2}", fontsize=10)
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
global ax
ax = None

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
        x = zeros.copy()
        x[n1: n1+h, n2: n2+w] = A

        x, _ = add_gnoise2d(x, noise_mean, noise_std)
        X = fft2(x)

        # update DFT list and compute phase correlation time-domain impulse
        X_list[i] = X
        x3 = phase_corr(*X_list)
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
