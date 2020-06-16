import numpy as np
from scipy.fftpack import fft2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
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


def image_init(A, h, w, N1, N2, n1, n2, noise_mean, noise_std):
    x = np.zeros((N1, N2))
    x[n1: n1+h, n2: n2+w] = A
    x, _ = add_gnoise2d(x, noise_mean, noise_std)
    return x

def image_processing(x1, x2):
    X1, X2 = ff2(x1), ff2(x2)
    x3 = phase_corr(X1, X2)
    x3 = np.abs(x3)
    e1, e2 = np.where(x3 == np.max(x3))
    return x3, (e1, e2)


# Image Initialization
## amplitude, span, and initial coords of non-zero values in image
A = 5
h, w = 5, 5

## span of full image
N1, N2 = 50, 50

## noise characteristics
noise_mean = 0
noise_std = 0.4

## init images with zeros, non-zeros, then add noise
x1 = np.zeros((N1, N2))
x1[0: 0+h, 0: 0+w] = A
x1, _ = add_gnoise2d(x1, noise_mean, noise_std)

x2 = np.zeros((N1, N2))
x2[0: 0+h, 0: 0+w] = A
x2, _ = add_gnoise2d(x2, noise_mean, noise_std)

# Image Processing
## DFT of images
X1, X2 = fft2(x1), fft2(x2)

## phase correlation, time-domain impulse peaks
x3 = phase_corr(X1, X2)
x3 = np.abs(x3)
e1, e2 = np.where(x3 == np.max(x3))

# Initial Plots
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)
[ax0, ax1, ax2] = axes

## to keep contrast between both images consistent
vmin = min(np.min(x1), np.min(x2))
vmax = max(np.max(x1), np.max(x2))

## extent to properly center imshow pixels
img1 = ax0.imshow(x1, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='binary', vmin=vmin, vmax=vmax)
img2 = ax1.imshow(x2, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='binary', vmin=vmin, vmax=vmax)
img3 = ax2.imshow(x3, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='Reds')

## titles, ticks, labels
fig.suptitle("Motion Estimation via Phase Correlation")
ax0.set_title("Original Image", fontsize=10)
ax1.set_title(f"Shifted Image\n Actual Shift: {0, 0}", fontsize=10)
ax2.set_title(f"Motion Estimate\n Estimated Shift: {e1[0], e2[0]}", fontsize=10)
ax1.tick_params(left=False)
ax2.tick_params(left=False, right=True, labelright=True)

# Collect data for easier indexing
data1 = {'X': X1, 'pos': (0, 0), 'img': img1}
data2 = {'X': X2, 'pos': (0, 0), 'img': img2}
data3 = {'X': None, 'pos': (0, 0), 'img': img3}
datas = {ax0: data1, ax1: data2, ax2: data3}

# Plot update functions
clicking = False

## true if LMB click
def mouse_click(event):
    global clicking
    if event.button == 1:
        clicking = True

## false if LMB release
def mouse_release(event):
    global clicking
    if event.button == 1:
        clicking = False

## if clicking and in first two axes, update images, motion estimation, then plot
def mouse_move(event):
    x, y = event.xdata, event.ydata
    ax = event.inaxes

    if (clicking and ax in axes[:2]):

        # get new image coords
        n1, n2 = int(y), int(x)
        datas[ax]['pos'] = (n1, n2)

        # Image Initialization
        ## create new noisy image and compute DFT
        x = np.zeros((N1, N2))
        x[n1: n1+h, n2: n2+w] = A
        x, snr = add_gnoise2d(x, noise_mean, noise_std)

        # Image Processing
        ## DFT of image
        X = fft2(x)
        datas[ax]['X'] = X

        ## phase correlation, time-domain impulse peaks
        x3 = phase_corr(datas[ax0]['X'], datas[ax1]['X'])
        x3 = np.abs(x3)
        e1, e2 = np.where(x3 == np.max(x3))
        datas[ax2]['pos'] = (e1[0], e2[0])

        # update axes
        datas[ax]['img'].set_data(x)
        datas[ax2]['img'].set_data(x3)
        ax0.set_title("Original Image", fontsize=10)
        ax1.set_title(f"Shifted Image\n Actual Shift: {tuple(np.subtract(datas[ax1]['pos'], datas[ax0]['pos']))}", fontsize=10)
        ax2.set_title(f"Motion Estimate\n Estimated Shift: {datas[ax2]['pos']}", fontsize=10)
        fig.canvas.draw_idle()


plt.connect('motion_notify_event', mouse_move)
plt.connect('button_press_event', mouse_click)
plt.connect('button_release_event', mouse_release)
plt.show(block=True)
