# Imports
# general
import numpy as np
from typing import NamedTuple

# scipy
from scipy.fftpack import fft2

# dzlib
from dzlib.common.devices import MouseButton
from dzlib.signal_processing.signals import Signal
from dzlib.signal_processing.utils import phase_correlation

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec


class FlatImage(NamedTuple):
    A: float
    shape: tuple

class BlockImage():
    def __init__(self, A1, A2, shape1, shape2, xycoords, noise):
        self.main = FlatImage(A1, shape1)
        self.block = FlatImage(A2, shape2)
        self.x, self.y = xycoords
        self.noise_mean, self.noise_std = noise
        self._set_image()

    def _set_image(self):
        # extract attributes
        x, y = self.x, self.y
        h, w = self.block.shape
        H, W = self.main.shape
        A1, A2 = self.main.A, self.block.A
        noise_mean, noise_std = self.noise_mean, self.noise_std

        # create clean image
        image = np.full((H, W), A1).astype(np.float32)
        image[y: y+h, x: x+w] = A2
        snr = float('inf')

        # add noise if valid
        if noise_std > 0:
            noise = noise_mean + noise_std * np.random.randn(H, W)
            snr = 10 * np.log10(np.mean(image ** 2) / np.mean(noise ** 2))
            image += noise

        # create / update attributes
        self.image = image
        self.snr = snr
        self.fft = fft2(self.image)

    def set_main(self, A, shape):
        self.main = FlatImage(A, shape)
        self._set_image()

    def set_block(self, A, shape):
        self.block = FlatImage(A, shape)
        self._set_image()

    def set_xy(self, xycoords):
        self.x, self.y = xycoords
        self._set_image()

    def set_noise(self, noise=(0, 1)):
        self.noise_mean, self.noise_std = noise
        # self.noise_mean = noise_mean
        # self.noise_std = noise_std
        self._set_image()


# general params
global eps
eps = 1e-6

# main params
N1, N2 = 50, 50 # main span
noise_mean = 0
noise_std = 1

# block params
A = 5 # amplitude
h, w = 5, 5 # shape
n1, n2 = 0, 0 # xy coords1
m1, m2 = 0, 0 # xy coords2

# init images
x1 = BlockImage(A1=0, A2=A, shape1=(N1, N2), shape2=(h, w), xycoords=(n1, n2), noise=(noise_mean, noise_std))
x2 = BlockImage(A1=0, A2=A, shape1=(N1, N2), shape2=(h, w), xycoords=(m1, m2), noise=(noise_mean, noise_std))

# image processing
x3 = phase_correlation(x2.fft, x1.fft, eps)
x3 = Signal(x3)

# calculate shifts
dx = x2.x - x1.x
dy = x2.y - x1.y
edy, edx = np.where(x3.real == x3.max)
edx, edy = edx[0], edy[0]

# initial plots
fig = plt.figure()
gs = GridSpec(nrows=1, ncols=3)
ax1 = plt.subplot(gs[0, 0:1])
ax2 = plt.subplot(gs[0, 1:2])
ax3 = plt.subplot(gs[0, 2:3])
fig.subplots_adjust(wspace=0, top=0.85, hspace=0.4)

## extent to properly center imshow pixels
img1 = ax1.imshow(x1.image.real, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='binary', vmin=0, vmax=x1.block.A)
img2 = ax2.imshow(x2.image.real, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='binary', vmin=0, vmax=x2.block.A)
img3 = ax3.imshow(x3.real, extent=[0, N2, 0, N1], origin='lower', interpolation='None', cmap='Reds', vmin=x3.min, vmax=x3.max)

## titles, ticks, labels
fig.suptitle("Motion Estimation via Phase Correlation")
ax1.set_title(f"Original Image", fontsize=10)
ax2.set_title(f"Shifted Image\nActual Shift: {dx, dy}", fontsize=10)
ax3.set_title(f"Motion Estimate\nEstimated Shift: {edx, edy}", fontsize=10)

ax1.tick_params(left=False, right=False, top=False, bottom=False, labelbottom=False, labelleft=False)
ax2.tick_params(left=False, right=False, top=False, bottom=False, labelbottom=False, labelleft=False)
ax3.tick_params(left=False, right=True, labelright=True, labelleft=False)

# slider for noise std
left = 0.15
bot = 0.01
width = 0.7
height = 0.03
slider1 = plt.axes([left, bot, width, height])
noise_std_var = Slider(ax=slider1, label='Noise STD', valmin=0, valmax=A, valinit=noise_std, valstep=0.01, valfmt='%.2f')

# axes to data mapping
axes_dict = {ax1: [x1, img1], ax2: [x2, img2]}

LMB = MouseButton(isdown=False)

def mouse_click(event, LMB):
    if event.button == 1:
        LMB.isdown = True

def mouse_release(event, LMB):
    if event.button == 1:
        LMB.isdown = False

def mouse_move(event, LMB):
    ax = event.inaxes
    if (LMB.isdown and ax in axes_dict.keys()):
        m1, m2 = int(event.xdata), int(event.ydata)
        update(ax, (m1, m2), None)

def update(ax, xycoords=None, noise=None):

    # get objects in current axis
    x, img = axes_dict[ax]

    # update BlockImage
    if xycoords is not None:
        x.set_xy(xycoords)
    if noise is not None:
        x.set_noise(noise)

    # update motion estimation image
    x3 = phase_correlation(x2.fft, x1.fft, eps)
    x3 = Signal(x3)

    # calculate shifts
    dx = x2.x - x1.x
    dy = x2.y - x1.y
    edy, edx = np.where(x3.real == x3.max)
    edx, edy = edx[0], edy[0]

    # update axes
    img.set_data(x.image)
    img3.set_data(x3.real)
    ax.set_xlabel(f"SNR: {x.snr:.4f}")
    ax2.set_title(f"Shifted Image\nActual Shift: {dx, dy}", fontsize=10)
    ax3.set_title(f"Motion Estimate\nEstimated Shift: {edx, edy}", fontsize=10)

    fig.canvas.draw_idle()

def slider_move(val):
    noise_std = noise_std_var.val
    noise = (noise_mean, noise_std)
    update(ax1, None, noise)
    update(ax2, None, noise)

noise_std_var.on_changed(slider_move)
plt.connect('motion_notify_event', lambda event: mouse_move(event, LMB))
plt.connect('button_press_event', lambda event: mouse_click(event, LMB))
plt.connect('button_release_event', lambda event: mouse_release(event, LMB))
plt.show(block=True)
