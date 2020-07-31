# Imports
# general
import os
from PIL import Image
import numpy as np

# scipy
from scipy.fftpack import fft2, fftshift

# dzlib
from dzlib.common.devices import MouseButton
from dzlib.signal_processing.signals import Signal
from dzlib.signal_processing.utils import phase_correlation, add_gaussian_noise

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# starting params
global eps
eps = 1e-6
N1, N2 = 400, 400 # scene span (rows, cols) or (height, width)
noise_mean = 0
noise_std = 20

# init scene
scene_path = '/'.join((os.getcwd(), 'data/image2.png'))
scene = Image.open(scene_path).convert('L')
scene = np.asarray(scene.resize((N1, N2))).astype(np.float32)
scene = np.flip(scene, axis=(0)) # flip to account for imshow(origin='lower')

# init noisy images
x1, snr1 = add_gaussian_noise(x=scene.copy(), noise_mean=noise_mean, noise_std=noise_std, return_snr=True)
x2, snr2 = add_gaussian_noise(x=scene.copy(), noise_mean=noise_mean, noise_std=noise_std, return_snr=True)
x1 = Signal(x1)
x2 = Signal(x2)

# image processing
X1, X2 = fft2(x1.arr), fft2(x2.arr)
x3 = phase_correlation(X2, X1)
x3 = fftshift(x3)
x3 = Signal(x3)

# calculate shifts
dx, dy = 0, 0 # actual shift
edy, edx = [int(x) for x in np.where(x3.real == x3.max)] # estimated shift

# initial plot
fig = plt.figure()
fig.subplots_adjust(wspace=0)
gs = GridSpec(nrows=1, ncols=3)
ax1 = plt.subplot(gs[0, 0:1])
ax2 = plt.subplot(gs[0, 1:2])
ax3 = plt.subplot(gs[0, 2:3])

img1 = ax1.imshow(x1.real, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=x1.min, vmax = x1.max)
img2 = ax2.imshow(x2.real, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=x2.min, vmax=x2.max)
img3 = ax3.imshow(x3.real, extent=[-N1/2, N1/2, -N2/2, N2/2], origin='lower', interpolation=None, cmap='Reds', vmin=x3.min, vmax=x3.max)

for ax in [ax1, ax2, ax3]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

LMB = MouseButton(isdown=False)

def mouse_click(event, LMB):
    if event.button == 1:
        LMB.isdown = True

def mouse_release(event, LMB):
    if event.button == 1:
        LMB.isdown = False

def mouse_move(event, LMB):
    x, y = event.xdata, event.ydata
    ax = event.inaxes
    if (LMB.isdown and ax == ax2):
        update(x, y)


def update(x, y):
    global x2_, X2
    x, y = int(x), int(y)

    # shift image
    x2_ = Signal(np.roll(x2.real.copy(), shift=(y, x), axis=(0, 1)))

    # process image
    X2 = fft2(x2_.arr)
    x3 = phase_correlation(X2, X1)
    x3 = fftshift(x3)
    x3 = Signal(x3)

    # calculate shifts
    dx, dy = x, y
    edy, edx = [int(x) for x in np.where(x3.real == x3.max)] # estimated shift

    img2.set_data(x2_.real)
    img3.set_data(x3.real)
    fig.canvas.draw_idle()

plt.connect('motion_notify_event', lambda event : mouse_move(event, LMB))
plt.connect('button_press_event', lambda event: mouse_click(event, LMB))
plt.connect('button_release_event', lambda event: mouse_release(event, LMB))
plt.show(block=True)
