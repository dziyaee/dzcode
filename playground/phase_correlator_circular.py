# Imports
# general
import os
from PIL import Image
import numpy as np

# scipy
from scipy.fftpack import fft2, fftshift

# dzlib
from dzlib.signal_processing.signals import signal
from dzlib.signal_processing.utils import phase_correlation2d, add_gaussian_noise
from dzlib.common.devices import MouseButton

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# starting params
global eps
eps = 1e-6
N2, N1 = 400, 400 # scene span (rows, cols) or (height, width)
noise_mean = 0
noise_std = 20

# init scene
scene_path = '/'.join((os.getcwd(), 'data/image2.png'))
scene = Image.open(scene_path).convert('L')
scene = np.asarray(scene.resize((N2, N1))).astype(np.float32)
scene = np.flip(scene, axis=(0)) # flip to account for imshow(center='lower')

# init noisy images
x1, snr1 = add_gaussian_noise(x=scene.copy(), noise_mean=noise_mean, noise_std=noise_std, return_snr=True)
x2, snr2 = add_gaussian_noise(x=scene.copy(), noise_mean=noise_mean, noise_std=noise_std, return_snr=True)
# x2 = np.roll(x2, shift=(0, 0), axis=(0, 1))
# x1 = signal(x1)
# x2 = signal(x2)

# image processing
X1, X2 = fft2(x1), fft2(x2)
x3 = phase_correlation2d(X2, X1)
x3 = fftshift(x3)
x3 = signal(x3)

# calculate shifts
a1, a0 = 0, 0 # actual shift
e2, e1 = np.where(x3.mag == x3.max) # estimated shift
e1, e2 = e1[0], e2[0]


# initial plot
fig = plt.figure()
fig.subplots_adjust(wspace=0)
gs = GridSpec(nrows=1, ncols=3)
ax1 = plt.subplot(gs[0, 0:1])
ax2 = plt.subplot(gs[0, 1:2])
ax3 = plt.subplot(gs[0, 2:3])

# img1 = ax1.imshow(x1, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=x1.min, vmax=x1.max)
# img2 = ax2.imshow(x2, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=x2.min, vmax=x2.max)
# img3 = ax3.imshow(x3, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='Reds', vmin=x3.min, vmax=x3.max)

img1 = ax1.imshow(x1, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray')
img2 = ax2.imshow(x2, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray')
img3 = ax3.imshow(x3.mag, extent=[-N1/2, N1/2, -N2/2, N2/2], origin='lower', interpolation=None, cmap='Reds', vmin=x3.min, vmax=x3.max)

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
        # n1, n2 = int(y), int(x)
        # update(n1, n2)
        update(x, y)


# def update(n1, n2):
def update(x, y):
    global x2_, X2
    x, y = int(x), int(y)

    # shift image
    x2_ = np.roll(x2.copy(), shift=(y, x), axis=(0, 1))

    # process image
    X2 = fft2(x2_)
    x3 = phase_correlation2d(X2, X1)
    x3 = fftshift(x3)
    x3 = signal(x3)

    # calculate shifts
    a1, a2 = x, y
    e2, e1 = np.where(x3.mag == x3.max) # estimated shift
    e1, e2 = e1[0], e2[0]

    img2.set_data(x2_)
    img3.set_data(x3.mag)
    fig.canvas.draw_idle()


# pdb.set_trace()
plt.connect('motion_notify_event', lambda event : mouse_move(event, LMB))
plt.connect('button_press_event', lambda event: mouse_click(event, LMB))
plt.connect('button_release_event', lambda event: mouse_release(event, LMB))
plt.show(block=True)
