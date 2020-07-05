# Imports
# general
import os
from PIL import Image
import numpy as np

# scipy
from scipy.fft import fft2, fftshift
from scipy.signal import windows

# dzlib
# from dzlib.signal_processing.signals import signal
from dzlib.signal_processing.signals import Signal
from dzlib.signal_processing.utils import phase_correlation2d, window2d
from dzlib.common.utils import Cyclic_Data

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches


class SignalCoords(Signal):
    def __init__(self, array, xy):
        super().__init__(array)
        self.x, self.y = xy

    def set_xy(self, xy):
        self.x, self.y = xy


def coords_bottomleft(coords_center, scene_shape, image_shape):
    """Transforms the center coords of an image within a scene to the bottom-left coords"""
    x, y = coords_center
    N2, N1 = scene_shape
    h, w = image_shape
    x = int(min(max(x, w/2), N1-w/2) - w/2)
    y = int(min(max(y, h/2), N2-h/2) - h/2)
    return (x, y)


def new_ydata(x):
    """Function to specify the data to track"""
    return np.var(x.real), np.var(x.mag)


def unroll_shift(x, N):
    """Function to transform shift coords span from (-N/2 to +N/2) to (0 to N)"""
    z = x%(N/2) - (x//(N/2))*(N/2)
    return int(z)


# starting params
global eps
eps = 1e-6
N2, N1 = 400, 400 # scene span (rows, cols) or (height, width)
h, w = 100, 100 # image span
n2, n1 = 100, 100 # image1 bottom-left coords
m2, m1 = 100, 100 # image2 bottom-left coords

# Some arrays are stored in my custom 'Signal' namedtuple with the following structure:
# Signal = namedtuple('Signal', 'arr real imag mag phase min max mean std var')
# arr stores the array itself, min, max, mean, std, var are all calculated on the array's magnitude

# Naming convention for arrays: lowercase for non-frequency domain, uppercase for frequency-domain. Example: x1 is a time-domain signal. DFT of x1 = X1, a frequency-domain signal. IDFT of X1 = x1

# init scene
scene_path = '/'.join((os.getcwd(), 'data/image2.png'))
scene = Image.open(scene_path).convert("L")
scene = np.asarray(scene.resize((N2, N1))).astype(np.float32)
scene = np.flip(scene, axis=(0)) # flip to account for imshow(center='lower')
scene = Signal(scene) # stored as Signal class

# init 2d window
window = windows.triang(w)
window = window2d(window) # create a 2d window out of a 1d window

# init images
x1 = scene.arr[n2: n2+h, n1: n1+w]
x1 = np.multiply(window, x1)
x1 = SignalCoords(x1, (n1, n2))
x2 = scene.arr[m2: m2+h, m1: m1+w]
x2 = np.multiply(window, x2)
x2 = SignalCoords(x2, (m1, m2))

# image processing
X1, X2 = fft2(x1.arr), fft2(x2.arr)
x3 = phase_correlation2d(X1, X2) # returns the normalized cross-correlation between X1 and X2
x3 = fftshift(x3)
x3 = Signal(x3)

# calc shifts
dx = x2.x - x1.x # actual shift
dy = x2.y - x1.y
edy, edx = [int(x) for x in np.where(x3.real == x3.max)] # estimated shift
edx -= int(w/2) # adjust for fftshift of x3
edy -= int(h/2)
edx = unroll_shift(edx, w)
edy = unroll_shift(edy, h)
peakreal = np.where(x3.real.flatten() == x3.max)[0][0]
peakmag = np.where(x3.mag.flatten() == np.max(x3.mag))[0][0]

# Init data tracking
n_xdata = 200
xdata = np.arange(n_xdata)
n_ydata = len(new_ydata(x3))
ydatas = Cyclic_Data(n_xdata, n_ydata) # cyclic data array class
ydatas.update(new_ydata(x3)) # new values will replace old values after reaching the end
i = ydatas.i

# Set up figure and subplots
fig1 = plt.figure(figsize=(14, 5))
gs = GridSpec(nrows=4, ncols=15)
ax0 = fig1.add_subplot(gs[0:2, 0:3])
ax1 = fig1.add_subplot(gs[0:2, 4:8])
ax2 = fig1.add_subplot(gs[0:2, 9:12])
ax3 = fig1.add_subplot(gs[0:2, 12:15])
ax4 = fig1.add_subplot(gs[2:3, 0:7])
ax5 = fig1.add_subplot(gs[2:3, 8:15])
ax6 = fig1.add_subplot(gs[3:4, 0:7])
ax7 = fig1.add_subplot(gs[3:4, 8:15])
axes_stats = [ax4, ax5]
fig1.subplots_adjust(left=0.04, right=0.99, hspace=0.5, wspace=0)
fig1.tight_layout()

# ticks, lims, titles
for ax in [ax2, ax3]:
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
for ax, ydata in zip(axes_stats, ydatas.data.T):
    ax.set_xlim(0, n_xdata)
    ax.set_ylim(0, ydata[i]*1.1)
ax0.set_title(f'Scene\nActual Shift = {dx, dy}')
ax1.set_title('$x_{3real}$ \n'f'Estimated Shift = {edx, edy}')
ax2.set_title(f'Image 1 Focus')
ax3.set_title(f'Image 2 Focus')
ax4.set_title('Variance $(x_{3real})$')
ax5.set_title('Variance $(x_{3mag})$')
ax6.set_title('Flattened $(x_{3real})$')
ax7.set_title('Flattened $(x_{3mag})$')

# bounding boxes
rect1 = patches.Rectangle(xy=(n1, n2), width=w, height=h, fill=False, color='blue', ls='--')
rect2 = patches.Rectangle(xy=(m1, m2), width=w, height=h, fill=False, color='red')

# scene image & bounding boxes plots
img_scene = ax0.imshow(scene.real, extent=[0, N1, 0, N2], origin='lower', cmap='gray', interpolation=None, vmin=scene.min, vmax=scene.max)
box1 = ax0.add_patch(rect1)
box2 = ax0.add_patch(rect2)

# motion estimation image plot
img1 = ax1.imshow(x3.real, extent=[-w/2, w/2, -h/2, h/2], origin='lower', cmap='Reds', interpolation=None, vmin=x3.min, vmax=x3.max)

# foci image plots
img2 = ax2.imshow(x1.real, extent=[0, h, 0, w], origin='lower', cmap='gray', interpolation=None, vmin=x1.min, vmax=x1.max)
img3 = ax3.imshow(x2.real, extent=[0, h, 0, w], origin='lower', cmap='gray', interpolation=None, vmin=x2.min, vmax=x2.max)

# motion estimation stats plots
lines = []
for ydata, ax in zip(ydatas.data.T, axes_stats):
    lines.append(ax.plot(xdata[:i], ydata[:i]))

# motion estimation line plot
line1, = ax6.plot(x3.real.flatten())
point1, = ax6.plot(peakreal, x3.max, 'ro')
ax6.axhline(0, color='black', lw=1)
line2, = ax7.plot(x3.mag.flatten())
point2, = ax7.plot(peakmag, x3.max, 'ro')
ax7.axhline(0, color='black', lw=1)

def update(x, y):
    # calculate new bottom-left coords from new center coords
    m1, m2 = coords_bottomleft((x, y), (N2, N1), (h, w))

    # create new windowed image with new coords
    x2 = scene.arr[m2: m2+h, m1: m1+w]
    x2 = np.multiply(window, x2)
    x2 = SignalCoords(x2, (m1, m2))

    # image processing
    X2 = fft2(x2.arr)
    x3 = phase_correlation2d(X1, X2) # returns the normalized cross-correlation between X1 and X2
    x3 = fftshift(x3)
    x3 = Signal(x3)

    # calc shifts
    dx = x2.x - x1.x # actual shift
    dy = x2.y - x1.y
    edy, edx = [int(x) for x in np.where(x3.real == x3.max)] # estimated shift
    edx -= int(w/2) # adjust for fftshift of x3
    edy -= int(h/2)
    edx = unroll_shift(edx, w)
    edy = unroll_shift(edy, h)
    peakreal = np.where(x3.real.flatten() == x3.max)[0][0]
    peakmag = np.where(x3.mag.flatten() == np.max(x3.mag))[0][0]

    # update data tracking
    ydatas.update(new_ydata(x3))
    i = ydatas.i

    # update box, image, stats plots
    box2.set_xy((m1, m2))
    img1.set_data(x3.real)
    img3.set_data(x2.real)
    for ydata, line, ax in zip(ydatas.data.T, lines, axes_stats):
        line[0].set_data(xdata[:i], ydata[:i])
    line1.set_ydata(x3.real.flatten())
    point1.set_data(peakreal, x3.max)
    line2.set_ydata(x3.mag.flatten())
    point2.set_data(peakmag, np.max(x3.mag))

    # update titles
    ax0.set_title(f'Scene\nActual Shift = {dx, dy}')
    ax1.set_title(f'Motion Estimate\nEstimated Shift = {edx, edy}')
    ax6.set_title('$Flattened x_{3real}$')
    ax7.set_title('$Flattened x_{3mag}$')

    fig1.canvas.draw_idle()


class MouseButton():
    def __init__(self, isdown=False):
        self.isdown = isdown

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

    if (LMB.isdown and ax == ax0):
        update(x, y)

# event trackers
plt.connect('motion_notify_event', lambda event : mouse_move(event, LMB))
plt.connect('button_press_event', lambda event : mouse_click(event, LMB))
plt.connect('button_release_event', lambda event : mouse_release(event, LMB))
plt.show(block=True)
