import numpy as np
from numpy import pi, exp, cos, sin
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import windows
from dzlib.signal_processing.signals import signal, printsignal, printstats
from PIL import Image
import os


def phase_corr2d(X1, X2):
    X3 = np.multiply(X1, np.conj(X2))
    X3 = X3 / np.abs(X3)
    x3 = ifft2(X3)
    x3 = fftshift(x3)
    return x3


def coords_bottomleft(coords_center, scene_shape, image_shape):
    """Transforms (x, y) center coords to bottom left coords of an image within the span of a scene
    """
    x, y = coords_center
    N1, N2 = scene_shape
    w, h = image_shape
    x = int(min(max(x, w/2), N1-w/2) - w/2)
    y = int(min(max(y, h/2), N2-h/2) - h/2)
    return (x, y)



# Starting Params
global eps, i, n_points
eps = 1e-6
n_points = 100
i = 0%n_points
N2, N1 = 400, 400
h, w = 100, 100
n2, n1 = 100, 100
m2, m1 = 100, 100

# init scene, window, and images
scene_path = '/'.join((os.getcwd(), 'data/image2.png'))
scene = Image.open(scene_path).convert("L")
scene = np.asarray(scene.resize((N2, N1))).astype(np.float32)
scene = np.flip(scene, axis=(0))
scene = signal(scene)
window = windows.triang(w)
window = np.outer(window, window)
x1 = scene.arr[n2: n2+h, n1: n1+w]
x1 = np.multiply(window, x1)
x1 = signal(x1)
x2 = scene.arr[m2: m2+h, m1: m1+w]
x2 = np.multiply(window, x2)
x2 = signal(x2)

# Image processing
X1, X2 = fft2(x1.arr), fft2(x2.arr)
x3 = phase_corr2d(X1, X2)
x3 = signal(x3)

# calc shifts
a1, a2 = n1-n1, n2-n2 # actual shift
e2, e1 = np.where(x3.mag == np.max(x3.mag)) # estimated shift
e1, e2 = e1[0] - int(w/2), e2[0] - int(h/2) # adjusted for center origin plot


# Init data tracking
xdata = np.arange(n_points)
ydatas = np.zeros((n_points, 4))
ydatas[i, :] = [x3.max, x3.mean, x3.std, x3.var]

# Set up figure and subplots
fig = plt.figure(figsize=(14, 5))
gs = GridSpec(nrows=3, ncols=15)
ax0 = fig.add_subplot(gs[0:2, 0:3])
ax3 = fig.add_subplot(gs[0:2, 4:8])
ax1 = fig.add_subplot(gs[0:2, 9:12])
ax2 = fig.add_subplot(gs[0:2, 12:15])
ax4 = fig.add_subplot(gs[2:3, 0:3])
ax5 = fig.add_subplot(gs[2:3, 4:7])
ax6 = fig.add_subplot(gs[2:3, 8:11])
ax7 = fig.add_subplot(gs[2:3, 12:15])
axes = [ax4, ax5, ax6, ax7]
fig.subplots_adjust(left=0.04, right=0.99, hspace=0.5, wspace=0)
for ax in [ax1, ax2]:
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
for ax in axes:
    ax.set_xlim(0, n_points)
ax0.set_title(f'Scene\nActual Shift = {a1, a2}')
ax1.set_title(f'Image 1 Focus')
ax2.set_title(f'Image 2 Focus')
ax3.set_title('$x_{3mag}$ \n'f'Estimated Shift = {e1, e2}')
ax4.set_title('$MAX(x_{3mag})$')
ax5.set_title('$MEAN(x_{3mag})$')
ax6.set_title('$STD(x_{3mag})$')
ax7.set_title('$VAR(x_{3mag})$')

# Set up bounding boxes
rect1 = patches.Rectangle(xy=(n1, n2), width=w, height=h, fill=False, color='blue', ls='--')
rect2 = patches.Rectangle(xy=(m1, m2), width=w, height=h, fill=False, color='red')

# Initial plots
img_scene = ax0.imshow(scene.real, extent=[0, N1, 0, N2], origin='lower', cmap='gray', interpolation=None, vmin=scene.min, vmax=scene.max)
box1 = ax0.add_patch(rect1)
box2 = ax0.add_patch(rect2)
img1 = ax1.imshow(x1.real, extent=[0, h, 0, w], origin='lower', cmap='gray', interpolation=None, vmin=x1.min, vmax=x1.max)
img2 = ax2.imshow(x2.real, extent=[0, h, 0, w], origin='lower', cmap='gray', interpolation=None, vmin=x2.min, vmax=x2.max)
img3 = ax3.imshow(x3.mag, extent=[-w/2, w/2, -h/2, h/2], origin='lower', cmap='Reds', interpolation=None, vmin=x3.min, vmax=x3.max)
lines = []
for ydata, ax in zip(ydatas.T, axes):
    lines.append(ax.plot(xdata[:i], ydata[:i]))

def update(x, y):
    global i
    i += 1

    # calculate new coords
    m1, m2 = coords_bottomleft((x, y), (N1, N2), (w, h))

    # create new windowed image with new coords
    x2 = scene.arr[m2: m2+h, m1: m1+w]
    x2 = np.multiply(window, x2)
    x2 = signal(x2)

    # image processing
    X2 = fft2(x2.arr)
    x3 = phase_corr2d(X1, X2)
    x3 = signal(x3)

    # update data tracking
    ydatas[i%100, :] = [x3.max, x3.mean, x3.std, x3.var]

    # calc shifts
    a1, a2 = m1-n1, m2-n2 # actual shift
    e2, e1 = np.where(x3.mag == np.max(x3.mag)) # estimated shift
    e1, e2 = e1[0] - int(w/2), e2[0] - int(h/2) # adjusted for center origin plot

    # update plots
    box2.set_xy((m1, m2))
    img2.set_data(x2.real)
    img3.set_data(x3.mag)
    for ydata, line, ax in zip(ydatas.T, lines, axes):
        line[0].set_data(xdata[:i%100], ydata[:i%100])
        ax.set_ylim(0, np.max(ydata)*2)
    ax0.set_title(f'Scene\nActual Shift = {a1, a2}')
    ax3.set_title(f'Motion Estimate\nEstimated Shift = {e1, e2}')

    fig.canvas.draw_idle()


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


plt.connect('motion_notify_event', lambda event : mouse_move(event, LMB))
plt.connect('button_press_event', lambda event : mouse_click(event, LMB))
plt.connect('button_release_event', lambda event : mouse_release(event, LMB))
plt.show(block=True)

