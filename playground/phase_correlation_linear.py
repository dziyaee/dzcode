import numpy as np
from scipy.fftpack import fft2
from PIL import Image
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from dzlib.common.utils import stats, info
import pdb
mpl.use("Qt5Agg")


def phase_correlation(X1, X2, shape):
    # shape
    N1, N2 = shape

    # cross-power spectrum (Hadamard Product)
    cps = np.multiply(X1, np.conj(X2))

    # normalized cross-power spectrum
    ncps = cps / (np.abs(cps) + eps)

    # normalized cross-correlation
    ncc = (1 / (N1 * N2)) * fft2(ncps)
    return ncc


eps = 1e-6

# create images
N2, N1 = 400, 400
image_path = '/'.join((os.getcwd(), 'data/image2.png'))
image = Image.open(image_path).convert('L')
image = np.asarray(image.resize((N1, N2))).astype(np.float32)
image = np.flip(image, axis=0)
# blank = np.zeros((N1, N2)) + np.max(image)
blank = np.zeros((N1, N2)) + 0

# shift and bounding box params
n1, n2 = 0, 0
a1, a2 = 0, 0
m1 = n1 + a1
m2 = n2 + a2
w, h = 100, 100

# initialize images within bounding boxes
x1 = blank.copy()
x2 = blank.copy()
x1[n2: n2+h, n1: n1+w] = image[n2: n2+h, n1: n1+w]
x2[m2: m2+h, m1: m1+w] = image[n2: n2+h, n1: n1+w]

# compute time-domain impulse image via phase correlation
X1, X2 = fft2(x1), fft2(x2)
ncc = phase_correlation(X1, X2, (N1, N2))
x3 = np.abs(ncc)

# estimate time-domain shift (peak of impulse image), and get actual shift
e1, e2 = np.where(x3 == np.max(x3))
eshift = (e1[0], e2[0])
ashift = (a1, a2)

# bounding boxes
rect1 = patches.Rectangle(xy=(n1, n2), width=w, height=h, fill=False, color='blue')
rect2 = patches.Rectangle(xy=(m1, m2), width=w, height=h, fill=False, color='red')

# initial plot
fig = plt.figure()
fig.subplots_adjust(wspace=0)
gs = GridSpec(nrows=1, ncols=3)
ax1 = plt.subplot(gs[0, 0:1])
ax2 = plt.subplot(gs[0, 1:2])
ax3 = plt.subplot(gs[0, 2:3])

## plot images
vmin = np.min(image)
vmax = np.max(image)
img1 = ax1.imshow(image, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=vmin, vmax=vmax)
img2 = ax2.imshow(x2, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=vmin, vmax=vmax)
img3 = ax3.imshow(x3, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='Reds', vmin=np.min(x3), vmax=np.max(x3))

## plot bounding boxes
box1 = ax1.add_patch(rect1)
box2 = ax2.add_patch(rect2)

# adjust ticks and tick labels
ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax3.tick_params(left=False, labelleft=False, right=True, labelright=True)

# plotting functions
clicking = False

def mouse_click(event):
    global clicking
    if event.button == 1:
        clicking = True


def mouse_release(event):
    global clicking
    if event.button == 1:
        clicking = False


def mouse_move(event):
    x, y = event.xdata, event.ydata
    ax = event.inaxes

    if (clicking and (ax == ax2)):
        m1, m2 = x, y
        update(m1, m2)


def update(m1, m2):
    # move box2 to new coords
    m1 = int(np.floor(min(m1, N2-w)))
    m2 = int(np.floor(min(m2, N1-h)))

    # update image2 with new coords
    x2 = blank.copy()
    x2[m2: m2+h, m1: m1+w] = image[m2: m2+h, m1: m1+w]

    # compute time-domain impulse image via phase correlation
    X2 = fft2(x2)
    ncc = phase_correlation(X1, X2, (N1, N2))
    x3 = np.abs(ncc)

    # estimate time-domain shift (peak of impulse image), and get actual shift
    e2, e1 = np.where(x3 == np.max(x3))
    eshift = (e1[0], e2[0])
    a1 = m1 - n1
    a2 = m2 - n2
    ashift = (a1, a2)

    # plot new images and box
    box2.set_xy((m1, m2))
    img2.set_data(x2)
    img3.set_data(x3)

    # update titles
    ax2.set_title(f'{ashift}')
    ax3.set_title(f'{eshift}')
    fig.canvas.draw_idle()
    pass


# pdb.set_trace()
plt.connect('motion_notify_event', mouse_move)
plt.connect('button_press_event', mouse_click)
plt.connect('button_release_event', mouse_release)
plt.show(block=True)
