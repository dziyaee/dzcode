import numpy as np
from scipy.fftpack import fft2
from PIL import Image
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from dzlib.common.utils import stats, info
from phase_correlator import image_processing
import pdb
mpl.use("Qt5Agg")


eps = 1e-6

# load images
N2, N1 = 400, 400
image_path = '/'.join((os.getcwd(), 'data/image2.png'))
image = Image.open(image_path).convert('L')
image = np.asarray(image.resize((N2, N1))).astype(np.float32)
image = np.flip(image, axis=0)
blank = np.zeros((N2, N1)) + np.max(image)

x1 = image.copy()
x2 = blank.copy()
# x2 = np.zeros((N2, N1)) + 255

# process images
X1, X2 = fft2(x1), fft2(x2)
x3, eshift = image_processing(X1, X2, N1, N2)

# bounding boxes
w, h = 100, 100
rect1 = patches.Rectangle(xy=(100, 100), width=w, height=h, fill=False, color='blue')
rect2 = patches.Rectangle(xy=(100, 100), width=w, height=h, fill=False, color='red', ls='--')

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
img1 = ax1.imshow(x1, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=vmin, vmax=vmax)
img2 = ax2.imshow(x2, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=vmin, vmax=vmax)
# img2 = ax2.imshow(x2, extent=[0, N1, 0, N2], origin='lower', interpolation=None)
img3 = ax3.imshow(x3, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='Reds')

## plot bounding boxes
box1 = ax1.add_patch(rect1)
box2 = ax2.add_patch(rect2)

for ax in [ax1, ax2, ax3]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# update plot functions
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
        n1, n2 = x, y
        update(n1, n2)


def update(n1, n2):

    # move bounding box to new coords
    n1 = int(np.floor(min(n1, N1-w)))
    n2 = int(np.floor(min(n2, N2-h)))
    box2.set_xy((n1, n2))

    # create new image2
    x2 = blank.copy()
    x2[n2: n2+h, n1: n1+w] = x1[n2: n2+h, n1: n1+w]
    img2.set_data(x2)


    fig.canvas.draw_idle()
    pass


# pdb.set_trace()
plt.connect('motion_notify_event', mouse_move)
plt.connect('button_press_event', mouse_click)
plt.connect('button_release_event', mouse_release)
plt.show(block=True)
