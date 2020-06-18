import numpy as np
from scipy.fftpack import fft2
from scipy.signal import windows
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.widgets import Button
from dzlib.common.utils import stats, info
import pdb
matplotlib.use("Qt5Agg")


def phase_correlation(X1, X2):
    """
    Function to compute the phase correlation between two signals by calculating the inverse DFT of the normalized cross-power spectrum of the two signals DFTs.
    Note1: The cross-power spectrum is calculated via element-wise multiplication (Hadamard Product)
    Note2: eps is added to the cross-power spectrum to avoid divide-by-zero errors when calculating the normalized cross-power spectrum

    Args:
        X1 (Numpy Array): DFT of signal 1, shape (N2 x N1)
        X2 (Numpy Array): DFT of signal 2, shape (N2 x N1)

    Returns:
        ncc (Numpy Array): IDFT of the Normalized Cross-Power Spectrum of the DFTs of both input signals
    """
    # cross-power spectrum (cps), normalized cps (ncps), normalized cross-correlation (ncc)
    N2, N1 = X1.shape
    cps = np.multiply(X1, np.conj(X2)) + eps
    ncps = cps / np.abs(cps)
    ncc = (1 / (N1 * N2)) * fft2(ncps)
    return ncc


def coords_bottomleft(coords_center, scene_shape, image_shape):
    """Transforms (x, y) center coords to bottom left coords of an image within the span of a scene
    """
    x, y = coords_center
    N1, N2 = scene_shape
    w, h = image_shape
    x = int(min(max(x, w/2), N1-w/2) - w/2)
    y = int(min(max(y, h/2), N2-h/2) - h/2)
    return (x, y)




def add_noise(x, mean, std):
    '''Function to add Gaussian noise'''
    noise = mean + std * np.random.randn(x.shape)
    x += noise
    return x

global eps
eps = 1e-6

# initialize scene
## scene span
N1, N2 = 400, 400
image_path = '/'.join((os.getcwd(), 'data/image2.png'))
scene = Image.open(image_path).convert('L')
scene = np.asarray(scene.resize((N2, N1))).astype(np.float32)
scene = np.flip(scene, axis=(0)) # scene is flipped along row axis to account for setting imshow origin at bottom left

# (x, y) center coords for each image
initial_n1, initial_n2 = 100, 150
initial_m1, initial_m2 = 100, 150

# image span
w, h = 100, 100

# calculate bottom left coords from center coords
n1, n2 = coords_bottomleft((initial_n1, initial_n2), (N1, N2), (w, h))
m1, m2 = coords_bottomleft((initial_m1, initial_m2), (N1, N2), (w, h))

# initialize images
x1 = scene[n2: n2+h, n1: n1+w]
x2 = scene[m2: m2+h, m1: m1+w]

# initialize window
std = 20
i = 0
window_list = [windows.triang(w), windows.gaussian(w, std=std)]
window_type = ['Triangular Window', 'Gaussian Window']
# window = windows.triang(w)
# window = windows.gaussian(w, std=std)
window = np.outer(window_list[i], window_list[i])

# windowed images
wx1 = np.multiply(window, x1)
wx2 = np.multiply(window, x2)

# compute time-domain impulse image via phase correlation
X1, X2 = fft2(wx1), fft2(wx2)
ncc = phase_correlation(X2, X1)
x3 = np.abs(ncc)
x3 = np.roll(a=x3, shift=(int(h/2), int(w/2)), axis=(0, 1))

# estimate time-domain shift (peak of impulse image), and get actual shift
e2, e1 = np.where(x3 == np.max(x3))
a1, a2 = m1-n1, m2-n2
eshift = (e1[0]-int(w/2), e2[0]-int(h/2))
ashift = (a1, a2)

# create focused image (for plotting purposes)
focus = np.zeros((N2, N1))
focus[m2: m2+h, m1: m1+w] = x2.copy()

# bounding boxes
rect1 = patches.Rectangle(xy=(n1, n2), width=w, height=h, fill=False, color='blue')
rect2 = patches.Rectangle(xy=(m1, m2), width=w, height=h, fill=False, color='red')

# initial plot
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0)
gs = GridSpec(nrows=2, ncols=4)
ax1 = plt.subplot(gs[0, 0:1])
ax2 = plt.subplot(gs[0, 1:2])
ax3 = plt.subplot(gs[0:2, 2:4])
ax4 = plt.subplot(gs[1, 0:1])
ax5 = plt.subplot(gs[1, 1:2])

# reset button
left = 0.03
bot = 0.8
width = 0.08
height = 0.04
button1 = plt.axes([left, bot, width, height])
reset = Button(ax=button1, label='Reset', image=None, color='0.85', hovercolor='0.95')

# filters button
left = 0.03
bot = 0.7
width = 0.08
height = 0.04
button2 = plt.axes([left, bot, width, height])
filters = Button(ax=button2, label='Cycle Filter', image=None, color='0.85', hovercolor='0.95')

## plot images
vmin = np.min(scene)
vmax = np.max(scene)
img1 = ax1.imshow(scene, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=vmin, vmax=vmax)
img2 = ax2.imshow(focus, extent=[0, N1, 0, N2], origin='lower', interpolation=None, cmap='gray', vmin=vmin, vmax=vmax)
# img3 = ax3.imshow(x3, extent=[0, w, 0, h], origin='lower', interpolation=None, cmap='Reds', vmin=np.min(x3), vmax=np.max(x3))
img3 = ax3.imshow(x3, extent=[-w/2, w/2, -h/2, h/2], origin='lower', interpolation=None, cmap='Reds', vmin=np.min(x3), vmax=np.max(x3))
img4 = ax4.imshow(wx1, extent=[0, w, 0, h], origin='lower', interpolation=None, cmap='gray', vmin=vmin, vmax=vmax)
img5 = ax5.imshow(wx2, extent=[0, w, 0, h], origin='lower', interpolation=None, cmap='gray', vmin=vmin, vmax=vmax)

## plot bounding boxes
box1 = ax1.add_patch(rect1)
box2 = ax2.add_patch(rect2)

# adjust ticks and tick labels
for ax in [ax1, ax2, ax4, ax5]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax3.tick_params(left=False, labelleft=False, right=True, labelright=True)
ax4.set_xlabel(window_type[i])
ax5.set_xlabel(window_type[i])

# titles
ax1.set_title('Scene + Focus1')
ax2.set_title(f'Just Focus2\nActual Shift: {ashift}')
ax3.set_title(f'Motion Estimate\nEstimated Shift: {eshift}')
ax4.set_title('Windowed Focus1')
ax5.set_title('Windowed Focus2')


# plotting
class MouseButton():
    def __init__(self, down=False):
        self.isdown = down

LMB = MouseButton()

def mouse_click(event, mousebutton):
    if event.button == 1:
        LMB.isdown = True

def mouse_release(event, mousebutton):
    if event.button == 1:
        LMB.isdown = False

def mouse_move(event, mousebutton):
    x, y = event.xdata, event.ydata
    ax = event.inaxes

    if (LMB.isdown and (ax == ax2)):
        update(x, y)


def update(x, y):

    # Image Processing
    # calculate new coords
    global m1, m2
    m1, m2 = coords_bottomleft((x, y), (N1, N2), (w, h))

    # update wx2 with new coords
    x2 = scene[m2: m2+h, m1: m1+w]
    wx2 = np.multiply(window, x2)

    # compute DFT X2 with new wx2
    X2 = fft2(wx2)

    # compute phase correlation x3 with new X2
    ncc = phase_correlation(X2, X1)
    x3 = np.abs(ncc)
    x3 = np.roll(a=x3, shift=(int(h/2), int(w/2)), axis=(0, 1))

    # compute estimated shift with new x3
    e2, e1 = np.where(x3 == np.max(x3))
    eshift = (e1[0]-int(w/2), e2[0]-int(h/2))

    # compute actual shift with new coords
    a1, a2 = m1-n1, m2-n2
    ashift = (a1, a2)

    # Image Plotting
    # update focus2 with new coords
    focus = np.zeros((N2, N1))
    focus[m2: m2+h, m1: m1+w] = x2.copy()

    # plot new images and box
    box2.set_xy((m1, m2))
    img2.set_data(focus)
    img3.set_data(x3)
    img4.set_data(wx1)
    img5.set_data(wx2)

    # update titles
    ax2.set_title(f'Just Focus2\nActual Shift: {ashift}')
    ax3.set_title(f'Motion Estimate\nEstimated Shift: {eshift}')
    fig.canvas.draw_idle()


# reset button, runs update function with initial coords
def button1_click(event):
    update(initial_n1, initial_n2)

# cycle filter button, runs update function with last coords
def button2_click(event):
    global window
    global i
    global X1
    global wx1
    i  = (i + 1) % len(window_list)
    window = np.outer(window_list[i], window_list[i])
    wx1 = np.multiply(window, x1)
    X1 = fft2(wx1)
    update(m1+w/2, m2+h/2)
    ax4.set_xlabel(window_type[i])
    ax5.set_xlabel(window_type[i])

# pdb.set_trace()
reset.on_clicked(button1_click)
filters.on_clicked(button2_click)
plt.connect('motion_notify_event', lambda event : mouse_move(event, LMB))
plt.connect('button_press_event', lambda event : mouse_click(event, LMB))
plt.connect('button_release_event', lambda event : mouse_release(event, LMB))
plt.show(block=True)
