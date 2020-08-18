import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
from dzlib.signal_processing.sweep2d import Sweep2d
from dzlib.signal_processing.hough import Hough
from dzlib.common.utils import stats, info


mpl.use("Qt5Agg")


## IMAGE LOADING ---------
# load image as array, normalize, transpose
image_path = "data/highway1.png"

image_rgb = Image.open(image_path).convert("RGB")  # (W x H x C)
array_rgb = np.asarray(image_rgb)  # (H x W x C)
array_rgb = array_rgb / 255

## IMAGE FILTERING ---------
# apply mask to filter all colors except yellow and white
# idea from: https://www.hackster.io/kemfic/simple-lane-detection-c3db2f

# convert to HSV (linked post uses HSL colorspace, but I had an easier time converting image to HSV)
array_hsv = rgb_to_hsv(array_rgb)
array_rgb = array_rgb.transpose(2, 0, 1)  # (C x H x W)
array_hsv = array_hsv.transpose(2, 0, 1)  # (C x H x W)

# yellow HSV mins/maxs
ymins = np.array([40/360, 70/100, 70/100]).reshape(3, 1, 1)
ymaxs = np.array([50/360, 100/100, 100/100]).reshape(3, 1, 1)

# white HSV mins/maxs
wmins = np.array([0/360, 0/100, 95/100]).reshape(3, 1, 1)
wmaxs = np.array([360/360, 100/100, 100/100]).reshape(3, 1, 1)

# convert any values outside yellow AND white HSV ranges to zero (this is highly tunable...)
x = array_hsv.copy()  # to keep the mask code fairly readable
x[((x > wmaxs) | (x < wmins)) & ((x > ymaxs) | (x < ymins))] = 0
array_hsv_masked = x.copy()
array_rgb_masked = hsv_to_rgb(array_hsv_masked.transpose(1, 2, 0))  # (H x W x C)
array_rgb_masked = array_rgb_masked.transpose(2, 0, 1)  # (C x H x W)
del x

## EDGE DETECTION --------
# convert masked RGB to binary for edge detection purposes
x = array_rgb_masked.copy()
x[x != 0] = 1
array_bn_masked = np.mean(x, axis=0)  # all channels of x have same values, so mean is taken to reduce down to 2d image
del x

# edge detection on grayscale image via Sobel operator kernels and Sweep2d class to perform the convolution operation
# idea from: https://en.wikipedia.org/wiki/Sobel_operator
# the results from the vertical and horizontal edge detections can be combined to produce a full image edge map
gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # vertical edge detection kernel (3 x 3)
gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # horizontal edge detection kernel (3 x 3)

# Sweep2d inputs
images = array_bn_masked[None, None]  # convert 1 2d grayscale image array to 1 4d array (1 x 1 x H x W)
kernels = np.array([[gx], [gy]])  # convert 2 2d kernel arrays to 1 4d array (2 x 1 x 3 x 3)
padding = (0, 0)
stride = (1, 1)
mode = "same"

# edge detection
sweeper = Sweep2d(images.shape, kernels.shape, padding, stride, mode)
vertical_edges, horizontal_edges = sweeper.convolve2d(images, kernels)[0]  # (H x W)
edges = np.sqrt((vertical_edges ** 2) + (horizontal_edges ** 2))

# normalize
edges = edges - np.min(edges)
edges = edges / np.max(edges)

# binarize
edges_bn = edges.copy()
edges_bn[edges_bn != 0] = 1

## LINE DETECTION -----------
rho_step = 1
theta_step = 1
theta_min = -85
theta_max = 85
threshold = 0.51
hough = Hough(edges_bn, rho_step, theta_step, (theta_min, theta_max))
hough.transform(threshold)
print(hough.n_lines)

# PLOTS -----------------
fig4, axes4 = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
ax10, ax11 = axes4

fig3, axes3 = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))
ax6, ax7 = axes3[0]
ax8, ax9 = axes3[1]

fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
ax3, ax4, ax5 = axes2

fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
ax1, ax2 = axes1

figs = [fig1, fig2, fig3, fig4]
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]

# Fig 1
fig1.suptitle("Original Images")
ax1.set_title("original image (RGB)")
ax1.imshow(array_rgb.transpose(1, 2, 0))
ax1.set_xlabel(f"{array_rgb.shape}")

ax2.set_title("original image (HSV)")
ax2.imshow(array_hsv.transpose(1, 2, 0))
ax2.set_xlabel(f"{array_hsv.shape}")

# Fig 2
fig2.suptitle("Masked Images (Keeping Yellow and White colors)")
ax3.set_title("masked image (HSV)")
ax3.imshow(array_hsv_masked.transpose(1, 2, 0))
ax3.set_xlabel(f"{array_hsv_masked.shape}")

ax4.set_title("masked image (RGB)")
ax4.imshow(array_rgb_masked.transpose(1, 2, 0))
ax4.set_xlabel(f"{array_rgb_masked.shape}")

ax5.set_title("masked image (BN)")
ax5.imshow(array_bn_masked, cmap='gray')
ax5.set_xlabel(f"{array_bn_masked.shape}")

# Fig 3
fig3.suptitle("Edge Maps")
ax6.set_title("vertical edge map (GS)")
ax6.imshow(vertical_edges, cmap='gray')
ax6.set_xlabel(f"{vertical_edges.shape}")

ax7.set_title("horizontal edge map (GS)")
ax7.imshow(horizontal_edges, cmap='gray')
ax7.set_xlabel(f"{horizontal_edges.shape}")

ax8.set_title("combined edge map (GS)")
ax8.imshow(edges, cmap='gray')
ax8.set_xlabel(f"{edges.shape}")

ax9.set_title("binarized edge map (GS)")
ax9.imshow(edges_bn, cmap='gray')
ax9.set_xlabel(f"{edges_bn.shape}")

# Fig 4
fig4.suptitle("Images with Line overlays")
ax10.set_title("binarized edge map (GS) with lines")
ax10.imshow(edges_bn, cmap='gray')
ax10.set_xlabel(f"{edges_bn.shape}")

ax11.set_title("original image (RGB) with lines")
ax11.imshow(array_rgb.transpose(1, 2, 0))
ax11.set_xlabel(f"{array_rgb.shape}")

for xcoords, ycoords in hough.lines:
    ax10.plot(xcoords, ycoords, color='forestgreen')
    ax11.plot(xcoords, ycoords, color='forestgreen')

for ax in axes:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

for fig in figs:
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.01)


plt.show()
