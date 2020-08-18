import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
from dzlib.signal_processing.sweep2d import Sweep2d
from dzlib.signal_processing import hough
from dzlib.common.utils import stats, info


mpl.use("Qt5Agg")


# load image as array, normalize, transpose
image_path = "data/highway1.png"
image_rgb = Image.open(image_path).convert("RGB")  # (W x H x C)
array_rgb = np.asarray(image_rgb)  # (H x W x C)
array_rgb = array_rgb / 255




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

# convert masked RGB to grayscale for edge detection purposes
x = array_rgb_masked.copy()
x[x != 0] = 1
array_gs_masked = np.mean(x, axis=0)  # all channels of x have same values, so mean is taken to reduce down to 2d image
del x

# edge detection on grayscale image via Sobel operator kernels and Sweep2d class to perform the convolution operation
# idea from: https://en.wikipedia.org/wiki/Sobel_operator
# the results from the vertical and horizontal edge detections can be combined to produce a full image edge map
gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # vertical edge detection kernel (3 x 3)
gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # horizontal edge detection kernel (3 x 3)

# Sweep2d inputs
images = array_gs_masked[None, None]  # convert 1 2d grayscale image array to 1 4d array (1 x 1 x H x W)
kernels = np.array([[gx], [gy]])  # convert 2 2d kernel arrays to 1 4d array (2 x 1 x 3 x 3)
padding = (0, 0)
stride = (1, 1)
mode = "full"

# edge detection
sweeper = Sweep2d(images.shape, kernels.shape, padding, stride, mode)
edges = sweeper.convolve2d(images, kernels)  # (1 x 2 x H x W)
edges = np.sum(edges, axis=1)  # sum to combine vertical and horizontal edge maps into a full edge map image (1 x H x W)
edges = edges - np.min(edges)  # normalize by subtracting min and dividing by max to get values between [0... 1]
edges = edges / np.max(edges)
edges = edges[0]  # (H x W)

# binarize image to throw away mean values, leaving only extreme values (edges)
edges[(edges >= 0.49) & (edges <= 0.51)] = 0  # convert values around mean to 0
edges[edges != 0] = 1  # convert remaining non-zero values to 1


fig1, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True, figsize=(10, 8))
fig1.suptitle(f"Line Detection via Hough Transform Example: Highway")
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()


# Col 1
ax1.set_title("original image (RGB)")
ax1.imshow(array_rgb.transpose(1, 2, 0))
ax4.set_title("original image (HSV)")
ax4.imshow(array_hsv.transpose(1, 2, 0))
ax4.set_xlabel(f"{array_hsv.shape}")

# Col 2
ax2.set_title("masked image (RGB)")
ax2.imshow(array_rgb_masked.transpose(1, 2, 0))
ax5.set_title("masked image (HSV)")
ax5.imshow(array_hsv_masked.transpose(1, 2, 0))
ax5.set_xlabel(f"{array_hsv_masked.shape}")

# Col 3
ax3.set_title("masked image (GS)")
ax3.imshow(array_gs_masked, cmap='gray')
ax3.set_xlabel(f"{array_gs_masked.shape}")
ax6.set_title("binarized edge map (GS)")
ax6.imshow(edges, cmap='gray')
ax6.set_xlabel(f"{edges.shape}")


for ax in axes.flatten():
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# fig2, ax = plt.subplots(1)
# ax.hist(edges.flatten(), bins=100)


plt.show()
