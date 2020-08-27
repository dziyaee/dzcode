import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from dzlib.signal_processing import sweep2d, kernels, edge_thinning


mpl.use("Qt5Agg")


def imshow(axis, array, **kwargs):
    '''axis imshow for (H x W) or (3 x H x W) array with some default kwargs'''

    defaults = {'extent': [0, array.shape[-1], 0, array.shape[-2]], 'interpolation': None}
    kwargs = {**defaults, **kwargs}

    try:
        axis.imshow(array, **kwargs)
    except TypeError:
        axis.imshow(array.transpose(1, 2, 0), **kwargs)

    axis.set_xlabel(array.shape)
    return None


def set_titles(axes, titles):
    '''set axis titles'''
    try:
        for axis, title in zip(axes.flatten(), titles):
            axis.set_title(title)

    except AttributeError:
        axis.set_title(*titles)

    return None


def tick_params(axes, **kwargs):
    '''set axis tick params with some defaults'''
    defaults = {'left': False, 'labelleft': False, 'bottom': False, 'labelbottom': False}
    kwargs = {**defaults, **kwargs}
    try:
        for axis in axes.flatten():
            axis.tick_params(**kwargs)

    except AttributeError:
        axis.tick_params(**kwargs)

    return None


def subplots_adjust(fig, **kwargs):
    '''set subplot params with some defaults'''
    defaults = {'left': 0.01, 'right': 0.99, 'wspace': 0.01, 'bottom': 0.05, 'top': 0.95, 'hspace': 0.2}
    kwargs = {**defaults, **kwargs}
    fig.subplots_adjust(**kwargs)
    return None


def hsv(*values):
    '''Returns 3-tuple of HSV values normalized to interval [0, 1] from human-readable HSV values'''
    norm = (360, 100, 100)
    return tuple([v / n for v, n in zip(values, norm)])


def mask_3channel(array, mins, maxs):
    '''Returns a boolean mask of a 3-channel array with condition that all 3 values per channel are within a min and max range'''
    mask_min = ((array[0] >= mins[0]) & (array[1] >= mins[1]) & (array[2] >= mins[2]))
    mask_max = ((array[0] <= maxs[0]) & (array[1] <= maxs[1]) & (array[2] <= maxs[2]))
    return mask_min & mask_max


def hsv_to_gs(array):
    '''Returns grayscale channel of an HSV array'''
    return array[2].copy()


def normalize(array):
    '''Returns array normalized to interval [0, 1]'''
    array = array - np.min(array)
    return array / np.max(array)


# arrays are commented with shape and min, max interval as follows: (N x M x ...), [min... max]
# IMAGE LOADING -------------------------------------------------------------

# load image
image_path = "data/image01.png"
pil_rgb = Image.open(image_path).convert("RGB")  # (W x H x 3)
image_rgb = np.asarray(pil_rgb)  # (H x W x 3), [0... 255]
image_rgb = normalize(image_rgb)  # (H x W x 3), [0... 1]
image_hsv = rgb_to_hsv(image_rgb)  # (H x W x 3)

image_rgb = image_rgb.transpose(2, 0, 1)  # (3 x H x W), [0... 1]
image_hsv = image_hsv.transpose(2, 0, 1)  # (3 x H x W), [0... 1]
image_gs = hsv_to_gs(image_hsv)  # (H x W), [0... 1]


# recurring parameters (used by Sweep2d class)
padding = (0, 0)
stride = (1, 1)
mode = "same"


# IMAGE MASKING -------------------------------------------------------------
# I used this website http://colorizer.org/ to settle on mins / maxs for yellow and white HSV values

# yellow
ymins = hsv(40, 70, 70)
ymaxs = hsv(50, 100, 100)

# white
wmins = hsv(0, 0, 70)
wmaxs = hsv(360, 10, 100)

# mask
masked_hsv = image_hsv.copy()  # (3 x H x W)
yellow_mask = mask_3channel(masked_hsv, ymins, ymaxs)  # mask with all non-yellow pixel values set to False
white_mask = mask_3channel(masked_hsv, wmins, wmaxs)  # mask with all non-white pixel values set to False
mask = ~(yellow_mask | white_mask)  # mask with all non-yellow or non-white pixels set to True
masked_hsv[:, mask] = 0

masked_rgb = hsv_to_rgb(masked_hsv.transpose(1, 2, 0))  # (H x W x 3)
masked_rgb = masked_rgb.transpose(2, 0, 1)  # (3 x H x W)
masked_gs = hsv_to_gs(masked_hsv)  # (H x W)


# IMAGE DENOISING (Gaussian Blur / Gaussian Smoothing) ------------------------
images = masked_gs[None, None]  # (1 x 1 x H x W)
gaussian_kernel = kernels.gaussian2d(size=5, sigma=1)[None, None]  # (1 x 1 x 5 x 5)
sweeper = sweep2d.Sweep2d(images.shape, gaussian_kernel.shape, padding, stride, mode)
blurred_gs = sweeper.convolve2d(images, gaussian_kernel)  # (1 x 1 x H x W)
blurred_gs = normalize(blurred_gs[0, 0, :, :])  # (H x W), [0... 1]


# IMAGE GRADIENT INTENSITIES & DIRECTIONS (Sobel Operator) ---------------------
images = blurred_gs[None, None]  # (1 x 1 x H x W)
sobel_operator = kernels.sobel_operator()  # (2 x 1 x 3 x 3)
sweeper = sweep2d.Sweep2d(images.shape, sobel_operator.shape, padding, stride, mode)
Gx, Gy = sweeper.convolve2d(images, sobel_operator)[0]  # 2 x (H x W)
gradient_magnitudes = np.sqrt(Gx ** 2 + Gy ** 2)  # (H x W)
gradient_magnitudes = normalize(gradient_magnitudes)  # [0... 1]
gradient_angles = np.arctan2(Gy, Gx)  # (H x W), [-pi... pi]


# EDGE THINNING via Non-Maximal Suppression (NMS) (without Interpolation)-----
thinned_gs = edge_thinning.non_maximal_suppression(gradient_magnitudes, gradient_angles)  # (H x W), [0... 1]


# BINARIZE EDGE MAP (FOR DISPLAY PURPOSES ONLY)
thinned_bn = thinned_gs.copy()
thinned_bn[thinned_bn <= np.mean(thinned_bn) + 10 * np.std(thinned_bn)] = 0
thinned_bn[thinned_bn != 0] = 1


# IMAGE PLOTTING -------------------------------------------------------------

# Fig 1
fig1, axes1 = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharex=True, sharey=True)
titles1 = ("Image (RGB)", "Image (HSV)", "Image (GS)", "Masked (RGB)", "Masked (HSV)", "Masked (GS)")
ax1, ax2, ax3 = axes1[0]
ax4, ax5, ax6 = axes1[1]

subplots_adjust(fig1)
tick_params(axes1)
set_titles(axes1, titles1)

# Row 1 (Original Images)
imshow(ax1, image_rgb, cmap=None)
imshow(ax2, image_hsv, cmap='hsv')
imshow(ax3, image_gs, cmap='gray')

# Row 2 (Masked Images)
imshow(ax4, masked_rgb, cmap=None)
imshow(ax5, masked_hsv, cmap='hsv')
imshow(ax6, masked_gs, cmap='gray')


# Fig 2
fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharex=True, sharey=True)
titles2 = ("Masked (GS)", "Horizontal Gradients (GS)", "Gradient Magnitudes (GS)", "Blurred (GS)", "Vertical Gradients (GS)", "Gradient Angles")
ax1, ax3, ax5 = axes2[0]
ax2, ax4, ax6 = axes2[1]

subplots_adjust(fig2)
tick_params(axes2)
set_titles(axes2, titles2)

# Col 1 (Masked & Blurred Images)
imshow(ax1, masked_gs, cmap='gray')
imshow(ax2, blurred_gs, cmap='gray')

# Col 2 (Horizontal & Vertical Gradient Intensities)
imshow(ax3, Gy, cmap='gray')
imshow(ax4, Gx, cmap='gray')

# Col 3 (Gradient Intensity Magnitudes & Gradient Angles)
imshow(ax5, gradient_magnitudes, cmap='gray')
imshow(ax6, gradient_angles, cmap=None)


# Fig 3
fig3, axes3 = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharex=True, sharey=True)
titles3 = ("Image (RGB)", "Masked (GS)", "Gradient Magnitudes (GS)", "Image (GS)", "Blurred (GS)", "Thinned (GS)")
ax1, ax3, ax5 = axes3[0]
ax2, ax4, ax6 = axes3[1]

subplots_adjust(fig3)
tick_params(axes3)
set_titles(axes3, titles3)

# Col 1 (Original Images)
imshow(ax1, image_rgb, cmap=None)
imshow(ax2, image_gs, cmap='gray')

# Col 2 (Masked & Blurred Images)
imshow(ax3, masked_gs, cmap='gray')
imshow(ax4, blurred_gs, cmap='gray')

# Col 3 (Gradient Intensity Magnitudes and Thinned Edge Image)
imshow(ax5, gradient_magnitudes, cmap='gray')
imshow(ax6, thinned_gs, cmap='gray')


# Fig 4
fig4, axes4 = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharex=True, sharey=True)
titles3 = ("Image (RGB)", "Blurred (GS)", "Thinned (GS)", "Image (GS)", "Gradient Magnitudes (GS)", "Thinned (BN)")
ax1, ax3, ax5 = axes4[0]
ax2, ax4, ax6 = axes4[1]

subplots_adjust(fig4)
tick_params(axes4)
set_titles(axes4, titles3)

# Col 1 (Original Images)
imshow(ax1, image_rgb, cmap=None)
imshow(ax2, image_gs, cmap='gray')

# Col 2 (Masked & Blurred Images)
imshow(ax3, blurred_gs, cmap='gray')
imshow(ax4, gradient_magnitudes, cmap='gray')

# Col 3 (Gradient Intensity Magnitudes and Thinned Edge Image)
imshow(ax5, thinned_gs, cmap='gray')
imshow(ax6, thinned_bn, cmap='binary')

plt.show()


# # # LINE DETECTION -----------
# rho_step = 1
# theta_step = 1
# theta_min = -85
# theta_max = 85
# threshold = 0.9
# hough = Hough(edges_bn, rho_step, theta_step, (theta_min, theta_max))
# hough.transform(threshold)
# print(hough.n_lines)
# for xcoords, ycoords in hough.lines:
#     ax10.plot(xcoords, ycoords, color='forestgreen')
#     ax11.plot(xcoords, ycoords, color='forestgreen')
