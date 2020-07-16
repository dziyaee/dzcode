import numpy as np
from numpy import pi, cos, sin
from PIL import Image
import os
from dzlib.common.utils import stats
from dzlib.signal_processing.sweep2d import SWP2d
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
mpl.use("Qt5Agg")


# load grayscale (gs) image
image_path = '/'.join((os.getcwd(), 'data/image6.png'))
image = Image.open(image_path).convert('L')
height, width = image.size
image_gs = np.asarray(image).astype(np.float32)
image_gs = image_gs[25:500, 86:811]
image_gs = np.flip(image_gs, axis=0)
height, width = image_gs.shape

# normalize image
image_gs = image_gs - np.min(image_gs)
image_gs = image_gs / np.max(image_gs)

# edge detection (Sobel Operator)
edge_detector = SWP2d(image=image_gs.reshape(1, 1, height, width), kernel_size=(3, 3), mode='none')
gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) # vertical edge kernel
gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # horizontal edge kernel
kernel = np.array([gx, gy])
kernel = kernel.reshape(1, *kernel.shape)
kernel = kernel.transpose(1, 0, 2, 3) # 4d kernel input (axis0 = number of kernels, axis1 = 1 channel, axis2 = height, axis3 = width)
yedges, xedges = edge_detector.convolve(kernel) # outputs vertical and horizontal edge maps
edges = yedges + xedges # combine vertical and horizontal edge maps to result in final edge map

# normalize edge map
edges = edges - np.min(edges)
edges = edges / np.max(edges)

# binarize edge map
image_bn = edges.copy()
mean = np.mean(image_bn)
std = np.std(image_bn)
z = 1.5 # std multiplier
image_bn[(image_bn < mean - z*std)] = 1
image_bn[(image_bn >= mean - z*std) & (image_bn <= mean + z*std)] = 0
image_bn[(image_bn > mean + z*std)] = 1
image = image_bn
n_points = image[image == 1].size
print(f"n_points: {n_points}")
print(f"image shape: {image.shape}")

D = (height ** 2 + width ** 2) ** 0.5
D = int(np.ceil(D))

n_rhos = 1 + D*2 # odd number ideal
rho_range = np.linspace(-D, D, n_rhos)

T = 90
n_thetas = 361 #181
theta_range = np.linspace(T-180, T, n_thetas) * (pi/180)

ycoords, xcoords = np.where(image == 1)
print(f"max distance: {D}")
print(f"n rhos: {n_rhos}")
print(f"n thetas: {n_thetas}")
# print(f"rho range:   {rho_range}")
# print(f"point xcoords: {xcoords}")
# print(f"point ycoords: {ycoords}")

hough = np.zeros((n_rhos, n_thetas))
thetas_idx = np.arange(n_thetas)


basis = np.array([cos(theta_range), sin(theta_range)]).T
print(basis.shape)

coords = np.array([xcoords, ycoords])
print(coords.shape)

rhos = np.matmul(basis, coords)
print(rhos.shape)

# diff = np.subtract.outer(rhos, rho_range)
# diff = np.abs(diff)
# rhos_idx = np.argmin(diff, axis=2)
rhos_idx_matrix = np.digitize(rhos, rho_range, right=True)
print(rhos_idx_matrix.shape)

for theta_idx, rhos_idx in enumerate(rhos_idx_matrix):
    rho_idx, freq = np.unique(rhos_idx, return_counts=True)
    # print(rho_idx.shape, freq.shape)
    # print(hough.shape)
    hough[rho_idx, theta_idx] = freq


# pdb.set_trace()
# for i, (x, y) in enumerate(zip(xcoords, ycoords)):
#     if i % 1000 == 0:
#         print(f"{i}/{n_points}")

#     rhos = x * cos(theta_range) + y * sin(theta_range)
#     diff = np.subtract.outer(rhos, rho_range)
#     diff = np.abs(diff)
#     rhos_idx = np.argmin(diff, axis=1)
#     hough[rhos_idx, thetas_idx] += 1

threshold = 0.5*np.max(hough)
print(f"hough max: {np.max(hough)}")
print(f"threshold: {threshold}")
peaks = np.where(hough >= threshold)
rhos_idx, thetas_idx = peaks
rhos = rho_range[rhos_idx]
thetas = theta_range[thetas_idx]
print(f"n winners: {rhos.size}")
if rhos.size > 1000:
    assert False, f"too many lines to plot {rhos.size}"
else:

    fx = lambda rhos, thetas, xcoords: (rhos - xcoords * cos(thetas)) / sin(thetas)
    fy = lambda rhos, thetas, ycoords: (rhos - ycoords * sin(thetas)) / cos(thetas)

    xlims = np.array([-1, width]).astype(np.float32)
    ylims = np.array([-1, height]).astype(np.float32)
    xcoords = []
    ycoords = []

    for rho, theta in zip(rhos, thetas):
        # print(theta, sin(theta), cos(theta))
        if not np.isclose(cos(theta), 0):
            xcoords.append(list(fy(rho, theta, ylims)))
            ycoords.append(list(ylims))
        else:
            ycoords.append(list(fx(rho, theta, xlims)))
            xcoords.append(list(xlims))

    # print(f"line xcoords: {xcoords}")
    # print(f"line ycoords: {ycoords}")

    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()

    ax1.imshow(image_gs, cmap='gray', interpolation=None, origin='lower')
    for xcoord, ycoord in zip(xcoords, ycoords):
        ax4.plot(xcoord, ycoord, color='g')
    ax4.imshow(image_gs, cmap='gray', interpolation=None, origin='lower')

    ax2.imshow(image_bn, cmap='binary', interpolation=None, origin='lower')
    for xcoord, ycoord in zip(xcoords, ycoords):
        ax5.plot(xcoord, ycoord, color='g')
    ax5.imshow(image_bn, cmap='binary', interpolation=None, origin='lower')

    ax3.imshow(edges, cmap='gray', interpolation=None, origin='lower')

    ax6.imshow(hough, interpolation=None, extent=[T-180, T, D, -D])
    ax6.set_aspect('auto')

    fig, ax = plt.subplots(1)
    ax.hist(edges.flatten(), bins=100)
    plt.show()
