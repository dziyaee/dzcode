import numpy as np
from dzlib.common.utils import quantize


class Hough():
    def __init__(self, image, rho_step, theta_step, theta_lims=(-90, 90)):
        self.set_image(image)
        rho_lims = -self.max_distance, self.max_distance
        self.set_ranges(rho_step, rho_lims, theta_step, theta_lims)

    def set_image(self, image):

        # max distance
        height, width = image.shape
        max_distance = np.sqrt(height ** 2 + width ** 2)
        max_distance = int(np.ceil(max_distance))

        # coordinates of all points in image
        ycoords, xcoords = np.where(image == 1)
        n_points = ycoords.size

        self.max_distance = max_distance
        self.ycoords = ycoords
        self.xcoords = xcoords
        self.n_points = n_points

    def set_ranges(self, rho_step, rho_lims, theta_step, theta_lims):

        # rho range (d = distance)
        Dmin, Dmax = rho_lims
        dx = rho_step
        rho_range = np.arange(Dmin, Dmax + dx, dx)

        # theta range (a = angle)
        theta_lims = np.clip(theta_lims, -90, 90)
        Amin, Amax = theta_lims
        da = theta_step
        theta_range = np.arange(Amin, Amax + da, da) * (np.pi/180)

        self.rho_step = dx
        self.rho_range = rho_range
        self.n_rhos = rho_range.size

        self.theta_step = da
        self.theta_range = theta_range
        self.n_thetas = theta_range.size

    def transform(self, threshold):

        # theta basis matrix (n_thetas x 2)
        theta_range = self.theta_range
        theta_basis = np.array([np.cos(theta_range), np.sin(theta_range)]).T

        # coords matrix (2 x n_points)
        xcoords, ycoords = self.xcoords, self.ycoords
        points = np.array([xcoords, ycoords])

        # rho matrix (n_thetas x n_points)
        rhos = np.matmul(theta_basis, points)
        Dmin, Dmax = -self.max_distance, self.max_distance
        dx = self.rho_step
        rhos = quantize(rhos, Dmin, Dmax, dx) # returns indices

        # init hough accumulator (zeros 2d matrix) and colinear points (empty 2d list)
        n_rhos, n_thetas = self.n_rhos, self.n_thetas
        accumulator = np.zeros((n_rhos, n_thetas))
        colinear_points = [[[] for theta in range(n_thetas)] for rho in range(n_rhos)]

        # iterate through rows of rhos and populate accumulator with unique rho counts
        for theta_index, rhos_ in enumerate(rhos):
            unique_rhos, counts = np.unique(rhos_, return_counts=True)
            accumulator[unique_rhos, theta_index] = counts

            # iterate through each rho value and populate colinear points with list of colinear point indices
            i = 0
            rhos_ = np.argsort(rhos_) # returns indices
            for unique_rho, count in zip(unique_rhos, counts):
                points = rhos_[i: i + count] # returns indices
                colinear_points[unique_rho][theta_index] = points
                i += count

        # get rho, thetas where accumulator threshold is met
        rhos, thetas = self.get_peaks(accumulator, threshold)

        # get lines from rhos, thetas, colinear points
        lines = self.get_lines(rhos, thetas, colinear_points)
        n_lines = len(lines)

        self.accumulator = accumulator
        self.colinear_points = colinear_points
        self.lines = lines
        self.n_lines = n_lines

    def get_peaks(self, accumulator, threshold):
        threshold *= np.max(accumulator)
        rhos, thetas = np.where(accumulator >= threshold)

        return rhos, thetas

    def get_lines(self, rhos, thetas, colinear_points):

        # get list of all points xcoords and ycoords
        xcoords, ycoords = self.xcoords, self.ycoords

        # returns signs of slopes of lines defined by the normal-coord thetas within span of quadrants 4 and 1 (-90 to +90)
        theta_range = self.theta_range
        slope_signs = [-1 if theta >= 0 else 1 for theta in theta_range[thetas]]

        # iterate through rho, theta, and slope sign arrays and populate lines list with tuple of (x, y) coordinate pairs
        lines = []
        for rho, theta, i in zip(rhos, thetas, slope_signs):

            # points at current (rho, theta) pair
            points = colinear_points[rho][theta]
            xs = xcoords[points]
            ys = ycoords[points]

            # get min and max (x, y) coords for points defining longest segment
            xmin, xmax = np.min(xs), np.max(xs)
            ymin, ymax = np.min(ys), np.max(ys)

            # reverse y coords if slope is negative (because ymin by definition will pair with xmax in a negative slope line)
            xs = [xmin, xmax]
            ys = [ymin, ymax][::i] # if i = -1, reverses list, if i = +1, preserves order

            lines.append((xs, ys))

        return lines


# test
if __name__ == "__main__":
    from PIL import Image
    import os
    from dzlib.signal_processing.sweep2d import SWP2d
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    mpl.use("Qt5Agg")

    # Image Params
    k = 1 # scale factor for resizing of original image
    z = 0.04 # value around mean to set to zero for binarization of edge map

    # Hough Transform Params
    rho_step=1
    theta_step=1
    theta_min = -90
    theta_max = 90
    threshold= 0.45

    # load grayscale (gs) image
    print(f"\nLoading and resizing image...")
    image_path = '/'.join((os.getcwd(), 'data/image7.png'))
    image = Image.open(image_path).convert('L')
    width, height = image.size
    image = image.resize((int(width/k), int(height/k)))
    image_gs = np.asarray(image).astype(np.float32)
    image_gs = np.flip(image_gs, axis=(0))
    print(f"original image shape: {height, width}")
    print(f"resizing scale factor: {k}")
    print(f"resized image shape:   {image_gs.shape}")

    # normalize gs image
    image_gs = image_gs - np.min(image_gs)
    image_gs = image_gs / np.max(image_gs)

    # edge detection (Sobel Operator)
    print("\nEdge Detection...")
    height, width = image_gs.shape
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
    mean = np.mean(edges)
    print(f"Edge Map Mean: {mean:.3f}")

    print(f"\nBinarizing Edge Map...")
    # binarize edge map
    image_bn = edges.copy()
    image_bn[(image_bn < mean - z)] = 1
    image_bn[(image_bn >= mean - z) & (image_bn <= mean + z)] = 0
    image_bn[(image_bn > mean + z)] = 1
    image = image_bn
    n_points = image[image == 1].size
    print(f"Values between {(mean-z):.3f} and {(mean+z):.3f} set to zero. All other values set to 1")
    print(f"number of non-zero points: {n_points}")

    print("\nHoughing...")
    hough = Hough(image=image, rho_step=rho_step, theta_step=theta_step, theta_lims=(theta_min, theta_max))
    hough.transform(threshold=threshold)
    print(f"number of lines: {hough.n_lines}")

    print("\nPlotting...")
    fig = plt.figure(1)
    gs = GridSpec(nrows=2, ncols=8)

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[1, 0:2])

    ax3 = fig.add_subplot(gs[0, 2:4])
    ax4 = fig.add_subplot(gs[1, 2:4])

    ax5 = fig.add_subplot(gs[:, 5:])

    ax1.imshow(image_gs,    cmap='gray',    origin='lower', interpolation=None)
    ax2.imshow(image_gs,    cmap='gray',    origin='lower', interpolation=None)

    ax3.imshow(image_bn,    cmap='binary',  origin='lower', interpolation=None)
    ax4.imshow(image_bn,    cmap='binary',  origin='lower', interpolation=None)
    ax3.tick_params(left=False, labelleft=False)
    ax4.tick_params(left=False, labelleft=False)

    for xcoords, ycoords in hough.lines:
        ax2.plot(xcoords, ycoords, color='forestgreen')
        ax4.plot(xcoords, ycoords, color='forestgreen')

    ax5.imshow(hough.accumulator, interpolation=None, extent=[theta_min, theta_max, hough.max_distance, -hough.max_distance])
    ax5.set_aspect('auto')
    ax5.tick_params(left=False, labelleft=False, right=True, labelright=True)
    ax5.set_xlabel(r"Angle from x-axis ($\theta$)")
    ax5.set_ylabel(r"Displacement from Origin ($\rho$)", rotation=90)
    ax5.yaxis.set_label_position('right')

    fig.suptitle("Edge Detection via Sobel Operator\nLine Detection via Hough Transform")
    ax1.set_title("Grayscale (GS)\nOriginal Image")
    ax2.set_title("With Hough lines overlay")
    ax3.set_title("Binary (BN)\nEdge Map")
    ax4.set_title("With Hough lines overlay")
    ax5.set_title("Hough Accumulator")

    fig.subplots_adjust(left=0.05, right=0.9, wspace=0)
    print(r"Complete")
    plt.show()
