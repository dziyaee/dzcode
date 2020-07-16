import numpy as np


class Hough():
    def __init__(self, image):
        """Summary

        Args:
            image (Numpy Array): Shape (HxW). Expecting a binary edge map
        """

        # set image
        self.set_image(image)

    def set_image(self, image):
        height, width = image.shape
        max_distance = np.sqrt(height ** 2 + width ** 2)
        max_distance = int(np.ceil(max_distance))
        point_coords = np.where(image == 1)
        n_points = image[image == 1].size

        self.point_coords = point_coords
        self.n_points = n_points
        self.max_distance = max_distance


    def transform(self, rho_res=1, theta_res=1, rho_lims=None, theta_lims=None):

        # ranges
        # rhos
        D = self.max_distance
        if rho_lims is None:
            rho_min, rho_max = -D, D
        else:
            rho_min, rho_max = rho_lims
        rho_range = np.arange(rho_min, rho_max + rho_res, rho_res)
        n_rhos = rho_range.size

        # thetas
        theta_min, theta_max = theta_lims
        theta_range = np.arange(theta_min, theta_max + theta_res, theta_res) * (np.pi / 180)
        n_thetas = theta_range.size

        # compute rhos matrix via dot product of theta basis matrix and point coords matrix
        ycoords, xcoords = self.point_coords
        theta_basis = np.array([np.cos(theta_range), np.sin(theta_range)]).T # shape (n_thetas x 2)
        point_coords = np.array([xcoords, ycoords]) # (2 x n_points)
        rhos = np.matmul(theta_basis, point_coords) # (n_thetas x n_points)

        # compute rhos indices matrix by binning values in rhos matrix to values in rho range
        rhos = np.digitize(rhos, rho_range, right=True) # (n_thetas x n_points)

        # iterate through rows of rhos matrix, each row corresponds to a theta value and contains a rho value for each point. By counting unique rho values in each row and their count, we are left with the number of points that intersect at each (rho, theta) pair
        accumulator = np.zeros((n_rhos, n_thetas)) # (n_rhos x n_thetas)
        for theta_index, rho_indices in enumerate(rhos):
            rho_indices, rho_frequencies = np.unique(rho_indices, return_counts=True)
            accumulator[rho_indices, theta_index] = rho_frequencies

        self.rho_range = rho_range
        self.theta_range = theta_range
        self.accumulator = accumulator

    def find_peaks(self, threshold_min=0.5, threshold_max=1):
        accumulator = self.accumulator
        max_intersections = np.max(accumulator)
        threshold_min *= max_intersections
        threshold_max *= max_intersections

        peaks = np.where((accumulator >= threshold_min) & (accumulator <= threshold_max))
        n_lines = peaks[0].size

        self.peaks = peaks
        self.n_lines = n_lines

    def compute_lines(self):
        peaks = self.peaks
        rho_indices, theta_indices = peaks

        rho_range = self.rho_range
        theta_range = self.theta_range
        rhos = rho_range[rho_indices]
        thetas = theta_range[theta_indices]

        self.rhos = rhos
        self.thetas = thetas


if __name__ == "__main__":
    import numpy as np
    # from PIL import Image
    # import os
    # from dzlib.signal_processing.sweep2d import SWP2d
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("Qt5Agg")

    # # load grayscale (gs) image
    # image_path = '/'.join((os.getcwd(), 'data/image7.png'))
    # image = Image.open(image_path).convert('L')
    # height, width = image.size
    # image_gs = np.asarray(image).astype(np.float32)
    # image_gs = image_gs[25:500, 86:811]
    # image_gs = np.flip(image_gs, axis=0)
    # height, width = image_gs.shape

    # # normalize image
    # image_gs = image_gs - np.min(image_gs)
    # image_gs = image_gs / np.max(image_gs)

    # # edge detection (Sobel Operator)
    # edge_detector = SWP2d(image=image_gs.reshape(1, 1, height, width), kernel_size=(3, 3), mode='none')
    # gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) # vertical edge kernel
    # gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # horizontal edge kernel
    # kernel = np.array([gx, gy])
    # kernel = kernel.reshape(1, *kernel.shape)
    # kernel = kernel.transpose(1, 0, 2, 3) # 4d kernel input (axis0 = number of kernels, axis1 = 1 channel, axis2 = height, axis3 = width)
    # yedges, xedges = edge_detector.convolve(kernel) # outputs vertical and horizontal edge maps
    # edges = yedges + xedges # combine vertical and horizontal edge maps to result in final edge map

    # # normalize edge map
    # edges = edges - np.min(edges)
    # edges = edges / np.max(edges)

    # # binarize edge map
    # image_bn = edges.copy()
    # mean = np.mean(image_bn)
    # std = np.std(image_bn)
    # z = 1.5 # std multiplier
    # image_bn[(image_bn < mean - z*std)] = 1
    # image_bn[(image_bn >= mean - z*std) & (image_bn <= mean + z*std)] = 0
    # image_bn[(image_bn > mean + z*std)] = 1
    # image = image_bn
    # n_points = image[image == 1].size

    image = np.zeros((50, 50))
    image[10:40, 10:40] = np.eye(30)
    n_points = image[image == 1].size

    print(f"n_points: {n_points}")
    print(f"image shape: {image.shape}")
    hough = Hough(image)
    hough.transform()
    hough.find_peaks(threshold_min=0.9, threshold_max=1)
    hough.compute_lines()
    print(hough.n_lines)
    print(hough.rhos)
    print(hough.thetas*(180/np.pi))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax1, ax2 = ax
    ax1.hist(hough.accumulator.flatten(), bins=100)
    ax2.imshow(hough.accumulator)
    plt.show()















