import numpy as np


class Hough():
    # self.fx = lambda rho, theta, x: (rho - x * np.cos(theta)) / np.sin(theta)
    # self.fy = lambda rho, theta, y: (rho - y * np.sin(theta)) / np.cos(theta)
    ''' Hough Transform class. Performs the hough transform on a binary image to detect lines fitting specified rho, theta parameters within a specified threshold of occurences'''
    def __init__(self, image):
        ''' Initializes the image input to the Hough class'''
        # set image
        self.set_image(image)

    def set_image(self, image):
        ''' Finds the coords of the peaks of the image input, as well as the maximum distance from the origin'''

        height, width = image.shape

        # max distance from origin
        max_distance = np.sqrt(height ** 2 + width ** 2)
        max_distance = int(np.ceil(max_distance))

        # find sorted unique values and counts, assert only 2 values (binary)
        unique_values, counts = np.unique(image, return_counts=True)
        assert unique_values.size == 2
        low, high = unique_values
        n_lows, n_highs = counts

        # get indices of high values
        point_coords = np.where(image == high)

        self.point_coords = point_coords
        self.n_points = n_highs
        self.max_distance = max_distance

    def transform(self, rho_res=1, theta_res=1, rho_lims=None, theta_lims=None):
        ''' Perform a vectorized hough transform operation on the set of high points of the binary image to result in a hough accumulator matrix'''

        # rho range
        D = self.max_distance
        if rho_lims is None:
            rho_min, rho_max = -D, D
        else:
            rho_min, rho_max = rho_lims
        rho_range = np.arange(rho_min, rho_max + rho_res, rho_res)
        n_rhos = rho_range.size

        # theta range
        if theta_lims is None:
            theta_min, theta_max = -90, 90
        else:
            theta_min, theta_max = theta_lims
        theta_range = np.arange(theta_min, theta_max + theta_res, theta_res) * (np.pi / 180)
        n_thetas = theta_range.size

        # compute rhos matrix via dot product of theta basis matrix and point coords matrix
        ycoords, xcoords = self.point_coords
        theta_basis = np.array([np.cos(theta_range), np.sin(theta_range)]).T # shape (n_thetas x 2)
        point_coords = np.array([xcoords, ycoords]) # (2 x n_points)
        rhos = np.matmul(theta_basis, point_coords) # (n_thetas x n_points)

        # compute rhos indices matrix by binning values in rhos matrix to values in rho range
        rhos_ = np.digitize(rhos, rho_range, right=True) # (n_thetas x n_points)

        # iterate through rows of rhos matrix. Each row corresponds to a single theta value, each column corresponds to a single high point in the binary image. By counting unique rho values in each row and their count, we are left with the number of points that intersect at each (rho, theta) pair
        accumulator = np.zeros((n_rhos, n_thetas)) # (n_rhos x n_thetas)
        coords_list = [[[] for j in range(n_thetas)] for i in range(n_rhos)]
        for theta_index, rhos in enumerate(rhos_):
            rho_uniques, rho_counts = np.unique(rhos, return_counts=True)
            # print(rho_uniques, rho_counts)
            accumulator[rho_uniques, theta_index] = rho_counts
            i = 0
            sorted_rhos_indices = np.argsort(rhos)
            for rho_unique, rho_count in zip(rho_uniques, rho_counts):
                point_indices = sorted_rhos_indices[i: i+rho_count]
                # print(f"{rho_count} points on line {rho_unique, theta_index} with indices {point_indices}")
                i += rho_count
                coords_list[rho_unique][theta_index] = point_indices

        self.rho_range = rho_range
        self.theta_range = theta_range
        self.accumulator = accumulator
        self.coords_list = coords_list

    def find_peaks(self, threshold_min=0.5, threshold_max=1):
        ''' Finds the peaks in the hough accumulator matrix that lie within a specified threshold range as a percentage of the maximum value'''
        accumulator = self.accumulator
        max_intersections = np.max(accumulator)
        threshold_min *= max_intersections # min value
        threshold_max *= max_intersections # max value

        peaks = np.where((accumulator >= threshold_min) & (accumulator <= threshold_max))
        n_lines = peaks[0].size

        self.peaks = peaks
        self.n_lines = n_lines

    def compute_lines(self):
        ''' Finds the lines corresponding to each rho, theta pair in peaks'''
        peaks = self.peaks
        rho_indices, theta_indices = peaks

        rho_range = self.rho_range
        theta_range = self.theta_range
        rhos = rho_range[rho_indices]
        thetas = theta_range[theta_indices]

        # for rho, theta in zip(rhos, thetas):
            # print(rho, theta)

        coords_list = self.coords_list
        point_coords = self.point_coords
        ycoords, xcoords = point_coords
        # print(f"xcoords_: {xcoords_}")
        # print(f"ycoords_: {ycoords_}")
        # print(f"POINT COORDS: {point_coords}")
        lines_coords = []
        for rho_index, theta_index in zip(rho_indices, theta_indices):
            # print("-" * 100)
            rho = rho_range[rho_index]
            theta = theta_range[theta_index] * (180/np.pi)
            point_indices = coords_list[rho_index][theta_index]
            point_min = np.min(point_indices)
            point_max = np.max(point_indices)
            xmin, ymin = xcoords[point_min], ycoords[point_min]
            xmax, ymax = xcoords[point_max], ycoords[point_max]
            # print(xcoords)
            # print(ycoords)
            # print(point_indices)
            # print(xcoords)
            # print(ycoords)
            # print("-" * 100)
            # xmin, xmax = np.min(xcoords), np.max(xcoords)
            # ymin, ymax = np.min(ycoords), np.max(ycoords)
            lines_coords.append((xmin, xmax, ymin, ymax))
            # print(f"Rho: {rho}, Theta: {theta}, Points: {(xmin, ymin), (xmax, ymax)}")
        # print("-" * 100)
        lines_coords = list(set(lines_coords))
        # print(lines_coords)
        lines_xcoords = []
        lines_ycoords = []
        for line_coords in lines_coords:
            lines_xcoords.append(line_coords[:2])
            lines_ycoords.append(line_coords[2:])
        # print(lines_xcoords)
        # print(lines_ycoords)

        self.lines_xcoords = lines_xcoords
        self.lines_ycoords = lines_ycoords
        self.rho_indices = rho_indices
        self.theta_indices = theta_indices
        self.rhos = rhos
        self.thetas = thetas


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import os
    from dzlib.signal_processing.sweep2d import SWP2d
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("Qt5Agg")

    # load grayscale (gs) image
    image_path = '/'.join((os.getcwd(), 'data/image9.png'))
    image = Image.open(image_path).convert('L')
    height, width = image.size
    k = 2
    image = image.resize((int(height/k), int(width/k)))
    height, width = image.size
    image_gs = np.asarray(image).astype(np.float32)
    # image_gs = image_gs[30:500, 86:811]
    image_gs = np.flip(image_gs, axis=(0))
    height, width = image_gs.shape

    # normalize image
    image_gs = image_gs - np.min(image_gs)
    image_gs = image_gs / np.max(image_gs)

    print("Beginning Edge Detection...")
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
    z = 1 # std multiplier
    image_bn[(image_bn < mean - z*std)] = 1
    image_bn[(image_bn >= mean - z*std) & (image_bn <= mean + z*std)] = 0
    image_bn[(image_bn > mean + z*std)] = 1
    image = image_bn
    n_points = image[image == 1].size
    print("Finished Edge Detection...")

    # image = np.zeros((50, 50))
    # image[10:40, 10:40] = np.eye(30)
    # image[10, 10:40] = 1
    # image[10:40, 40] = 1
    # image[10:30, 10] = 1
    # n_points = image[image == 1].size

    # image_path = '/'.join((os.getcwd(), 'data/image8.png'))
    # image = Image.open(image_path).convert('L')
    # height, width = image.size
    # image_gs = np.asarray(image).astype(np.float32)
    # # image_gs = image_gs[25:500, 86:811]
    # image_gs = np.flip(image_gs, axis=(0))
    # height, width = image_gs.shape

    # # normalize image
    # image_gs = image_gs - np.min(image_gs)
    # image_gs = image_gs / np.max(image_gs)
    # image = image_gs

    # # binarize
    # image[image == 0] = 2
    # image[image == 1] = 0
    # image[image == 2] = 1
    # n_points = image[image == 1].size

    print(f"n_points: {n_points}")
    print(f"image shape: {image.shape}")

    print("Beginning Hough Transform...")
    hough = Hough(image)
    hough.transform(rho_res=1, theta_res=0.5, theta_lims=(-25, 25))
    hough.find_peaks(threshold_min=0.7, threshold_max=1)
    hough.compute_lines()
    print("Finished Hough Transform...")
    print(f"n lines: {hough.n_lines}")
    lines_xcoords = hough.lines_xcoords
    lines_ycoords = hough.lines_ycoords

    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax1, ax2, ax3 = ax
    # ax1.hist(hough.accumulator.flatten(), bins=100)
    # ax2.imshow(hough.accumulator)
    # ax3.imshow(image_bn, cmap='binary', origin='lower')
    # # ax3.imshow(image, cmap='binary', origin='lower')
    # for line_xcoords, line_ycoords in zip(lines_xcoords, lines_ycoords):
    #     ax3.plot(line_xcoords, line_ycoords, color='r')

    print("Beginning Plotting...")
    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()
    # ax1.imshow(image_bn, cmap='binary', origin='lower')
    ax1.imshow(image_gs, cmap='gray', origin='lower', interpolation=None)
    ax2.imshow(edges, cmap='gray', origin='lower', interpolation=None)
    ax3.imshow(image_bn, cmap='binary', origin='lower', interpolation=None)
    ax4.imshow(image_gs, cmap='gray', origin='lower', interpolation=None)
    ax5.imshow(edges, cmap='gray', origin='lower', interpolation=None)
    ax6.imshow(image_bn, cmap='binary', origin='lower', interpolation=None)

    for line_xcoords, line_ycoords in zip(lines_xcoords, lines_ycoords):
        ax4.plot(line_xcoords, line_ycoords, color='g')
        ax5.plot(line_xcoords, line_ycoords, color='g')
        ax6.plot(line_xcoords, line_ycoords, color='g')

    ax1.set_title("Grayscale Original Image")
    ax2.set_title("Grayscale Edge Map")
    ax3.set_title("Binary Edge Map")

    fig, ax = plt.subplots(1)
    ax.imshow(hough.accumulator, interpolation=None)
    ax.set_aspect('auto')
    ax.set_title("Hough Accumulator")
    print("Finished Plotting...")


    plt.show()















