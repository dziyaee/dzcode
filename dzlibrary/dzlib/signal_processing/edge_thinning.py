import numpy as np


def non_maximal_suppression(gradient_magnitudes, gradient_angles):

    # copy arrays as changes will be made in-place during NMS iteration
    gradient_magnitudes = gradient_magnitudes.copy()  # (H x W)
    gradient_angles = gradient_angles.copy()

    # relative row/col indices to center pixel of 3x3 kernel in binned gradient direction
    row_map = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0])  # (9,)
    col_map = np.array([1, 1, 0, -1, -1, -1, 0, 1, 1])
    row_map = np.array([row_map, -row_map])  # (2 x 9)
    col_map = np.array([col_map, -col_map])

    # indices to all non-edge non-zero points for which to perform NMS on
    rows, cols = np.where(gradient_magnitudes[1:-1, 1:-1] != 0)  # 2 x (N,)
    rows += 1  # return indices within frame of reference of whole image (including edges)
    cols += 1

    #  get indices of each binned angle for use in row/col index map
    angles = gradient_angles[rows, cols]  # (N,)
    angles *= (180 / np.pi)
    angles += 180
    bins = np.arange(0-22.5, 360-22.5+45, 45)  # (9,)
    binned_indices = np.digitize(angles, bins[1:], right=False)  # (N,)

    relative_rows = row_map[:, binned_indices]  # (2 x N)
    relative_cols = col_map[:, binned_indices]

    actual_rows = (rows + relative_rows).T  # (N x 2)
    actual_cols = (cols + relative_cols).T

    magnitudes = gradient_magnitudes[rows, cols]  # (N,)

    for r, c, mag, rows_, cols_ in zip(rows, cols, magnitudes, actual_rows, actual_cols):
        neighbour1 = gradient_magnitudes[rows_[0], cols_[0]]
        neighbour2 = gradient_magnitudes[rows_[1], cols_[1]]
        if (mag <= neighbour1 or mag <= neighbour2):
            gradient_magnitudes[r, c] = 0

    return gradient_magnitudes



def non_maximal_suppression_with_interpolation(gradient_magnitudes, gradient_angles):
    # copy arrays as changes will be made in-place during NMS iteration
    gradient_magnitudes = gradient_magnitudes.copy()
    gradient_angles = gradient_angles.copy()

    # relative indices of 4 closest neighbour pixels to center pixel of a 3x3 window. maps to angle bin indices
    row_map = np.array([(1, 0), (1, 1), (1, 1), (1, 0), (1, 0)])
    col_map = np.array([(1, 1), (1, 0), (-1, 0), (-1, -1), (1, 1)])
    row_map = np.array([row_map, -row_map])
    col_map = np.array([col_map, -col_map])

    # non-edge non-zero gradient magnitudes
    rows, cols = np.where(gradient_magnitudes[1:-1, 1:-1] != 0)
    rows += 1  # normalize indices for use in full image
    cols += 1

    # get angle bin indices of all angles corresponding to non-edge non-zero gradient magnitudes
    angles = gradient_angles[rows, cols]
    angles[angles < 0] += np.pi
    bins = np.arange(0, np.pi+np.pi/4, np.pi/4)
    binned_indices = np.digitize(angles, bins[1:], right=False)

    # angle modulo 90 splits the space into two 90 degree halves (quadrants 1 and 2)
    # floor div 45 gives which half of a quadrant the angle is in (0 or 1)
    # multiply by 45 gives which base line the angle is being measured from (0 or 45)
    # angle modulo 45 splits the space into 4 quarters (quarters 1, 2, 3, 4)
    # subtract that angle from the base line angle to obtain the angle theta
    thetas = abs((angles % (np.pi/2) // (np.pi/4)) * (np.pi/4) - (angles % (np.pi/4)))

    # interpolant position and its complement for use in linear interpolation
    interpolant_positions = 1 * np.tan(thetas)
    interpolant_positions_ = 1 - interpolant_positions

    # row and column indices of 4 neighbourhood pixels (2 pairs) of each non-edge non-zero gradient magnitude pixel in direction of corresponding gradient angle
    relative_rows = row_map[:, binned_indices]
    relative_cols = col_map[:, binned_indices]
    actual_rows = rows[:, None] + relative_rows
    actual_cols = cols[:, None] + relative_cols

    pair1_rows, pair2_rows = actual_rows
    pair1_cols, pair2_cols = actual_cols

    magnitudes = gradient_magnitudes[rows, cols]

    # the pair indices are iterated over instead of the pair values because the interpolation and NMS has to be done recursively
    for r, c, mag, rows1, rows2, cols1, cols2, x, x_ in zip(rows, cols, magnitudes, pair1_rows, pair2_rows, pair1_cols, pair2_cols, interpolant_positions, interpolant_positions_):

        # index into updating gradient magnitudes array and get the interpolant pairs
        interpolant1, interpolant1_ = gradient_magnitudes[rows1, cols1]
        interpolant2, interpolant2_ = gradient_magnitudes[rows2, cols2]

        # this saves time and ensures that if interpolant pair values are equal, the proper value is passed
        if interpolant1 != interpolant1_:
            interpolant1 = interpolant1 * x + interpolant1_ * x_

        if interpolant2 != interpolant2_:
            interpolant2 = interpolant2 * x + interpolant2_ * x_

        # the NMS comparison (if pixel magnitude is not greater than both neighbours, set to zero)
        if (mag <= interpolant1 or mag <= interpolant2):
            gradient_magnitudes[r, c] = 0


    # neighbour_mags = gradient_intensities[actual_rows, actual_cols]
    # interpolant_positions = np.array([interpolant_positions, 1 - interpolant_positions]).T
    # interpolants = np.sum(neighbour_mags * interpolant_positions, axis=2)
    # mags = np.array([magnitudes, interpolants[0], interpolants[1]]).T
    # for r, c, mag in zip(rows, cols, mags):
    #     print(mag)
    #     if (mag[0] <= mag[1] or mag[0] <= mag[2]):
    #         gradient_intensities[r, c] = 0

    return gradient_magnitudes


