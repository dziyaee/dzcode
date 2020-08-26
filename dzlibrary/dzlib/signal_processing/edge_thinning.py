import numpy as np


def non_maximal_suppression(gradient_magnitudes, gradient_angles):
    gradient_magnitudes = gradient_magnitudes.copy()

    row_map = np.array([0, -1, -1, -1, 0])
    col_map = np.array([1, 1, 0, -1, 1])
    row_map = np.array([row_map, -row_map])
    col_map = np.array([col_map, -col_map])

    rows, cols = np.where(gradient_magnitudes[1:-1, 1:-1] != 0)
    rows += 1
    cols += 1

    angles = gradient_angles[rows, cols]
    angles[angles < 0] += np.pi
    bins = np.arange(0, np.pi+np.pi/4, np.pi/4)
    bin_indices = np.digitize(angles, bins[1:], right=False)

    relative_rows = row_map[:, bin_indices]
    relative_cols = col_map[:, bin_indices]

    actual_rows = (rows + relative_rows).T
    actual_cols = (cols + relative_cols).T

    magnitudes = gradient_magnitudes[rows, cols]

    for r, c, mag, rows_, cols_ in zip(rows, cols, magnitudes, actual_rows, actual_cols):
        neighbour1 = gradient_magnitudes[rows_[0], cols_[0]]
        neighbour2 = gradient_magnitudes[rows_[1], cols_[1]]
        if (mag <= neighbour1 or mag < neighbour2):
            gradient_magnitudes[r, c] = 0

    return gradient_magnitudes


def non_maximal_suppression_with_interpolation(gradient_magnitudes, gradient_angles):
    gradient_magnitudes = gradient_magnitudes.copy()

    # relative indices of 4 closest neighbour pixels to center pixel of a 3x3 window. maps to angle bin indices
    row_map = np.array([(-1, 0), (-1, -1), (-1, -1), (-1, 0), (1, 0)])
    col_map = np.array([(1, 1), (1, 0), (-1, 0), (-1, -1), (-1, -1)])
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
    bin_indices = np.digitize(angles, bins[1:], right=False)

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
    relative_rows = row_map[:, bin_indices]
    relative_cols = col_map[:, bin_indices]
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

    return gradient_magnitudes


