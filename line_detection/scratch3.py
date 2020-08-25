import numpy as np
import math
from dzlib.common.utils import timer
from dzlib.signal_processing.utils import im2col
import pdb


# inputs ------------------------------------------------------------------------
# (supposed outputs from performing edge detection via Sobel operator)
# gradient_intensities: (H x W) (magnitudes >= 0)
# gradient_directions: (H x W) (-pi <= angles <= pi)
np.random.seed(43)
# shape = (1000, 800)
shape = (5, 4)
gradient_intensities = np.random.randint(0, 10, shape)
# gradient_intensities = np.random.randn(*shape)
gradient_directions = np.random.uniform(-np.pi, np.pi, shape)

# angle bins
# bins = np.arange(0, 180+45, 45)
bins = np.arange(0, np.pi+np.pi/4, np.pi/4)

# direction to neighbourhood pixel mapping
pixel_map = {0: ([2, 5], [6, 3]), 45: ([2, 1], [6, 7]), 90: ([0, 1], [8, 7]), 135: ([0, 3], [8, 5]), 180: ([2, 5], [6, 3])}

def linear_interpolation(pixel_pair_values, pixel_pair_positions, interpolant_position):
    '''Returns interpolant result which is the linear interpolation at a given interpolant position between two points of specified values and at specified positions'''
    v1, v2 = pixel_pair_values
    x1, x2 = pixel_pair_positions
    x = interpolant_position
    return (v1 * (x - x2) + v2 * (x1 - x)) / (x1 - x2)

# -------------------------------------------------------------------------------

@timer
def func1(gradient_intensities, gradient_directions, bins, pixel_map):
    '''Function 1 attempts to perform all or most operations on a per-pixel basis'''

    # im2col indices for 3x3 NMS
    im2col_indices = im2col(gradient_intensities.shape, (3, 3), 1).T

    gradient_directions[gradient_directions < 0] += np.pi  # shift angles in [-pi, 0) interval to angles in [0, pi] interval. Gradient directions separated by pi radians represent the same gradient lines, pointing in opposite directions. The 'pointing' information is not needed and thus can be removed by adding pi radians to all negative angles.
    gradient_directions *= (180 / np.pi)  # convert from radians to degrees

    # bin angles to represent 4 directions (E-W, NE-SW, N-S, NW-SE) in order to map certain directions with indices for neighbourhood pixels
    indices = np.digitize(gradient_directions, bins[1:], right=False)
    gradient_directions_binned = bins[indices]

    for window in im2col_indices:
        center_pixel = window[4]
        direction_binned = np.take(gradient_directions_binned, center_pixel)

        direction = np.take(gradient_directions, center_pixel)
        theta = abs((direction % 90 // 45) * 45 - (direction % 45))
        x = math.tan(theta * (math.pi / 180))

        pixel_pair1, pixel_pair2 = pixel_map[direction_binned]
        indices1 = window[pixel_pair1]
        indices2 = window[pixel_pair2]
        values1 = np.take(gradient_intensities, indices1)
        values2 = np.take(gradient_intensities, indices2)
        interpolant1 = linear_interpolation(values1, (1, 0), x)
        interpolant2 = linear_interpolation(values2, (1, 0), x)
        center_value = np.take(gradient_intensities, center_pixel)

        if (center_value <= interpolant1 or center_value <= interpolant2):
            gradient_intensities.ravel()[center_pixel] = 0
    return gradient_intensities


@timer
def func2(gradient_intensities, gradient_directions, bins, pixel_map):
    '''Function 2 similar to Function1 but with ravelling and directing indexing of arrays instead of np.take'''
    shape = gradient_intensities.shape

    # im2col indices for 3x3 NMS
    im2col_indices = im2col(gradient_intensities.shape, (3, 3), 1).T

    gradient_directions[gradient_directions < 0] += np.pi  # shift angles in [-pi, 0) interval to angles in [0, pi] interval. Gradient directions separated by pi radians represent the same gradient lines, pointing in opposite directions. The 'pointing' information is not needed and thus can be removed by adding pi radians to all negative angles.
    gradient_directions *= (180 / np.pi)  # convert from radians to degrees

    # bin angles to represent 4 directions (E-W, NE-SW, N-S, NW-SE) in order to map certain directions with indices for neighbourhood pixels
    indices = np.digitize(gradient_directions, bins[1:], right=False)
    gradient_directions_binned = bins[indices]

    gradient_directions = gradient_directions.ravel()
    gradient_directions_binned = gradient_directions_binned.ravel()
    gradient_intensities = gradient_intensities.ravel()

    for window in im2col_indices:
        center_pixel = window[4]
        direction_binned = gradient_directions_binned[center_pixel]

        direction = gradient_directions[center_pixel]
        theta = abs((direction % 90 // 45) * 45 - (direction % 45))
        x = math.tan(theta * (math.pi / 180))

        pixel_pair1, pixel_pair2 = pixel_map[direction_binned]
        indices1 = window[pixel_pair1]
        indices2 = window[pixel_pair2]
        values1 = gradient_intensities[indices1]
        values2 = gradient_intensities[indices2]
        interpolant1 = linear_interpolation(values1, (1, 0), x)
        interpolant2 = linear_interpolation(values2, (1, 0), x)
        center_value = gradient_intensities[center_pixel]

        if (center_value <= interpolant1 or center_value <= interpolant2):
            gradient_intensities[center_pixel] = 0
    return gradient_intensities.reshape(shape)


@timer
def func3(gradient_intensities, gradient_directions, bins, pixel_map):
    '''Function 3 attempts to vectorize as much of the NMS as possible (except for the actual value comparisons as that must be performed recursively)'''
    # pdb.set_trace()


    # index_map = [[(1, 1), (0, 1)], [(1, 1), (1, 0)], [(1, -1), (1, 0)], [(1, -1), (0, -1)], [(-1, -1), (0, -1)]]
    index_map = [[-2, 1], [-2, -3], [-4, -3], [-4, -1], [2, -1]]
    # index_map = [([2, 5], [6, 3]), ([2, 1], [6, 7]), ([0, 1], [8, 7]), ([0, 3], [8, 5]), ([2, 5], [6, 3])]
    index_map = np.array(index_map)

    im2col_indices = im2col(gradient_intensities.shape, (3, 3), 1)
    center_values = np.take(gradient_intensities, im2col_indices[4])
    center_angles = np.take(gradient_directions, im2col_indices[4])
    # rows, cols = np.where(gradient_intensities != 0)

    center_angles[center_angles < 0] += np.pi
    bin_indices = np.digitize(center_angles, bins[1:], right=False)
    center_angles_binned = bins[bin_indices]

    thetas = (center_angles % np.pi/2 // np.pi/4) * np.pi/4 - (center_angles % np.pi/4)
    interpolant_positions = np.tan(thetas)

    relative_indices = index_map[bin_indices]
    # print(im2col_indices)
    # print()
    # print(relative_indices)
    # print()
    rows_ = 4 + relative_indices
    cols_ = np.array([np.arange(im2col_indices.shape[1]), np.arange(im2col_indices.shape[1])]).T
    # print(rows_)
    # print(cols_)
    # print()
    actual_indices = im2col_indices[rows_, cols_]
    actual_values = np.take(gradient_intensities, actual_indices)
    # print(actual_indices)
    # print(gradient_intensities)
    # print(actual_values)
    return None


@timer
def func4(gradient_intensities, gradient_directions, bins, pixel_map):
    # pdb.set_trace()

    row_map = np.array([(-1, 0), (-1, -1), (-1, -1), (-1, 0), (1, 0)])
    col_map = np.array([(1, 1), (1, 0), (-1, 0), (-1, -1), (-1, -1)])

    row_map = np.array([row_map, -row_map])
    col_map = np.array([col_map, -col_map])

    # row_map = np.array([(-1, 0, 1, 0), (-1, -1, 1, 1), (-1, -1, 1, 1), (-1, 0, 1, 0), (1, 0, -1, 0)])
    # col_map = np.array([(1, 1, -1, -1), (1, 0, -1, 0), (-1, 0, 1, 0), (-1, -1, 1, 1), (-1, -1, 1, 1)])

    rows, cols = np.where(gradient_intensities[1:-1, 1:-1] != 0)
    rows += 1
    cols += 1
    magnitudes = gradient_intensities[rows, cols]

    angles = gradient_directions[rows, cols]
    angles[angles < 0] += np.pi
    bin_indices = np.digitize(angles, bins[1:], right=False)
    binned_angles = bins[bin_indices]

    # angle modulu 90 splits the space into two 90 degree halves (quadrants 1 and 2)
    # floor div 45 gives which half of a quadrant the angle is in (0 or 1)
    # multiply by 45 gives which base line the angle is being measured from (0 or 45)
    # angle modulo 45 splits the space into 4 quarters (quarters 1, 2, 3, 4)
    # subtract that angle from the base line angle to obtain the angle theta
    thetas = (angles % np.pi/2 // np.pi/4) * np.pi/4 - (angles % np.pi/4)
    interpolant_positions = np.tan(thetas)  # tan(theta) = interpolant_position / adjacent, where adjacent is always 1

    # relative_rows = row_map[bin_indices]
    # relative_cols = col_map[bin_indices]
    relative_rows = row_map[:, bin_indices]
    relative_cols = col_map[:, bin_indices]

    actual_rows = rows[:, None] + relative_rows
    actual_cols = cols[:, None] + relative_cols

    actual_mags = gradient_intensities[actual_rows, actual_cols]

    print(gradient_intensities)
    print(rows)
    print(cols)
    print(magnitudes)
    print()
    print(np.round(gradient_directions * (180 / np.pi), 1))
    print(np.round(angles * (180 / np.pi), 1))
    print(bin_indices)
    print(binned_angles * (180 / np.pi))
    print()
    print(rows)
    print(relative_rows)
    print(f"actual rows:\n{actual_rows}")
    print()
    print(cols)
    print(relative_cols)
    print(f"actual cols:\n{actual_cols}")
    print()
    print(f"actual_mags:\n{actual_mags}")
    return None



N = 1
time = func4.timer(N, gradient_intensities.copy(), gradient_directions.copy(), bins, pixel_map)
print(f"time: {(time / N * 1e3):.4f} ms")

# g1 = func1(gradient_intensities.copy(), gradient_directions.copy(), bins, pixel_map)
# g2 = func2(gradient_intensities.copy(), gradient_directions.copy(), bins, pixel_map)
# print(np.array_equal(g1, g2))

# N = 1
# time1 = func1.timer(N, gradient_intensities.copy(), gradient_directions.copy(), bins, pixel_map)
# print("\nfunc1")
# print(f"{(time1 * 1e3):10.4f} ms")
# print(f"{((time1 / N) * 1e3):10.4f} ms")

# time2 = func2.timer(N, gradient_intensities.copy(), gradient_directions.copy(), bins, pixel_map)
# print("\nfunc2")
# print(f"{(time2 * 1e3):10.4f} ms")
# print(f"{((time2 / N) * 1e3):10.4f} ms")


