import numpy as np


def gaussian2d(size, sigma):
    '''Returns an estimation of a square gaussian kernel with specified size and standard deviation'''

    # kernel indices with top-left origin
    i = np.arange(1, size + 1, 1)
    j = np.arange(1, size + 1, 1)
    i, j = np.meshgrid(i, j)

    # kernel indices with center origin
    k = (size - 1) / 2
    x = i - (k + 1)
    y = j - (k + 1)

    # Formula from: https://en.wikipedia.org/wiki/Gaussian_blur#Mathematics
    norm = 1 / (2 * np.pi * (sigma ** 2))
    num = (x ** 2) + (y ** 2)
    den = (2 * (sigma ** 2))
    exponent = -(num / den)
    return norm * np.exp(exponent)


def sobel_operator():
    '''Returns sobel operator kernels in the shape (2 x 1 x 3 x 3)'''
    gx = np.array([
                  [1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]
                  ])

    gy = np.array([
                  [1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]
                  ])

    return np.array([[gx], [gy]])
