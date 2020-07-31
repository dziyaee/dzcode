import numpy as np
from dzlib.common.utils import timer


@timer
def im2col4d(xdims, kdims, sdims):
    # standardize input shape sizes to 3 to represent channel, height, width dims
    dims = [xdims, kdims, sdims]
    size = 4
    new = []
    for dim in dims:
        dim = np.asarray(dim)
        diff = size - dim.size
        value = [1] * diff
        dim = np.insert(dim, 0, value)
        new.append(dim)

    xdim, kdim, sdim = new
    sdim = [1 if x == 0 else x for x in sdim]

    # channel, height, width dimensions of input, window, stride
    xn, xc, xh, xw = xdim
    _, kc, kh, kw = kdim
    _, _, sh, sw = sdim
    assert xc == kc

    # first window index vector
    deps = np.array(np.arange(kc), ndmin=2) * xh * xw
    rows = np.array(np.arange(kh), ndmin=2) * xw
    cols = np.array(np.arange(kw), ndmin=2)
    window = np.array((deps.T + rows).ravel(), ndmin=2)
    window = np.array((window.T + cols).ravel(), ndmin=2)

    # number of windows along rows and cols
    nh = int(np.floor((xh - kh) / sh) + 1)
    nw = int(np.floor((xw - kw) / sw) + 1)

    # index offset vector
    nums = np.array(np.arange(xn), ndmin=2) * xc * xh * xw
    rows = np.array(np.arange(nh), ndmin=2) * sh * xw
    cols = np.array(np.arange(nw), ndmin=2) * sw
    offset = np.array((nums.T + rows).ravel(), ndmin=2)
    offset = np.array((offset.T + cols).ravel(), ndmin=2)

    # add offset to window via broadcasting to create final indices
    time1 = time.time()
    indices = (window.T + offset)
    print(time.time() - time1)
    time2 = time.time()
    indices = np.array(np.split(indices, xn, axis=1))
    print(time.time() - time2)
    return indices


lines = '-' * 100
# x = (5, 5, 500, 500)
# k = (1, 5, 10, 10)
# x = (5, 5, 700, 700)
# k = (1, 5, 7, 7)
x = (100, 3, 500, 500)
k = (1, 3, 3, 1)


s = (1, 1)
z = np.zeros((x))
images = np.arange(z.size).reshape(z.shape)
# print(f"{images.shape}\n{images}\n{lines}")


indices1 = im2col(x, k, s)
indices2 = indices1
indices3 = im2col_(x, k, s)
# xn, xc, xh, xw = x
# W = ((indices3 / (1 * 1  * 1  * 1 )) % xw).astype(np.int32)
# H = ((indices3 / (1 * 1  * 1  * xw)) % xh).astype(np.int32)
# C = ((indices3 / (1 * 1  * xh * xw)) % xc).astype(np.int32)
# N = ((indices3 / (1 * xc * xh * xw)) % xn).astype(np.int32)
# indices4 = (N, C, H, W)


@timer
def take1(indices, images):
    im2cols = np.array([np.take(image, indices) for image in images])
    return im2cols

@timer
def take2(indices, images):
    im2cols = []
    for image in images:
        im2cols.append(np.take(image, indices))
    return np.array(im2cols)

@timer
def take3(indices, images):
    im2cols = np.take(images, indices)
    return im2cols

@timer
def take4(indices, images):
    N, C, H, W = indices
    im2cols = images[N, C, H, W]
    return im2cols


out1 = take1(indices1, images)
out2 = take2(indices2, images)
out3 = take3(indices3, images)
# out4 = take4(indices4, images)

print(out1.shape)
print(out2.shape)
print(out3.shape)
# print(out4.shape)

check12 = np.array_equal(out1, out2)
check23 = np.array_equal(out2, out3)
# check34 = np.array_equal(out3, out4)
print(f"check12: {check12}")
print(f"check23: {check23}")
# print(f"check34: {check34}")
