import numpy as np


# functions are separated to allow for increased time efficiency and flexibility. If input, kernel, and stride sizes are always the same, then the im2col_indices need only be computed a single time. Once these indices are computed, im2col can be called on any number of signal batches.

def im2col_indices(vector_size, kernel, stride):
    '''Returns the im2col indices for a 1d vector and 1d kernel using a vectorized approach via numpy array broadcasting
    My implementation is inspired by this lovely post: https://stackoverflow.com/a/30110497/3826634

    Args:
        vector_size (int): N (number of elements in the 1d vector). must be >= K
        kernel (int): K (number of elements in the 1d window). must be >= 0 and <= N
        stride (int): must be >= 1

    Returns:
        im2col_indices (2d numpy array): (K x S) (kernel_size x n_snippets)
    '''

    # first kernel vector
    first_kernel = np.arange(kernel)[None, :]  # (1 x K)

    # number of snippets (S)
    n_snippets = int(((vector_size - kernel) // stride) + 1)

    # index offset vector
    offset = np.arange(n_snippets)[None, :] * stride  # (1 x S)

    # constructs im2col indices via broadcasting array addition
    return first_kernel.T + offset  # (K x S)


def im2col(vectors, indices):
    '''Returns a 3d im2col array using a 2d im2col indices matrix: np.take is called on each 1d vector in the 2d vectors matrix, added to a list, and that list is converted into a 3d numpy array.

    Args:
        vectors (2d numpy array): (C x N) (channels x vector_size)
        indices (2d numpy array): (K x S) (kernel_size x n_snippets)

    Returns:
        (3d numpy array): (C x K x S) (channels x kernel_size x n_snippets)

    '''
    return np.array([np.take(vector, indices) for vector in vectors])


if __name__ == "__main__":
    # Example to convert a multi-channel vector matrix into an im2cols array and perform a median filter on the result

    # input
    input_shape = (16, 1024)
    vectors = np.random.randn(*input_shape)  # (C x N)

    # sizes
    vector_size = vectors.shape[-1]
    stride = 1

    # windowing technique commonly used in Radar CFAR detection: https://www.mathworks.com/help/phased/examples/constant-false-alarm-rate-cfar-detection.html
    # kernel size: |REF|REF|GUARD|CUT|GUARD|REF|REF| --> |REF|REF|REF|REF|
    guard = 1  # Guard cells (discarded)
    ref = 2  # Reference cells (R) (kept)
    CUT = 1  # Cell Under Test (CUT) (discarded)
    kernel = 2 * (guard + ref) + CUT

    # im2col indices
    indices = im2col_indices(vector_size, kernel, stride)  # (K x S)

    # discard guard + cut indices
    slice1 = np.arange(0, ref)  # keep (0: ref)
    slice2 = np.arange(kernel - ref, kernel)  # keep (kernel - ref: kernel)
    slices = (*slice1, *slice2)
    indices = indices[slices, :]  # (2R x S)

    # im2col
    im2cols = im2col(vectors, indices)  # (2R x S)

    # median
    medians = np.median(im2cols, axis=1)  # (C x S)
