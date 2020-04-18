# General Imports
import numpy as np
import time


# Helper Stats function
def stats(x, var_name="Variable", array=True, minmax=True, mean=True, median=True, std=True):

    print(f"\n{var_name} stats:")

    if array:
        print(x)

    if minmax:
        print(f"Min:    {np.min(x):.4f}")
        print(f"Max:    {np.max(x):.4f}")

    if mean:
        print(f"Mean:   {np.mean(x):.4f}")

    if median:
        print(f"Median: {np.median(x):.4f}")

    if std:
        print(f"Std:    {np.std(x):.4f}")

    return None


# Helper Show function
def show(x, var_name="Variable", array=True, size=True, shape=True, ndims=True):

    print(f"\n{var_name} info")

    if array:
        print(x)

    if size:
        print(f"{var_name} Size:  {x.size}")

    if shape:
        print(f"{var_name} Shape: {x.shape}")

    if ndims:
        print(f"{var_name} Ndims: {x.ndim}")

    return None


# 1D Conv on 1D Signals via Toeplitz Matrix
def conv1d_1d(x, k, prints=False):

    # Input Width & Kernel Width
    x_width = x.size
    k_width = k.size

    # Output Width
    y_width = x_width + k_width - 1

    # Kernel Toeplitz Matrix Width = Input Width
    tk_width = x_width

    # Kernel Toeplitz Matrix Height = Output Width
    tk_height = y_width

    # Zero Padding Kernel k
    overlap = 1
    pad = x_width - overlap

    if pad > 0:
        k_padded = np.zeros((pad + k_width + pad))
        k_padded[pad: -pad] = k

    elif pad == 0:
        k_padded = k

    # Init Kernel Toeplitz Matrix with zeros
    if (x.dtype == "int64") and (k.dtype == "int64"):
        k_toeplitz = np.zeros((tk_height, tk_width)).astype(np.int32)

    else:
        k_toeplitz = np.zeros((tk_height, tk_width)).astype(np.float64)

    # Padded k width
    k_padded_width = k_padded.size

    # Fill Kernel Toeplitz Matrix with padded k slices. Row index is used as starting index for slice each iteration. np.flip() is used to create a Toeplitz Matrix. If np.flip() is not used, the result is known as a Hankel Matrix
    for row in range(tk_height):

        k_toeplitz[row] = np.flip(k_padded[row: row + tk_width])

    # The convolution of x and k is equivalent to the matrix multiplication of k's toeplitz matrix and x
    y = np.matmul(k_toeplitz, x)

    # Prints
    if prints:
        show(x, "Input")
        show(k, "Kernel")
        show(k_toeplitz, "Kernel Toeplitz Matrix")

    return y


# 2d Conv on 2d Signals via Block Toeplitz Matrix
def conv2d_2d(x, k, shows=False):
    '''Function to convolve two 2d signals, x and k along 2 axes (2d conv, 2d signals)

    Args:
        x (Numpy Array): Shape = (Hx, Wx), Input 1 = Image
        k (Numpy Array): Shape = (Hk, Wk), Input 2 = Kernel

    Returns:
        y (Numpy Array): Shape = (Hx + Hk - 1, Wx + Wk - 1), Output
    '''

    # Image Dimensions
    x_height, x_width = x.shape

    # Kernel Dimensions
    k_height, k_width = k.shape

    # Output Dimensions
    y_height = x_height + k_height - 1
    y_width = x_width + k_width - 1

    # Kernel Row Toeplitz Dimensions
    # Each Kernel Row has a corresponding Toeplitz Matrix
    # This Toeplitz Matrix shape = (output width x input width)
    t_height = y_width
    t_width = x_width

    # A Toeplitz Tensor is created to store one Kernel Row Toeplitz Matrix per each depth dimension
    k_toeplitz = np.zeros((k_height, t_height, t_width))

    # Iterate through depth of k_toeplitz tensor which is equivalent for rows in kernel
    for row in range(k_height):

        # Iterate through columns of each toeplitz matrix within a given depth of the k_toeplitz tensor
        for col in range(t_width):

            # Index the Tensor by (depth = row, col = col, row = col + col + kernel width), and store kernel given by row index
            k_toeplitz[row, col: col + k_width, col] = k[row]

    # Reshape Toeplitz Tensor to a combined Toeplitz Matrix, flattened along axes 0 and 1
    # This Toeplitz Matrix Block will be plugged into the final Convolution Matrix along the main diagonal
    k_toeplitz = k_toeplitz.reshape(k_height * t_height, t_width)

    # Final Convolution Matrix A Dimensions
    A_height = y_height * y_width
    A_width = x_height * x_width

    # Create final Convolution Matrix A
    A = np.zeros((A_height, A_width))

    # Row, Column indices and stride lengths
    row = 0
    col = 0
    row_stride = k_height * y_width
    col_stride = x_width

    # Iterate through Convolution Matrix A x_height times
    for block in range(x_height):

        # Plug in Kernel Toeplitz Block into the main diagonal of A
        A[row: row + row_stride, col: col + col_stride] = k_toeplitz

        # Iterate the row and column indices
        row += y_width
        col += x_width

    # Final Output = Matrix Multiplication of Conv Matrix A and flattened input x
    # Reshape into expected output shape
    y = np.matmul(A, x.flatten()).reshape(y_height, y_width)

    if shows:

        show(x, "Input x")
        show(k, "Kernel k")
        show(A, "Convolution Matrix A")

    return y


# 2D Conv on 3d signals via im2col method (implementation idea of im2col derived from cs231n course @ http://cs231n.github.io/convolutional-networks/#overview)
def conv2d_3d(x, k, k_stride=1, pad=(0, 0), shows=False):
    ''' Function that performs a 2d convolution of two 3d signals.
    The function expects x to be a 3d tensor containing the 3d "image" signal.
    The function expects k to be a 4d tensor containing all of the 3d "kernel" signals.

    Args:
        x (Numpy Array 3d Tensor): Shape (D x H x W). A Tensor containing one 3d Image
        k (Numpy Array 3d Tensor): Shape (N x D x H x W). A Tensor containing 3d Kernel Tensors along the N axis
        k_stride (int): The spacing between convolution intervals / patches
        pad (tuple of ints): The amount of zero padding to add to the image x along height and width, respectively
        shows (bool): debugging helper

    Returns:
        y (Numpy Array 3d Tensor): Shape (D x H x W). A Tensor containing the convolution outputs. Each slice along the D axis of y corresponds to the convolution output of image x and a single kernel slice along the N axis of k

    '''

    # Input / Image (x) dimensions (D x H x W)
    x_depth, x_height, x_width = x.shape

    # Kernel / Filter (k) dimensions (N x D x H x W)
    k_num, k_depth, k_height, k_width = k.shape

    # Input / Image Zero Padding along height and width
    x_pad_height = pad[0]
    x_pad_width = pad[1]

    # Output (y) dimensions (D x H x W), where each slice D in y is the result of the convolution between the image x and the kernel within the corresponding slice k_num in k
    y_depth = k_num
    y_height = int(((x_height - k_height + 2 * x_pad_height) / k_stride) + 1)
    y_width = int(((x_width - k_width + 2 * x_pad_width) / k_stride) + 1)

    # Conv Matrix (A) (H x W)
    A_height = k_height * k_width * k_depth
    A_width = y_height * y_width
    A = np.zeros((A_height, A_width))

    # Zero Padded Image (padded_x) (D x H x W)
    padded_x_depth = x_depth
    padded_x_height = x_pad_height + x_height + x_pad_height
    padded_x_width = x_pad_width + x_width + x_pad_width
    padded_x = np.zeros((padded_x_depth, padded_x_height, padded_x_width))
    padded_x[:, x_pad_height: padded_x_height - x_pad_height, x_pad_width: padded_x_width - x_pad_width] = x[:, :, :]

    # Iterate through Conv Matrix A Columns
    for A_col in range(A_width):

        # Calculate Row Index as Floor Division of current column in A by output width y_width * k_stride
        r = int(A_col // y_width) * k_stride

        # Calculate Column Index as Modulus of current column in A by output width y_width * k_stride
        c = int(A_col % y_width) * k_stride

        # Create a flattened patch of the padded image for each row and column index, and insert into the corresponding column in A
        A[:, A_col] = padded_x[:, r: r + k_height, c: c + k_width].flatten()

    # Flatten the 4d Kernel Tensor into a 2d Matrix (N x D x H x W) > (N x (D*H*W)), and matrix multiply with A
    # In this way, each row of the Kernel Matrix corresponds to a distinct Kernel, and each column of the Conv Matrix A corresponds to a distinct image patch that each Kernel overlaps with
    # k = k.reshape(k_num, -1)

    # show(k, "Original Kernel")
    k = np.flip(k, axis=(2, 3))
    # show(k, "Flipped Kernel")
    k = k.reshape(k.shape[0], -1)
    # show(k, "Flipped & Reshaped Kernel")
    # show(A[:, 0], "Conv Matrix A Slice")

    y = np.matmul(k, A).reshape(y_depth, y_height, y_width)

    if shows:

        show(padded_x, "Padded Input")
        show(A, "Conv Matrix A")

    return y

