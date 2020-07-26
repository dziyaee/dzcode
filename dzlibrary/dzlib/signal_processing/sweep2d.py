import numpy as np
from dzlib.signal_processing.utils import im2col
from dzlib.signal_processing.signals import Dims


class Sweep2d():
    def __init__(self, image_shape, kernel_shape, pad=0, stride=1, mode='user'):
        # convert all input shapes to tuples
        shapes = [image_shape, kernel_shape, pad, stride]
        shapes = [(int(shape),) if isinstance(shape, (int, float)) else tuple(shape) for shape in shapes]
        image_shape, kernel_shape, pad, stride = shapes

        # expand image, kernel shapes to 4d and pad, stride shapes to 2d
        expand = lambda x, ndim, fill_value: (*((fill_value,) * (ndim - len(x))), *x)[-ndim:]
        image_shape = expand(image_shape, 4, 1)
        kernel_shape = expand(kernel_shape, 4, 1)
        pad = expand(pad, 2, pad[0])
        stride = expand(stride, 2, stride[0])

        # create Dims objects out of shapes for standardized accessing of shape dimensions (num, depth, height, width)
        xx = Dims(image_shape)
        kk = Dims(kernel_shape)

        # return padding and stride based on selected mode and create Dims objects (height, width)
        pad, stride = self._mode(xx, kk, pad, stride, mode)
        pp = Dims(pad)
        ss = Dims(stride)

        # calc padded image shape and init array of zeros
        padded_height = int(xx.height + 2 * pp.height)
        padded_width = int(xx.width + 2 * pp.width)
        padded_shape = (xx.num, xx.depth, padded_height, padded_width)
        xx = Dims(padded_shape)
        padding = np.zeros((xx.shape)).astype(np.float32)

        # ensure kernel and padded image shapes are valid
        if kk.depth != xx.depth:
            raise ValueError(f"Kernel depth ({kk.depth}) must be = image depth ({xx.depth})")
        if (kk.height > xx.height) or (kk.width > xx.width):
            raise ValueError(f"Kernel height, width ({kk.height, kk.width}) must be <= image height, width ({xx.height, xx.width})")

        # calc output shape
        output_height = int(np.floor((xx.height - kk.height) / ss.height) + 1)
        output_width = int(np.floor((xx.width - kk.width) / ss.width) + 1)
        output_shape = (xx.num, kk.num, output_height, output_width)
        yy = Dims(output_shape)

        # im2col indices
        indices = im2col(xx.shape, kk.shape, ss.shape)
        im2col_height, im2col_width = indices.shape
        assert im2col_height == kk.depth * kk.height * kk.width  # these two asserts should always be true
        assert im2col_width == yy.height * yy.width

        # assign shape objects
        self.xx = xx
        self.kk = kk
        self.pp = pp
        self.yy = yy

        # assign arrays
        self.padding = padding
        self.indices = indices

    def _mode(self, image_Dims, kernel_Dims, pad, stride, mode):
        modes = ['user', 'full', 'keep']
        xx = image_Dims
        kk = kernel_Dims
        pad_height, pad_width = pad
        stride_height, stride_width = stride

        if mode not in modes:
            raise ValueError(f"mode ({mode}) must be one of: {modes}")

        else:

            # user mode: no change to default or input pad and stride values
            if mode == 'user':
                pass

            # full mode: stride = 1, pad input by kernel size - 1
            elif mode == 'full':
                pad_height = kk.height - 1
                pad_width = kk.width - 1
                pad = (pad_height, pad_width)

                stride_height = 1
                stride_width = 1
                stride = (stride_height, stride_width)

            # keep mode: calculate minimum padding necessary using user or default stride values with image and kernel dimensions
            # note: the minimum padding may not be an even number, thus resulting in extra padding on the 'right' side
            elif mode == 'keep':
                min_pad = lambda x, k, s: (k - s + x * (s - 1)) / 2
                pad_height = min_pad(xx.height, kk.height, stride_height)
                pad_width = min_pad(xx.width, kk.width, stride_width)
                pad = (pad_height, pad_width)

        return pad, stride

    def correlate2d(self, images, kernels):
        # standardize inputs to 4d
        images = np.array(images, ndmin=4)
        kernels = np.array(kernels, ndmin=4)

        # pad images
        images = self._pad2d(images)

        # create im2col matrices
        im2cols = self._im2col(images)

        # create kr2row matrix
        kr2row = self._kr2row(kernels, 'correlate2d')

        # corr via dot product
        yy = self.yy
        outputs = np.matmul(kr2row, im2cols).reshape(yy.shape)
        return outputs

    def convolve2d(self, images, kernels):
        # standardize inputs to 4d
        images = np.array(images, ndmin=4)
        kernels = np.array(kernels, ndmin=4)

        # pad images
        images = self._pad2d(images)

        # create im2col matrices
        im2cols = self._im2col(images)

        # create kr2row matrix
        kr2row = self._kr2row(kernels, 'convolve2d')

        # corr via dot product
        yy = self.yy
        outputs = np.matmul(kr2row, im2cols).reshape(yy.shape)
        return outputs

    def median2d(self, images):
        # standardize inputs to 4d
        images = np.array(images, ndmin=4)

        # pad images
        images = self._pad2d(images)

        # create im2col matrices
        im2cols = self._im2col(images)

        # update output shape depth to 1 (because number of kernels = 1 for median2d operation)
        xx = self.xx
        yy = self.yy
        output_shape = (xx.num, 1, yy.height, yy.width)
        yy = Dims(output_shape)

        # Median Filter calculates the median of the 3d im2col matrix along axis 1 (down through each column / across rows). Reshape to proper output dimensions
        outputs = np.median(im2cols, axis=1).reshape(yy.shape)
        return outputs

    def mean2d(self, images):
        # standardize inputs to 4d
        images = np.array(images, ndmin=4)

        # pad images
        images = self._pad2d(images)

        # create im2col matrices
        im2cols = self._im2col(images)

        # update output shape depth to 1 (because number of kernels = 1 for mean2d operation)
        xx = self.xx
        yy = self.yy
        output_shape = (xx.num, 1, yy.height, yy.width)
        yy = Dims(output_shape)

        # Median Filter calculates the median of the 3d im2col matrix along axis 1 (down through each column / across rows). Reshape to proper output dimensions
        outputs = np.mean(im2cols, axis=1).reshape(yy.shape)
        return outputs

    def _pad2d(self, images):
        xx = self.xx
        pp = self.pp
        padding = self.padding

        # starting indices, left side bias
        h1 = int(np.floor(pp.height))
        w1 = int(np.floor(pp.width))

        # ending indices
        h2 = int(np.floor(xx.height - pp.height))
        w2 = int(np.floor(xx.width - pp.width))
        padding[:, :, h1: h2, w1: w2] = images
        return padding

    def _im2col(self, images):
        indices = self.indices
        im2cols = np.array([np.take(image, indices) for image in images])
        return im2cols

    def _kr2row(self, kernels, operation):
        kk = self.kk
        if operation == "convolve2d":
            kr2row = np.flip(kernels, axis=(2, 3)).reshape(kk.num, -1)

        elif operation == "correlate2d":
            kr2row = kernels.reshape(kk.num, -1)
        return kr2row
