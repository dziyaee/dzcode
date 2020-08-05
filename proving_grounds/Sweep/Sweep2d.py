import numpy as np
from ShapeNd import ShapeNd, Dimension
from SweepNd import SweepNd


class Input4d(ShapeNd):
    def __init__(self, shape):
        super(Input4d, self).__init__(shape)

    width = Dimension(-1)
    height = Dimension(-2)
    depth = Dimension(-3)
    num = Dimension(-4)


class Param2d(ShapeNd):
    def __init__(self, shape):
        super(Param2d, self).__init__(shape)

    width = Dimension(-1)
    height = Dimension(-2)


class Sweep2d(SweepNd):
    operations = ("correlate", "convolve")
    modes = ("user", "full", "keep")
    dtype = np.float32

    def __init__(self, unpadded, window, padding, stride, mode="user"):
        INPUT_NDIM = 4  # number of elements in unpadded & window shape tuples
        PARAM_NDIM = 2  # number of elements in padding & stride shape tuples
        INPUT_SHAPE = Input4d  # ShapeNd object defining unpadded & window dimension attributes
        PARAM_SHAPE = Param2d  # ShapeNd object defining padding & stride dimension attributes

        super().__init__(unpadded, window, padding, stride, mode, INPUT_NDIM, PARAM_NDIM, INPUT_SHAPE, PARAM_SHAPE)

        # # padding, stride shapes
        # self.padding, self.stride = self._mode(self.unpadded, self.window, self.padding, self.stride, self.mode)

        # # padded shape
        # self.padded = self._calc_padded(self.unpadded, self.padding, self.padded)

        # # padded array and indices
        # self.padding_indices = self._padding_indices(self.padded, self.padding)
        # self.padded_array = np.zeros((self.padded.shape)).astype(self.dtype)

        # # output shape
        # self.output = self._calc_output(self.padded, self.window, self.stride, self.output)

        # # im2col indices
        # i = self.PARAM_NDIM + 1
        # self.im2col_indices = im2col(self.padded.shape[-i:], self.window.shape[-i:], self.stride.shape)

        # shape check
        if not self.window.width <= self.padded.width or not self.window.height <= self.padded.height or not self.window.depth == self.padded.depth:
            raise ValueError(f"bad window dimensions")

    def correlate(self, images, kernels):
        # images
        images = self._expand(images, self.unpadded.shape)
        images = self._pad(images)
        im2cols = self._im2col(images)

        # kernels
        self.operation = "correlate"
        kernels = self._expand(kernels, self.window.shape)
        kr2row = self._kr2row(kernels)

        # correlate via dot product
        return np.matmul(kr2row, im2cols).reshape(self.output.shape).astype(self.dtype)

    def _mode(self, unpadded, window, padding, stride, mode):
        padding.width, stride.width = super()._mode(unpadded.width, window.width, padding.width, stride.width, mode)
        padding.height, stride.height = super()._mode(unpadded.height, window.height, padding.height, stride.height, mode)
        return padding, stride

    def _calc_padded(self, padded, unpadded, padding):
        padded.width = super()._calc_padded(unpadded.width, padding.width)
        padded.height = super()._calc_padded(unpadded.height, padding.height)
        return padded

    def _calc_output(self, padded, window, stride, output):
        output.width = super()._calc_output(padded.width, window.width, stride.width)
        output.height = super()._calc_output(padded.height, window.height, stride.height)
        output.depth = window.num
        output.num = padded.num
        return output

    def _padding_indices(self, padded, padding):
        cols = super()._padding_indices(padded.width, padding.width)
        rows = super()._padding_indices(padded.height, padding.height)
        return (rows, cols)

    def _pad(self, images):
        padded_array = self.padded_array
        rows, cols = self.padding_indices
        padded_array[..., rows, cols] = images
        return padded_array
