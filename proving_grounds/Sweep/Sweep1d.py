import numpy as np
from ShapeNd import ShapeNd, Dimension
from SweepNd import SweepNd


class Input3d(ShapeNd):
    def __init__(self, shape):
        super(Input3d, self).__init__(shape)

    width = Dimension(-1)
    depth = Dimension(-2)
    num = Dimension(-3)


class Param1d(ShapeNd):
    def __init__(self, shape):
        super(Param1d, self).__init__(shape)

    width = Dimension(-1)


class Sweep1d(SweepNd):
    operations = ("correlate", "convolve")
    modes = ("user", "full", "keep")
    dtype = np.float32

    def __init__(self, unpadded, window, padding, stride, mode="user"):
        INPUT_NDIM = 3  # number of elements in unpadded & window shape tuples
        PARAM_NDIM = 1  # number of elements in padding & stride shape tuples
        INPUT_SHAPE = Input3d  # ShapeNd object defining unpadded & window dimension attributes
        PARAM_SHAPE = Param1d  # ShapeNd object defining padding & stride dimension attributes

        super().__init__(unpadded, window, padding, stride, mode, INPUT_NDIM, PARAM_NDIM, INPUT_SHAPE, PARAM_SHAPE)

        # shape check
        if not self.window.width <= self.padded.width or not self.window.depth == self.padded.depth:
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
        return padding, stride

    def _calc_padded(self, padded, unpadded, padding):
        padded.width = super()._calc_padded(unpadded.width, padding.width)
        return padded

    def _calc_output(self, padded, window, stride, output):
        output.width = super()._calc_output(padded.width, window.width, stride.width)
        output.depth = window.num
        output.num = padded.num
        return output

    def _padding_indices(self, padded, padding):
        cols = super()._padding_indices(padded.width, padding.width)
        return cols

    def _pad(self, images):
        padded_array = self.padded_array
        cols = self.padding_indices
        padded_array[..., cols] = images
        return padded_array
