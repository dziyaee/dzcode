import numpy as np
from dzlib.signal_processing.shape import ShapeNd, Dimension
from dzlib.signal_processing.sweep import SweepNd


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
    modes = ("user", "full", "same")
    dtype = np.float32

    def __init__(self, unpadded, window, padding, stride, mode="user"):
        INPUT_NDIM = 4  # number of elements in unpadded & window shape tuples
        PARAM_NDIM = 2  # number of elements in padding & stride shape tuples
        INPUT_SHAPE = Input4d  # ShapeNd object defining unpadded & window dimension attributes
        PARAM_SHAPE = Param2d  # ShapeNd object defining padding & stride dimension attributes

        super().__init__(unpadded, window, padding, stride, mode, INPUT_NDIM, PARAM_NDIM, INPUT_SHAPE, PARAM_SHAPE)

        # shape check
        if not self.window.width <= self.padded.width or not self.window.height <= self.padded.height or not self.window.depth == self.padded.depth:
            raise ValueError(f"bad window dimensions")

    def _mode(self, unpadded, window, padding, stride, mode):
        padding.width, stride.width = super()._mode(unpadded.width, window.width, padding.width, stride.width, mode)
        padding.height, stride.height = super()._mode(unpadded.height, window.height, padding.height, stride.height, mode)
        return padding, stride

    def _padded_shape(self, padded, unpadded, padding):
        padded.width = super()._padded_shape(unpadded.width, padding.width)
        padded.height = super()._padded_shape(unpadded.height, padding.height)
        return padded

    def _output_shape(self, padded, window, stride, output):
        output.width = super()._output_shape(padded.width, window.width, stride.width)
        output.height = super()._output_shape(padded.height, window.height, stride.height)
        output.depth = window.num
        output.num = padded.num
        return output

    def _padding_indices(self, padded, padding):
        cols = super()._padding_indices(padded.width, padding.width)
        rows = super()._padding_indices(padded.height, padding.height)

        padding_indices = np.s_[..., rows, cols]
        return padding_indices

    def correlate2d(self, images, kernels):
        return super()._correlate(images, kernels)

    def convolve2d(self, images, kernels):
        return super()._convolve(images, kernels)
