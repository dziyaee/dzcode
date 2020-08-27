import math
import numpy as np
import numbers
from dzlib.signal_processing.utils import im2col


class SweepNd():
    operations = ("correlate", "convolve")
    modes = ("user", "full", "same")
    dtype = np.float32

    def __init__(self, unpadded, window, padding, stride, mode, INPUT_NDIM, PARAM_NDIM, INPUT_SHAPE, PARAM_SHAPE):
        # REPLACE INPUT_ / PARAM_ NDIM WITH LEN(INPUT_ / PARAM_ SHAPE)

        # inputs to tuples
        unpadded = self._totuple(unpadded)
        window = self._totuple(window)
        padding = self._totuple(padding)
        stride = self._totuple(stride)

        # input tuple validations
        self._validate(unpadded, INPUT_NDIM, 1)
        self._validate(window, INPUT_NDIM, 1)
        self._validate(padding, PARAM_NDIM, 0)
        self._validate(stride, PARAM_NDIM, 1)

        # Shape objects
        unpadded = INPUT_SHAPE(unpadded)
        window = INPUT_SHAPE(window)
        padding = PARAM_SHAPE(padding)
        stride = PARAM_SHAPE(stride)
        padded = unpadded.copy()
        output = padded.copy()

        # padding & stride shapes
        padding, stride = self._mode(unpadded, window, padding, stride, mode)

        # padded shape
        padded = self._padded_shape(padded, unpadded, padding)

        # padding indices & padded array
        padding_indices = self._padding_indices(padded, padding)
        padded_array = np.zeros((padded.shape)).astype(self.dtype)

        # output shape
        output = self._output_shape(padded, window, stride, output)

        # im2col indices
        i = PARAM_NDIM + 1
        im2col_indices = im2col(padded.shape[-i:], window.shape[-i:], stride.shape)

        # sweep axes (currently only used for kr2row method)
        SWEEP_AXES = tuple(i for i in range(INPUT_NDIM)[-PARAM_NDIM:])

        # Assign Shape objects
        self.unpadded = unpadded
        self.window = window
        self.padding = padding
        self.stride = stride
        self.padded = padded
        self.output = output

        # Assign other data
        self.padded_array = padded_array
        self.padding_indices = padding_indices
        self.im2col_indices = im2col_indices
        self.mode = mode

        # Assign constants
        self.INPUT_NDIM = INPUT_NDIM
        self.PARAM_NDIM = PARAM_NDIM
        self.SWEEP_AXES = SWEEP_AXES

    @staticmethod
    def _totuple(x):
        try:
            return tuple(x)
        except TypeError:
            return (x,)

    @staticmethod
    def _validate(x, length, min_value):
        if not len(x) == length:
            raise ValueError(f"Length of {x} must = {length}, got {len(x)}")

        if not all(i >= min_value for i in x):
            raise ValueError(f"All elements in {x} must be >= {min_value}")

        if not all(isinstance(i, numbers.Integral) for i in x):
            raise TypeError(f"All elements in {x} must be integers")
        return None

    def _mode(self, x, k, p, s, mode):
        if mode == "user":
            return p, s

        elif mode == "full":
            s = 1
            p = k - 1
            return p, s

        elif mode == 'same':
            s = 1
            p = (k - s + x * (s - 1)) / 2  # do not int() this; multiples of 0.5 are allowed
            return p, s

        else:
            if mode in self.modes:
                raise ValueError(f"Mode {mode} has no defined behavior")
            else:
                raise ValueError(f"Mode must be in {self.modes}, got {mode}")

    @staticmethod
    def _padded_shape(x, p):
        return int(x + 2 * p)

    @staticmethod
    def _output_shape(x, k, s):
        return int(((x - k) // s) + 1)

    @staticmethod
    def _padding_indices(x, p):
        i1 = int(math.ceil(p))
        i2 = int(math.ceil(x - p))
        return slice(i1, i2)

    def _expand(self, array, shape):
        diff = self.INPUT_NDIM - array.ndim
        if diff != 0:
            array = array[(None,) * diff]

        if not array.shape == shape:
            raise ValueError(f"Array shape must = {shape}, got {array.shape}")
        return array

    def _pad(self, unpadded_array):
        i = self.padding_indices
        padded_array = self.padded_array  # faster to not copy
        padded_array[i] = unpadded_array
        return padded_array

    def _im2col(self, images):
        return np.array([np.take(image, self.im2col_indices) for image in images])

    def _kr2row(self, kernels):
        window = self.window
        operation = self.operation
        SWEEP_AXES = self.SWEEP_AXES

        if operation == "correlate":
            return kernels.reshape(window.num, -1).astype(self.dtype)

        elif operation == "convolve":
            return np.flip(kernels, axis=SWEEP_AXES).reshape(window.num, -1).astype(self.dtype)

        else:
            raise ValueError(f"operation must be in {self.operations}, got {operation}")

    def _correlate(self, images, kernels):
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

    def _convolve(self, images, kernels):
        # images
        images = self._expand(images, self.unpadded.shape)
        images = self._pad(images)
        im2cols = self._im2col(images)

        # kernels
        self.operation = "convolve"
        kernels = self._expand(kernels, self.window.shape)
        kr2row = self._kr2row(kernels)

        # convolve via dot product
        return np.matmul(kr2row, im2cols).reshape(self.output.shape).astype(self.dtype)

    def _median(self, images):
        # images
        images = self._expand(images, self.unpadded.shape)
        images = self._pad(images)
        im2cols = self._im2col(images)
        median = np.median(im2cols, axis=1)
        return median.reshape(self.output.shape).astype(self.dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(unpadded = {self.unpadded.shape}, window = {self.window.shape}, padding = {self.padding.shape}, stride = {self.stride.shape}, mode = '{self.mode}', id = {hex(id(self))})"

    def __str__(self):
        return f"{self.__class__.__name__}(unpadded = {self.unpadded}, window = {self.window}, padding = {self.padding}, stride = {self.stride}, mode = '{self.mode}')"
