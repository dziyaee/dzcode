import numpy as np
from dzlib.signal_processing.utils import im2col
from dzlib.common.data import Shape


class Sweep2d():
    def __init__(self, images_shape, kernels_shape, padding=0, stride=1, mode='user'):
        # convert inputs to tuples and validate against max lengths and min values
        images_shape = self._totuple(images_shape, 4, 1)
        kernels_shape = self._totuple(kernels_shape, 4, 1)
        padding = self._totuple(padding, 2, 0)
        stride = self._totuple(stride, 2, 1)

        # expand inputs to max lengths with fill values
        images_shape = self._expand(images_shape, 4, 1)
        kernels_shape = self._expand(kernels_shape, 4, 1)
        padding = self._expand(padding, 2, padding[0])
        stride = self._expand(stride, 2, stride[0])

        # create Shape objects from images_shape and kernels_shape
        xx = Shape(images_shape)
        kk = Shape(kernels_shape)

        # validate kernels depth against images depth
        if not kk.depth == xx.depth:
            raise ValueError(f"Kernel and image depths must be equal, got {kk.depth}, {xx.depth}")

        # return padding and stride based on mode
        padding, stride = self._mode(xx, kk, padding, stride, mode)
        padding_height, padding_width = padding
        stride_height, stride_width = stride

        # calculate padded dimensions, update images Shape object
        xdim = lambda x, p: int(x + 2 * p)
        padded_height = xdim(xx.height, padding_height)
        padded_width = xdim(xx.width, padding_width)
        xx.shape = (xx.num, xx.depth, padded_height, padded_width)

        # validate kernels shape against padded images shape
        if not (kk.height <= xx.height) or not (kk.width <= xx.width):
            raise ValueError(f"Kernel dimensions must be <= image dimensions, got {kk.height, kk.width}, {xx.height, xx.width}")

        # calculate output dimensions, create output Shape object
        ydim = lambda x, k, s: int(np.floor((x - k) / s) + 1)
        output_height = ydim(xx.height, kk.height, stride_height)
        output_width = ydim(xx.width, kk.width, stride_width)
        output_shape = (xx.num, kk.num, output_height, output_width)
        yy = Shape(output_shape)

        # compute im2col indices matrix
        self.indices = im2col(xx.shape, kk.shape, stride)

        # init padded array and calculate padding indices with left side bias
        self.padded = np.zeros((xx.shape)).astype(np.float32)
        self.rows = int(np.floor(padding_height)), int(np.floor(xx.height - padding_height))
        self.cols = int(np.floor(padding_width)), int(np.floor(xx.width - padding_width))

        # assign Shape objects
        self.xx = xx
        self.kk = kk
        self.yy = yy

        # assign sweep params
        self.padding = padding
        self.stride = stride

    @staticmethod
    def _totuple(input_, max_len, min_val):
        # convert input_ to tuple if valid
        if isinstance(input_, int):
            input_ = (input_,)

        elif isinstance(input_, (list, tuple)):
            input_ = tuple(input_)

            # validations
            if len(input_) > 4 or len(input_) == 0:
                raise ValueError(f"number of elements in {input_} must be between 1 and {max_len}, got {len(input_)}")

        # more validations
        else:
            raise TypeError(f"{input_} must be of types (list, tuple), got {type(input_)}")

        # a few more validations
        for x in input_:
            if not isinstance(x, int):
                raise TypeError(f"elements in {input_} must be of type (int), got {type(x)}")

            if not x >= min_val:
                raise ValueError(f"elements in {input_} must be >= {min_val}, got {x}")

        return input_

    @staticmethod
    def _expand(input_, len_, fill):
        new = [fill] * len_
        i = len(input_)
        new[-i:] = input_
        return tuple(new)

    @staticmethod
    def _mode(xx, kk, padding, stride, mode):
        modes = ('user', 'full', 'keep')
        if mode not in modes:
            raise ValueError(f"mode must be in {modes}, got {mode}")

        if mode == 'user':
            # user mode: no change to default or input padding and stride values
            pass

        elif mode == 'full':
            # full mode: stride = 1, padding =  kernel size - 1
            padding = kk.height - 1, kk.width - 1
            stride = 1, 1

        elif mode == 'keep':
            # keep mode: calculate minimum padding necessary using user or default stride values with image and kernel dimensions
            # note: the minimum padding may not be an even number, thus resulting in extra padding on the 'right' side
            pmin = lambda x, k, s: (k - s + x * (s - 1)) / 2
            padding = pmin(xx.height, kk.height, stride[0]), pmin(xx.width, kk.width, stride[1])

        return padding, stride

    def correlate2d(self, images, kernels):
        # validate inputs and expand to 4d if valid
        images = self._make4d(images)
        kernels = self._make4d(kernels)

        # pad images
        images = self._pad2d(images)

        # validate input shapes
        if images.shape != self.xx.shape:
            raise ValueError(f"Expected images shape to be {self.xx.shape}, got {images.shape}")

        if kernels.shape != self.kk.shape:
            raise ValueError(f"Expected kernels shape to be {self.kk.shape}, got {kernels.shape}")

        # create im2col matrices
        im2cols = self._im2col(images)

        # create kr2row matrix
        kr2row = self._kr2row(kernels, 'correlate2d')

        # corr via dot product
        yy = self.yy
        outputs = np.matmul(kr2row, im2cols).reshape(yy.shape)
        return outputs

    def convolve2d(self, images, kernels):
        # validate inputs and expand to 4d if valid
        images = self._make4d(images)
        kernels = self._make4d(kernels)

        # pad images
        images = self._pad2d(images)

        # validate input shapes
        if images.shape != self.xx.shape:
            raise ValueError(f"Expected images shape to be {self.xx.shape}, got {images.shape}")

        if kernels.shape != self.kk.shape:
            raise ValueError(f"Expected kernels shape to be {self.kk.shape}, got {kernels.shape}")

        # create im2col matrices
        im2cols = self._im2col(images)

        # create kr2row matrix
        kr2row = self._kr2row(kernels, 'convolve2d')

        # conv via dot product
        yy = self.yy
        outputs = np.matmul(kr2row, im2cols).reshape(yy.shape)
        return outputs

    def median2d(self, images):
        # validate inputs and expand to 4d if valid
        images = self._make4d(images)
        kernels = self._make4d(kernels)

        # pad images
        images = self._pad2d(images)

        # validate input shapes
        if images.shape != self.xx.shape:
            raise ValueError(f"Expected images shape to be {self.xx.shape}, got {images.shape}")

        if kernels.shape != self.kk.shape:
            raise ValueError(f"Expected kernels shape to be {self.kk.shape}, got {kernels.shape}")

        # create im2col matrices
        im2cols = self._im2col(images)

        # update output shape depth to 1 (because number of kernels = 1 for median2d operation)
        xx = self.xx
        yy = self.yy
        output_shape = (xx.num, 1, yy.height, yy.width)
        yy = Shape(output_shape)

        # Median Filter calculates the median of the 3d im2col matrix along axis 1 (down through each column / across rows). Reshape to proper output dimensions
        outputs = np.median(im2cols, axis=1).reshape(yy.shape)
        return outputs

    def mean2d(self, images):
        # validate inputs and expand to 4d if valid
        images = self._make4d(images)
        kernels = self._make4d(kernels)

        # pad images
        images = self._pad2d(images)

        # validate input shapes
        if images.shape != self.xx.shape:
            raise ValueError(f"Expected images shape to be {self.xx.shape}, got {images.shape}")

        if kernels.shape != self.kk.shape:
            raise ValueError(f"Expected kernels shape to be {self.kk.shape}, got {kernels.shape}")

        # create im2col matrices
        im2cols = self._im2col(images)

        # update output shape depth to 1 (because number of kernels = 1 for mean2d operation)
        xx = self.xx
        yy = self.yy
        output_shape = (xx.num, 1, yy.height, yy.width)
        yy = Shape(output_shape)

        # Median Filter calculates the median of the 3d im2col matrix along axis 1 (down through each column / across rows). Reshape to proper output dimensions
        outputs = np.mean(im2cols, axis=1).reshape(yy.shape)
        return outputs

    @staticmethod
    def _make4d(array):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"input must be of type (np.ndarray), got {type(array)}")

        if not array.ndim <= 4:
            raise ValueError(f"input ndims must be <= 4, got {array.ndim}")

        # if ndim = 4, skip this step for time savings
        if array.ndim < 4:
            array = array[(None,) * (4 - array.ndim)]

        return array

    def _pad2d(self, images):
        r1, r2 = self.rows
        c1, c2 = self.cols
        padded = self.padded

        padded[:, :, r1: r2, c1: c2] = images
        return padded

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


if __name__ == "__main__":
    images_shape = (1, 3, 100, 100)
    kernels_shape = (1, 3, 3, 3)
    padding = 0, 0
    stride = 1, 1
    mode = 'user'
    sweeper = Sweep2d(images_shape, kernels_shape, padding, stride, mode)
    print(f"images shape:  {images_shape}")
    print(f"kernels shape: {sweeper.kk.shape}")
    print(f"padding:       {sweeper.padding}")
    print(f"stride:        {sweeper.stride}")
    print(f"padded shape:  {sweeper.xx.shape}")
    print(f"output shape:  {sweeper.yy.shape}")
    print(f"padding rows:  {sweeper.rows}")
    print(f"padding cols:  {sweeper.cols}")
