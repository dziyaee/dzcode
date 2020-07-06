import numpy as np


class Signal():

    def __init__(self, array):
        self.array = array

    def _getarray(self):
        return self._array

    def _setarray(self, array):
        n = array.ndim
        assert 1 <= n <= 4, f"Invalid Number of Dimensions ({n}) for Signal Object. Must be between 1 and 4"

        self._array = np.asarray(array)

        dims = [None, None, None, None]
        dims[4-n:] = array.shape

        self.num, self.depth, self.height, self.width = dims

    def _delarray(self):
        del self._array
        del self.num
        del self.depth
        del self.height
        del self.width

    array = property(_getarray, _setarray, _delarray)


class SWP2d():
    ''' Scanning Window Processor (SWP)2d Class. Simulates a 2d scan of a 4d Image signal of shape (N x D x H x W) along the last 2 dimensions (H, W). Array Shape Convention is as follows:
        N or 'num' represents the Number of 1d, 2d, or 3d tensors stored in the array.
        D or 'depth' represents the Depth of any given 3d tensor stored in the array.
        H or 'height' represents the Height of any given 2d or 3d tensor stored in the array.
        W or 'width' represents the Width of any given 1d, 2d, or 3d tensors stored in the array.
    For lower dimensional inputs, reshape the input such that the corresponding dimensions = 1.
    The 2d scan is parametrized by a 2d window shape, 2d padding values, and 1d stride.
    '''

    # List of valid modes and operations
    modes = ["user", "full", "none", "auto", "keep"]
    operations = ["convolution", "correlation", "median"]

    def __init__(self, image, kernel_size, pad=0, stride=1, mode="user"):
        ''' SWP2d Object is initialized by calculating valid padding and strides according to the specified mode, image size, and kernel size. There are several asserts throughout the initialization to ensure that shapes requirements are met as these can changed based on the input parameters. Once a valid padding is calculated, the image is padded, the expected output dimensions are calculated, and an 'im2col' Matrix of the padded image is created. This Matrix is a variation of a Block Toeplitz Matrix of the image, where each column represents successive kernel_size patches. This Matrix need only be calculated one time for any given set of input parameters. Once this is done, a Matrix Multiplication can be performed with a Kernel 'Row Matrix' to perform Convolution, Cross-Correlation, or other operations.

            Arguments:

                image: 4d Array (N x D x H x W), where:
                    N = 1
                    D, H, W >= 1

                The following arguments can be passed as 1 Int or a Tuple of 2 Ints. If 1 Int is passed, it will be duplicated into a Tuple of 2 Ints. These will correspond to the Height and Width dimensions respectively:
                kernel_size: Size of Kernel along H and W
                pad: Size of Padding along H and W

                stride: Int, represents the number of pixels that the Kernel skips with each successive scan operation

                mode: String from the following list: ['user', 'full', 'none', 'auto', 'keep']
                    'user': Uses the user given or default values for pad and stride. The Padded Image Size must be greater than or equal to the Kernel Size.
                    'full': Uses stride of 1, and pad of Kernel Size - 1. This causes every element of the Kernel to scan over every element of the Image.
                    'none': Uses pad of 0, and any given stride or default value. The Image Size must be greater than or equal to the Kernel Size.
                    'auto': Uses minimum padding necessary to achieve a valid Padded Image Size that 'fits' the Kernel Size. If no padding is needed to 'fit' the Kernel Size within the Image Size, no padding is used along that axis. Uses any given stride or default value.
                    'keep': Uses necessary padding to keep Output Dimensions exactly the same as Input Dimensions. Note that this is not possible for every combination of Image Size, Kernel Size, and Stride. This is dependant on the parity of these values (even / odd). It is only possible to keep same dimensions for the following parity combinations:
                        (odd, odd, odd/even)
                        (odd/even, odd, odd)
                        (even, even, even)

                Notes:
                    if all defaults are used, mode = 'user' will use pad = 0 and stride = 1, which will be equivalent to mode = 'none' with stride = 1.

                    mode = 'auto' will use pad = 0 if no padding is required, which will be equivalent to mode = 'none' with equivalent strides. This mode is really only useful for cases with Kernel Size is larger than Image Size, in which case the minimum amount of padding to achieve an output size of >= 1 is used.
        '''

        # This is not used anywhere, but is stored as an attribute
        unpadded_shape = image.shape

        # Store Image as Signal object and ensure that Image is a 4d array (N x D x H x W) with N = 1
        image = Signal(image)
        assert image.num == 1, f"Invalid Image Shape: {image.array.shape}. Must be a 4d tensor with dims: (1 x D x H x W)"

        # Get Kernel Height and Width and ensure that they are greater than zero
        kernel_height, kernel_width = self.var2d(kernel_size)
        assert (kernel_height > 0) and (kernel_width > 0), f"Invalid Kernel Size {kernel_size}. Must be greater than zero"

        # Create Empty Kernel of Shape (N x D x H x W), where N = 1, D = Image Depth, H = Kernel height, W = Kernel Width
        kernel = np.zeros((1, image.depth, kernel_height, kernel_width))
        kernel = Signal(kernel)

        # Ensure mode is a valid mode within modes list
        assert mode in self.modes, f"Invalid mode '{mode}'. Must be one of: {self.modes}"

        # Ensure that stride it is greater than zero and a whole number
        assert (stride > 0) and (stride % 1 == 0), f"Invalid Stride ({stride}). Must be greater than zero and a whole number"

        # User mode will take given input argument for pad and use given or default stride
        if mode == "user":

            pad_height, pad_width = self.var2d(pad)

            assert (pad_height >= 0) and (pad_width >= 0), f"Invalid Pad Size {pad_height, pad_width} for mode '{mode}'. Pad Size must be greater than or equal to zero"

        # Full mode will pad the image by the Kernel Size - 1. This will always result in a sufficiently padded image, regardless of Image or Kernel Sizes. This mode ignores any user input of pad and stride and sets stride to 1
        elif mode == "full":

            pad_height = kernel.height - 1
            pad_width = kernel.width - 1
            stride = 1

        # None mode will set pad to zero. For this to be valid, Image Size must be greater than or equal to Kernel Size. This is unaffected by Stride
        elif mode == "none":

            pad_height = 0
            pad_width = 0

            assert (image.height >= kernel.height) and (image.width >= kernel.width), f"Incompatible Image/Kernel Sizes for mode '{mode}': Image Size {image.height, image.width} must be greater than or equal to Kernel Size {kernel.height, kernel.width}"

        # Auto mode will calculate the minimum padding necessary for a given input size and window size to guarantee a valid output size of greater than or equal to 1. This is unaffected by stride size.
        elif mode == "auto":

            # Function to calculate the minimum padding ((wdim - xdim) / 2). The max between the result and 0 is taken to set any negative pad values to zero. The ceiling is taken to set any non-whole numbers to the next highest whole number to ensure a valid padding. The function (wdim - xdim) / 2 comes from setting ydim to >= 1 and solving the ydim formula for pdim
            min_pad = lambda xdim, kdim : np.ceil(np.maximum(((kdim - xdim) / 2), 0))

            pad_height = min_pad(image.height, kernel.height)
            pad_width = min_pad(image.width, kernel.width)

        # Keep mode will calculate the minimum padding necessary to keep the Output Size equal to the Image Size for any given Image Size, Kernel Size, and Stride. The only valid combinations of (Image Size, Kernel Size, Stride) are (odd/even, odd, odd), or (odd, odd, odd/even), or (even, even, even). These combinations ensure that padding is a whole number. Whole number is checked via min_pad % 1 == 0. Alternatively, can check if 2*min_pad % 2 == 0 to ensure that the numerator is an even number prior to being divided by 2, which will ensure a whole number result.
        elif mode == "keep":

            min_pad = lambda xdim, kdim, stride : (((xdim - 1) * stride) - xdim + kdim) / 2

            pad_height = min_pad(image.height, kernel.height, stride)
            pad_width = min_pad(image.width, kernel.width, stride)

            assert pad_height % 1 == 0, f"Invalid Pad Height ({pad_height}) for mode '{mode}'. Unable to preserve Image Height with given parameters. Image Height: {image.height}, Kernel Height: {kernel.height}, Stride: {stride}) must be (odd/even, odd, odd), or (odd, odd, odd/even), or (even, even, even)"

            assert pad_width % 1 == 0, f"Invalid Pad Width ({pad_width}) for mode '{mode}'. Unable to preserve Image Width with given parameters. Image Width: {image.width}, Kernel Width: {kernel.width}, Stride: {stride}) must be (odd/even, odd, odd), or (odd, odd, odd/even), or (even, even, even)"

        # Padding and Stride are now all set based on Image Size, Kernel Size, and Mode
        # Pad Image and check that Padded Image Size is greater than or equal to Kernel Size
        pad_height = int(pad_height)
        pad_width = int(pad_width)

        # Call pad2d method only if pad is not zero on both H and W dimensions
        if (pad_height == 0) and (pad_width == 0):
            pass

        else:
            image = self.pad2d(image, pad_height, pad_width)

        assert (image.height >= kernel.height) and (image.width >= kernel.width), f"Incompatible Image/Kernel Sizes for mode '{mode}'. Padded Image Size {image.height, image.width} must be greater than or equal to Kernel Size {kernel.height, kernel.width}"

        # Function to calculate the number of scanning operations (ops) along Height and Width for a given Padded Image Size, Kernel Size, and Stride
        # In most cases, this is equivalent to the Output Height and Width
        # The Output Depth is equal to the Number of Kernels, which defaults to 1 during the init. This value can change based on Kernels passed as input arguments to methods for this class.
        ydim = lambda pxdim, kdim, stride : int(((pxdim - kdim) / stride) + 1)
        output_depth = kernel.num
        output_height = ydim(image.height, kernel.height, stride)
        output_width = ydim(image.width, kernel.width, stride)

        # Create im2col Matrix using the now padded image, empty kernel (for sizes), output size, and stride
        im2col = self.im2cols(image, kernel, output_height, output_width, stride)

        # Assign Instance attributes
        # Sweep Params
        self.mode = mode
        self.kernel_size = (kernel_height, kernel_width)
        self.padding = (pad_height, pad_width)
        self.stride = int(stride)

        # Signal Objects
        self.image = image
        self.kernel = kernel
        self.im2col = im2col

        # Shapes
        self.unpadded_shape = unpadded_shape
        self.padded_shape = image.array.shape
        self.output_shape = (output_depth, output_height, output_width)
        self.output_depth = output_depth
        self.output_height = output_height
        self.output_width = output_width


    def var2d(self, var):
        ''' Function to check if a variable is an int or tuple of 2 ints and convert it into a tuple of 2 ints if needed

            Arguments:
                var: variable

            Returns:
                var: (tuple of 2 ints) if possible
        '''

        # If var is an int, duplicate var and store in a tuple
        if isinstance(var, (int, np.integer)):

            var = (var, var)

        # If var is a tuple of 2 ints, pass
        elif isinstance(var, tuple) and all(isinstance(n, (int, np.integer)) for n in var) and (len(var) == 2):

            pass

        # Else, raise exception
        else:
            raise Exception(f"{var, type(var), var[0], var[1], len(var)} must be either 1 int or tuple of 2 ints")

        return var


    def pad2d(self, image, pad_height, pad_width):
        ''' Function to pad a 4d Signal object with array of shape (N x D x H x W) along the last 2 dimensions (H, W)

        Returns:
            padded_image: Signal object with array of shape (N x D x H x W), where new H, W = old H, W + 2 * pad
        '''

        # Padded Image Size = Image Size + 2 * Pad
        Ph = image.height + 2 * pad_height
        Pw = image.width + 2 * pad_width

        # Create array with new padded dimensions and overlay original image into the center
        padded_image = np.zeros((image.num, image.depth, Ph, Pw))
        padded_image[:, :, pad_height: Ph - pad_height, pad_width: Pw - pad_width] = image.array[:, :, :, :]

        image = Signal(padded_image)

        return image


    def im2cols(self, image, kernel, output_height, output_width, stride):
        ''' Function to convert a 4d Image Array to a 2d Matrix where each column corresponds to successive 3d Kernel-Size patches of the Image, flattened along (D x H x W), parametrized by Kernel Size and Stride

            Returns:
                im2col: Signal object with array of shape (H x W)
        '''

        # im2col dimensions (H x W), where:
        # H = Size of Flattened Kernel along (D x H x W)
        # W = Size of Flattened Output along (H x W)
        im2col_height = kernel.depth * kernel.height * kernel.width
        im2col_width = output_height * output_width

        # Create Signal object containing Empty im2col Matrix
        im2col = np.zeros((im2col_height, im2col_width))
        im2col = Signal(im2col)

        # Row and Col indices are both a function of number of im2col columns (im2col W), number of column operations (Output W), and Stride
        # Floor Division is used for Rows, Modulus is used for Cols
        rows = ((np.arange(im2col.width) // output_width) * stride).astype(np.int32)
        cols = ((np.arange(im2col.width) % output_width) * stride).astype(np.int32)

        # Iterate through im2col columns and use the row and column indices to insert flattened Image patches
        for i in range(im2col.width):

            im2col.array[:, i] = image.array[:, :, rows[i]: rows[i] + kernel.height, cols[i]: cols[i] + kernel.width].flatten()

        return im2col


    def kr2rows(self, operation):
        ''' Function to convert a 4d Kernel Array to a 2d Matrix where each row corresponds to successive 3d Kernels, flattened along (D x H x W)

            Arguments:
                operation: String, If 'convolution', the kernel is flipped along the last 2 axes (H, W) prior to flattening. This is equivalent to performing a time-reversal of the Kernel which is done during a convolution. If 'correlation', the same operation is performed, but without flipping the Kernel.

            Returns:
                kr2row: Signal object with 2d array of shape (H x W)
        '''

        # For convolution, flip the Kernel along the (H, W), then flatten along the (D, H, W)
        if operation == "convolution":

            kr2row = np.flip(self.kernel.array, axis=(2, 3)).reshape(self.kernel.num, -1)

        # For correlation, just flatten along (D, H, W) without flipping
        elif operation == "correlation":

            kr2row = self.kernel.array.reshape(self.kernel.num, -1)

        else:

            raise Exception("'operation' must be either 'convolution' or 'correlation")

        kr2row = Signal(kr2row)

        return kr2row


    def kr2cols(self):
        kr2col = self.kernel.array.reshape(-1, 1)
        kr2col = Signal(kr2col)
        return kr2col


    def init_kernel(self, kernel):
        ''' Function to initialize a Kernel Signal object after passing shape checks

            Arguments:
                kernel: 4d Array of shape (N x D x H x W)

            Returns:
                kernel: Signal object with 4d array of shape (N x D x H x W)
        '''

        # Check that Array is 4d
        assert kernel.ndim == 4, "Kernel must be a 4d tensor (N x D x H x W)"
        k_num, k_depth, k_height, k_width = kernel.shape

        # Check that Kernel Depth = Image Depth, Kernel Height, Width = Kernel Size
        assert (k_depth == self.image.depth) and (k_height == self.kernel.height) and (k_width == self.kernel.width), \
        f"Actual Kernel Shape ({kernel.shape}) must be equal to Expected Kernel Shape (N, {self.image.depth, self.kernel.height, self.kernel.width})"

        kernel = Signal(kernel)

        return kernel


    def convolve(self, kernel):
        ''' Function to perform a convolution of the stored Image Array and a given Kernel Array

            Arguments:
                kernel: Array of shape (N x D x H x W)

            Returns:
                output: Signal object with 3d array of shape (D x H x W) representing the result of the Convolution of the Image with the Kernel. Output D = Kernel N. Output (H, W) are calculated during the init and are a function of Image Size, Kernel Size, Padding, and Stride.
        '''

        # Init Kernel (returns Signal object)
        self.kernel = self.init_kernel(kernel)

        # Convert Kernel Array to a Conv kr2row Matrix
        self.kr2row = self.kr2rows(operation = "convolution")

        # Output D = Kernel N
        self.output_depth = self.kernel.num

        # Convolve by performing a matrix multiplication of the Conv kr2row and im2col matricies and reshape to proper output dimensions
        output = np.matmul(self.kr2row.array, self.im2col.array).reshape(self.output_depth, self.output_height, self.output_width)
        self.output = Signal(output)

        return output


    def correlate(self, kernel):
        ''' Function to perform a cross-correlation of the stored Image Array and a given Kernel Array

            Arguments:
                kernel: Array of shape (N x D x H x W)

            Returns:
                output: Signal object with 3d array of shape (D x H x W) representing the result of the Cross-Correlation of the Image with the Kernel. Output D = Kernel N. Output (H, W) are calculated during the init and are a function of Image Size, Kernel Size, Padding, and Stride.
        '''

        # Init Kernel (returns Signal object)
        self.kernel = self.init_kernel(kernel)

        # Convert Kernel Array to a Corr kr2row Matrix
        self.kr2row = self.kr2rows(operation = "correlation")

        # Output D = Kernel N
        self.output_depth = self.kernel.num

        # Cross-Correlate by performing a matrix multiplication of the Corr kr2row and im2col matricies and reshape to proper output dimensions
        output = np.matmul(self.kr2row.array, self.im2col.array).reshape(self.output_depth, self.output_height, self.output_width)
        self.output = Signal(output)

        return output

    def MAE(self, kernel):
        self.kernel = self.init_kernel(kernel)
        self.kr2col = self.kr2cols()
        output = np.mean(np.abs(self.im2col.array - self.kr2col.array), axis=0)
        output = Signal(output)
        return output

    def median(self):
        ''' Function to perform a Median Filter of the Image. This function has no Kernels, so output D = 1

            Returns:
                output: Signal object with 3d array of shape (D x H x W) representing the result of the Median Filter Scan of the Image. Output D = 1. Output (H, W) are calculated during the init and are a function of Image Size, Kernel Size, Padding, and Stride.
        '''

        # Output D = 1
        self.output_depth = 1

        # Median Filter calculates the median of the 2d im2col matrix along axis 0 (down each column / across rows). Reshape to proper output dimensions
        output = np.median(self.im2col.array, axis=0).reshape(self.output_depth, self.output_height, self.output_width)
        self.output = Signal(output)

        return output


    def print_sweep2d_attrs(self):
        ''' Function to print SWP2d Instance Attributes'''

        print(f"\nSWP2d Instance Attributes:\n")

        for key, value in self.__dict__.items():
            if not isinstance(value, Signal):

                print(f"{key:15} {value}")

        return None


    def print_signal_attrs(self):
        ''' Function to print Signal Attributes'''
        print(f"\nSignal Instance Attributes:\n")

        for key1, value1 in self.__dict__.items():
            if isinstance(value1, Signal):

                for key2, value2 in value1.__dict__.items():

                    if type(value2) == np.ndarray:
                        print(f"\n{key1}.{key2} = ")
                        print(f"{value2}\n")
                        print(f"{key1}.{key2}.shape{'':4} =  {value2.shape}")

                    else:
                        print(f"{key1}.{key2:15} =  {value2}")

                print("-" * 100)

        return None




# Signal Test
if __name__ == '__main__':

    arr4 = np.random.randint(1, 10, (1, 2, 3, 4))
    arr3 = np.random.randint(1, 10, (1, 2, 3))
    arr2 = np.random.randint(1, 10, (1, 2))
    arr1 = np.random.randint(1, 10, (1))

    arrs = [arr1, arr2, arr3, arr4]

    for arr in arrs:

        y = Signal(arr)
        print(f"\narr ndim =  {y.array.ndim}")
        print(f"arr shape =   {y.array.shape}")
        print(f"Num =         {y.num}")
        print(f"Depth =       {y.depth}")
        print(f"Height =      {y.height}")
        print(f"Width =       {y.width}\n")

    pass
