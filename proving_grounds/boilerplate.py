
## programmatic class properties example
width =  property(lambda self: self._get_dim(-1), lambda self, val: self._set_dim(-1, val))
height = property(lambda self: self._get_dim(-2), lambda self, val: self._set_dim(-2, val))
depth =  property(lambda self: self._get_dim(-3), lambda self, val: self._set_dim(-3, val))
num =    property(lambda self: self._get_dim(-4), lambda self, val: self._set_dim(-4, val))


## slicing of arbitrary ndim array
cols = slice(1, 3)
rows = slice(1, 3)
i = np.s_[..., rows, cols]
# example input / output
x = np.zeros((1, 1, 4, 4))
x[i] = 1


## matplotlib tick labels w/ scientific notation, scilimits can be adjusted to determine range where scientific notation is used
fig, ax1 = plt.subplots(ncols=1, nrows=1)
ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))


## timer decorator with auto unit calc
def timer(func):
    units = ('s', 'ms', 'us', 'ns')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        stop_time = time.perf_counter()
        run_time = stop_time - start_time
        m = int(np.abs(np.log10(run_time) // 3))
        run_time *= (1e3) ** m
        print(f"{func.__name__!r} time: {(run_time):.1f} {units[m]}")
        return value
    return wrapper


## ntimes decorator preserving normal function
def ntimes(func):
    def inner(n, *args, **kwargs):
        def wrapper():
            return [func(*args, **kwargs) for i in range(n)]

        result = wrapper()
        return result
    func.ntimes = inner
    return func

## rearranging input args in yaml file
inputs = [[tuple(input_arg) for input_arg in input_args.values()] for input_args in tests.values()]



class Im():
    '''Class to store and correlate multiple colorspace representations of a numpy array image. Colorspace representations are as follows:
    RGB: Red, Green, Blue; Shape (3 x H x W)
    HSV: Hue, Saturation, Value; Shape (3 x H x W)
    GS: Grayscale; Shape (H x W)
    BN: Binary; Shape (H x W)
    Class is instantiated with no inputs. If an RGB image is set via the RGB attribute, the HSV and GS images will be derived and set from the RGB image. Similar, setting a HSV image will derive and set the RGB and GS images. Settings a GS image will set the RGB and HSV images to None. Settings a BN image will have no effect on the other attributes as the Binary representation is arbitrary.'''
    def __init__(self):
        self._rgb = None
        self._hsv = None
        self._gs = None
        self._bn = None

    def _rgb_to_hsv(self):
        return rgb_to_hsv(self.rgb.transpose(1, 2, 0)).transpose(2, 0, 1)

    def _hsv_to_rgb(self):
        return hsv_to_rgb(self.hsv.transpose(1, 2, 0)).transpose(2, 0, 1)

    def _hsv_to_gs(self):
        return hsv_to_gs(self.hsv)

    @property
    def rgb(self):
        return self._rgb

    @rgb.setter
    def rgb(self, array):
        if (array.ndim != 3 and array.shape[0] != 3):
            raise ValueError(f"Expected array of shape (3 x H x W), got {array.shape}")
        else:
            self._rgb = array
            self._hsv = self._rgb_to_hsv()
            self._gs = self._hsv_to_gs()

    @property
    def hsv(self):
        return self._hsv

    @hsv.setter
    def hsv(self, array):
        if (array.ndim != 3 and array.shape[0] != 3):
            raise ValueError(f"Expected array of shape (3 x H x W), got {array.shape}")
        else:
            self._hsv = array
            self._rgb = self._hsv_to_rgb()
            self._gs = self._hsv_to_gs()

    @property
    def gs(self):
        return self._gs

    @gs.setter
    def gs(self, array):
        if array.ndim != 2:
            raise ValueError(f"Expected array of shape (H x W), got {array.shape}")
        else:
            self._gs = array
            self._rgb = None
            self._hsv = None

    @property
    def bn(self):
        return self._bn

    @bn.setter
    def bn(self, array):
        if array.ndim != 2:
            raise ValueError(f"Expected array of shape (H x W), got {array.shape}")
        elif np.unique(array) != 2:
            raise ValueError(f"Expected array with 2 unique values, {np.unique(array)}")
        else:
            self._bn = array

    @staticmethod
    def show(axis, array, *args, **kwargs):
        try:
            axis.imshow(array, *args, **kwargs)
        except TypeError:
            axis.imshow(array.transpose(1, 2, 0), *args, **kwargs)
        axis.set_xlabel(array.shape)
        return None


def matrix_to_latex(array):
    '''Convert a 2d array to a latex-formatted string'''
    string = r"\begin{bmatrix} " + "\n"
    width = array.shape[-1]
    for i, x in enumerate(array.flatten()):
        sep = " & "
        if (i + 1) % width == 0:
            sep = r" \\ " + "\n"
        string += str(x) + sep
    return string + r"\end{bmatrix}"
