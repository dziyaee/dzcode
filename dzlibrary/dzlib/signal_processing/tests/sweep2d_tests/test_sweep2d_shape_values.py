import math
import numpy as np
import yaml
import pytest
from dzlib.signal_processing.sweep2d import Sweep2d
from dzlib.signal_processing.tests.sweep2d_tests.utils import generate_shape_test_params


# All of the tests in this module are for data computed during Sweep2d class instatiation / initialization.

# Test Inputs
settings_path = 'settings.yml'
with open(settings_path) as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
shapes_list, mode_list = generate_shape_test_params(settings)

# Test Fixtures
scope = "module"


# "actual" input fixtures. All fixtures below derive from these two
@pytest.fixture(scope=scope, params=mode_list)
def mode(request):
    '''Returns a Sweep2d input mode argument'''
    mode = request.param
    return mode


@pytest.fixture(scope=scope, params=shapes_list)
def shapes(request):
    '''Returns a set of Sweep2d input shape arguments (unpadded, window, padding, stride)'''
    shapes = request.param
    return shapes


# This fixture generates the results based directly on the "actual" inputs
@pytest.fixture(scope=scope)
def sweeper(shapes, mode):
    '''Returns a Sweep2d object to be tested'''
    sweeper = Sweep2d(*shapes, mode)
    return sweeper


# All of the fixtures below compute and/or return "expected" data which are calculated from the "actual" inputs, but are completely independent of the object being tested, which is returned from the "sweeper" fixture.
@pytest.fixture(scope=scope)
def expected(shapes, mode):
    '''Returns a tuple of all expected shapes. All other fixtures below derive their data from this one'''
    unpadded, window, padding, stride = shapes

    # Unpadded & Window (Unchanged)
    unpadded_num, unpadded_depth, unpadded_height, unpadded_width = unpadded
    window_num, window_depth, window_height, window_width = window

    # Padding & Stride (Can change based on mode)
    if mode == "user":
        stride_height, stride_width = stride
        padding_height, padding_width = padding

    elif mode == "full":
        stride_height, stride_width = 1, 1
        padding_height = window_height - stride_height
        padding_width = window_width - stride_width

    elif mode == "same":
        stride_height, stride_width = 1, 1
        padding_height = (window_height - stride_height) / 2
        padding_width = (window_width - stride_width) / 2

    else:
        raise ValueError(f"Expected mode in {mode_list}, got {mode}")

    padding = (padding_height, padding_width)
    stride = (stride_height, stride_width)

    # Padded (Calculated from Unpadded & Padding)
    padded_height = unpadded_height + 2 * padding_height
    padded_width = unpadded_width + 2 * padding_width
    padded = (unpadded_num, unpadded_depth, padded_height, padded_width)

    # Output (Calculated from Padded, Window, & Stride)
    output_height = int(((padded_height - window_height) // stride_height) + 1)
    output_width = int(((padded_width - window_width) // stride_width) + 1)
    output = (unpadded_num, window_num, output_height, output_width)

    # im2col (Calculated from Window & Output)
    im2col_height = window_depth * window_height * window_width
    im2col_width = output_height * output_width
    im2col_indices = (im2col_height, im2col_width)
    return (unpadded, window, padding, stride, padded, output, im2col_indices)


@pytest.fixture(scope=scope)
def unpadded(expected):
    '''Returns the unpadded shape'''
    unpadded, *_ = expected
    return unpadded


@pytest.fixture(scope=scope)
def window(expected):
    '''Returns the window shape'''
    _, window, *_ = expected
    return window


@pytest.fixture(scope=scope)
def padding(expected):
    '''Returns the padding shape'''
    _, _, padding, *_ = expected
    return padding


@pytest.fixture(scope=scope)
def stride(expected):
    '''Returns the stride shape'''
    _, _, _, stride, _, _, _ = expected
    return stride


@pytest.fixture(scope=scope)
def padded(expected):
    '''Returns the padded shape'''
    *_, padded, _, _ = expected
    return padded


@pytest.fixture(scope=scope)
def output(expected):
    '''Returns the output shape'''
    *_, output, _ = expected
    return output


@pytest.fixture(scope=scope)
def im2col_indices(expected):
    '''Returns the im2col_indices shape'''
    *_, im2col_indices = expected
    return im2col_indices


@pytest.fixture(scope=scope)
def padding_indices(padded, padding):
    '''Computes and returns the expected padding indices'''
    _, _, padded_height, padded_width = padded
    padding_height, padding_width = padding

    i1 = int(math.ceil(padding_height))
    i2 = int(math.ceil(padded_height - padding_height))
    rows = slice(i1, i2)

    i1 = int(math.ceil(padding_width))
    i2 = int(math.ceil(padded_width - padding_width))
    cols = slice(i1, i2)

    return np.s_[..., rows, cols]


# Tests
class Test_ShapeNd_Values:
    '''Tests all object instances of ShapeNd for correct .shape values'''

    def test_unpadded(self, sweeper, unpadded):
        assert sweeper.unpadded.shape == unpadded

    def test_window(self, sweeper, window):
        assert sweeper.window.shape == window

    def test_padding(self, sweeper, padding):
        assert sweeper.padding.shape == padding

    def test_stride(self, sweeper, stride):
        assert sweeper.stride.shape == stride

    def test_padded(self, sweeper, padded):
        assert sweeper.padded.shape == padded

    def test_output(self, sweeper, output):
        assert sweeper.output.shape == output


class Test_Numpy_NdArray_Shape_Values:
    '''Tests all objects instnaces of numpy.ndarray for correct .shape values'''

    def test_padded_array(self, sweeper, padded):
        assert sweeper.padded_array.shape == padded

    def test_im2col_indices(self, sweeper, im2col_indices):
        assert sweeper.im2col_indices.shape == im2col_indices


def test_padding_indices(sweeper, padding_indices):
    '''Tests the padding indices, a numpy.s_ object for correct values'''
    assert sweeper.padding_indices == padding_indices
