import math
import numpy as np
import pytest
import yaml
from dzlib.signal_processing.sweep2d import Sweep2d
from dzlib.signal_processing.tests.sweep2d_tests.utils import generate_shape_test_params


# Sweep2d Test Inputs
settings_path = 'settings.yml'
with open(settings_path) as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

# inputs for valid tests
valid_settings = settings['Unit']['Valid']
valid_shapes_list, valid_mode_list = generate_shape_test_params(valid_settings)

# inputs for exception tests
invalid_settings = settings['Unit']['Invalid']


# Test Fixtures
scope = "module"


@pytest.fixture(scope=scope, params=valid_mode_list)
def mode(request):
    '''Input fixture: returns a mode string argument for Sweep2d instantiation'''
    mode = request.param
    return mode


@pytest.fixture(scope=scope, params=valid_shapes_list)
def shapes(request):
    '''Input fixture: returns a set of shape tuple arguments (unpadded, window, padding, stride) for Sweep2d instantiation'''
    shapes = request.param
    return shapes


@pytest.fixture(scope=scope)
def sweeper(shapes, mode):
    '''Actual results fixture: returns a Sweep2d object to be tested'''
    sweeper = Sweep2d(*shapes, mode)
    return sweeper


@pytest.fixture(scope=scope, params=["num", "depth", "height", "width"])
def sweeper_shape_dims(request, sweeper):
    '''Returns a tuple of each of the primary ShapeNd values per dimension when parametrized indirectly'''
    unpadded = sweeper.unpadded.shape
    window = sweeper.window.shape
    padding = sweeper.padding.shape
    stride = sweeper.stride.shape
    padded = sweeper.padded.shape
    output = sweeper.output.shape

    if request.param == "num":
        return (unpadded[-4], window[-4], None, None, padded[-4], output[-4])

    elif request.param == "depth":
        return (unpadded[-3], window[-3], None, None, padded[-3], output[-3])

    elif request.param == "height":
        return (unpadded[-2], window[-2], padding[-2], stride[-2], padded[-2], output[-2])

    elif request.param == "width":
        return (unpadded[-1], window[-1], padding[-1], stride[-1], padded[-1], output[-1])


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
        raise ValueError(f"Expected mode in {valid_mode_list}, got {mode}")

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
