import pytest
import yaml
from dzlib.signal_processing.sweep2d import Sweep2d
from dzlib.signal_processing.tests.sweep2d_tests.utils import generate_shape_test_params


# Test Inputs
settings_path = 'settings.yml'
with open(settings_path) as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
shapes_list, mode_list = generate_shape_test_params(settings)

# Test Fixtures
scope = "module"


# First two fixtures generate the "actual" inputs
@pytest.fixture(scope=scope, params=mode_list)
def mode(request):
    mode = request.param
    return mode


@pytest.fixture(scope=scope, params=shapes_list)
def shapes(request):
    shapes = request.param
    return shapes


# This fixture generates the results based directly on the "actual" inputs
@pytest.fixture(scope=scope)
def sweeper(shapes, mode):
    sweeper = Sweep2d(*shapes, mode)
    return sweeper


# this fixture generates the "expected" results based directly on the "actual" inputs but completely independent of the sweeper fixture or Sweep2d class being tested
@pytest.fixture(scope=scope)
def expected(shapes, mode):
    unpadded, window, padding, stride = shapes

    # Unpadded & Window (Unchanged)
    unpadded_num, unpadded_depth, unpadded_height, unpadded_width = unpadded
    window_num, window_depth, window_height, window_width = window

    # Padding & Stride (Change based on mode)
    if mode == "user":
        stride_height, stride_width = stride
        padding_height, padding_width = padding

    elif mode == "full":
        stride_height, stride_width = 1, 1
        padding_height = window_height - stride_height
        padding_width = window_width - stride_width

    else:  # equivalent to elif mode == "same":
        stride_height, stride_width = 1, 1
        padding_height = (window_height - stride_height) / 2
        padding_width = (window_width - stride_width) / 2

    padding = (padding_height, padding_width)
    stride = (stride_height, stride_width)

    # Padded (Calculated from Unpadded & Padding)
    padded_height = unpadded_height + 2 * padding_height
    padded_width = unpadded_width + 2 * padding_width
    padded = (unpadded_num, unpadded_depth, padded_height, padded_width)

    # Output (Calculated from Padded, Window, & Stride)
    output_height = ((padded_height - window_height) // stride_height) + 1
    output_width = ((padded_width - window_width) // stride_width) + 1
    output = (unpadded_num, window_num, output_height, output_width)

    # im2col (Calculated from Window & Output)
    im2col_height = window_depth * window_height * window_width
    im2col_width = output_height * output_width
    im2col_indices = (im2col_height, im2col_width)
    return (unpadded, window, padding, stride, padded, output, im2col_indices)


# Remaining fixtures just return each expected output independently if needed
@pytest.fixture(scope=scope)
def unpadded(expected):
    unpadded, *_ = expected
    return unpadded


@pytest.fixture(scope=scope)
def window(expected):
    _, window, *_ = expected
    return window


@pytest.fixture(scope=scope)
def padding(expected):
    _, _, padding, *_ = expected
    return padding


@pytest.fixture(scope=scope)
def stride(expected):
    _, _, _, stride, _, _, _ = expected
    return stride


@pytest.fixture(scope=scope)
def padded(expected):
    *_, padded, _, _ = expected
    return padded


@pytest.fixture(scope=scope)
def output(expected):
    *_, output, _ = expected
    return output


@pytest.fixture(scope=scope)
def im2col_indices(expected):
    *_, im2col_indices = expected
    return im2col_indices


# Tests
class Test_Shape_Values:
    ''' Because all of the "expected" results logic happens in the expected fixture, these tests are very simple. They simply compare the "actual" results from the Sweep2d class instance returned by the sweeper fixture to the "expected" results returned directly or indirectly by the expected fixture
    '''
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

    def test_padded_array(self, sweeper, padded):
        assert sweeper.padded_array.shape == padded

    def test_output(self, sweeper, output):
        assert sweeper.output.shape == output

    def test_im2col_indices(self, sweeper, im2col_indices):
        assert sweeper.im2col_indices.shape == im2col_indices
