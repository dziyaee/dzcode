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


@pytest.fixture(scope=scope, params=mode_list)
def mode(request):
    mode = request.param
    return mode


@pytest.fixture(scope=scope, params=shapes_list)
def shapes(request):
    shapes = request.param
    return shapes


@pytest.fixture(scope=scope)
def expected1(shapes, mode):
    unpadded, window, padding, stride = shapes
    window_num, window_depth, window_height, window_width = window

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
    return (unpadded, window, padding, stride)


@pytest.fixture(scope=scope)
def expected2(expected1):
    unpadded, window, padding, stride = expected1
    unpadded_num, unpadded_depth, unpadded_height, unpadded_width = unpadded
    window_num, window_depth, window_height, window_width = window
    padding_height, padding_width = padding
    stride_height, stride_width = stride

    padded_height = unpadded_height + 2 * padding_height
    padded_width = unpadded_width + 2 * padding_width
    padded = (unpadded_num, unpadded_depth, padded_height, padded_width)

    output_height = ((padded_height - window_height) // stride_height) + 1
    output_width = ((padded_width - window_width) // stride_width) + 1
    output = (unpadded_num, window_num, output_height, output_width)

    im2col_height = window_depth * window_height * window_width
    im2col_width = output_height * output_width
    im2col_indices = (im2col_height, im2col_width)
    return (padded, output, im2col_indices)


@pytest.fixture(scope=scope)
def sweeper(shapes, mode):
    sweeper = Sweep2d(*shapes, mode)
    return sweeper


# Tests
class Test_Shape_Values:
    def test_unpadded(self, sweeper, expected1):
        unpadded, *_ = expected1
        assert sweeper.unpadded.shape == unpadded

    def test_window(self, sweeper, expected1):
        _, window, *_ = expected1
        assert sweeper.window.shape == window

    def test_padding(self, sweeper, expected1):
        *_, padding, _ = expected1
        assert sweeper.padding.shape == padding

    def test_stride(self, sweeper, expected1):
        *_, stride = expected1
        assert sweeper.stride.shape == stride

    def test_padded(self, sweeper, expected2):
        padded, *_ = expected2
        assert sweeper.padded.shape == padded

    def test_padded_array(self, sweeper, expected2):
        padded, *_ = expected2
        assert sweeper.padded_array.shape == padded

    def test_output(self, sweeper, expected2):
        _, output, _ = expected2
        assert sweeper.output.shape == output

    def test_im2col_indices(self, sweeper, expected2):
        *_, im2col_indices = expected2
        assert sweeper.im2col_indices.shape == im2col_indices
