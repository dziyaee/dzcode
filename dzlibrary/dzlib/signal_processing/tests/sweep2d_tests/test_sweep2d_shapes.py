import numbers
import yaml
import pytest
from dzlib.signal_processing.sweep2d import Sweep2d
from dzlib.signal_processing.tests.sweep2d_tests.utils import generate_shape_test_params


# All of the tests in this module are for data computed during Sweep2d class instatiation / initialization.

# # Test Inputs
settings_path = 'settings.yml'
with open(settings_path) as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
inputs, modes = generate_shape_test_params(settings)

# Test Fixtures
scope = "module"  # scope must be shared by both fixtures as sweeper is dependent on mode


@pytest.fixture(scope=scope, params=modes)
def mode(request):
    '''Fixture to iterate through each mode'''
    return request.param


@pytest.fixture(scope=scope, params=inputs)
def sweeper(request, mode):
    '''Fixture to iterate through each set of inputs, including modes from mode fixture'''
    args = request.param
    return Sweep2d(*args, mode)


# Test Classes
class Test_Shape_Lengths:
    """
    Length = 2: Padding, Stride, im2col indices array
    Length = 4: Unpadded, Window, Output, Padded, Padded Array
    """

    def test_padding_length(self, sweeper):
        assert len(sweeper.padding.shape) == 2

    def test_stride_length(self, sweeper):
        assert len(sweeper.stride.shape) == 2

    def test_im2col_indices_length(self, sweeper):
        assert len(sweeper.im2col_indices.shape) == 2

    def test_unpadded_length(self, sweeper):
        assert len(sweeper.unpadded.shape) == 4

    def test_window_length(self, sweeper):
        assert len(sweeper.window.shape) == 4

    def test_output_length(self, sweeper):
        assert len(sweeper.output.shape) == 4

    def test_padded_length(self, sweeper):
        assert len(sweeper.padded.shape) == 4

    def test_padded_array_length(self, sweeper):
        assert len(sweeper.padded_array.shape) == 4


class Test_Shape_Types:
    '''
    Shape is instance of tuple: unpadded, window, padding, stride, output, padded
    '''

    def test_unpadded_type(self, sweeper):
        assert isinstance(sweeper.unpadded.shape, tuple)

    def test_window_type(self, sweeper):
        assert isinstance(sweeper.window.shape, tuple)

    def test_padding_type(self, sweeper):
        assert isinstance(sweeper.padding.shape, tuple)

    def test_stride_type(self, sweeper):
        assert isinstance(sweeper.stride.shape, tuple)

    def test_output_type(self, sweeper):
        assert isinstance(sweeper.output.shape, tuple)

    def test_padded_type(self, sweeper):
        assert isinstance(sweeper.padded.shape, tuple)


class Test_Shape_Dim_Types:
    """
    Element is instance of numbers.Integral: unpadded, window, padding (if mode is "user" or "full"), stride
    Element is instance of numbers.Real: padding (if mode is "same")
    """

    def test_unpadded_dim_type(self, sweeper):
        for dim in sweeper.unpadded.shape:
            assert isinstance(dim, numbers.Integral)

    def test_window_dim_type(self, sweeper):
        for dim in sweeper.window.shape:
            assert isinstance(dim, numbers.Integral)

    def test_padding_dim_type(self, sweeper):
        if sweeper.mode == "same":
            for dim in sweeper.padding.shape:
                assert isinstance(dim, numbers.Real)

        else:
            for dim in sweeper.padding.shape:
                assert isinstance(dim, numbers.Integral)

    def test_stride_dim_type(self, sweeper):
        for dim in sweeper.stride.shape:
            assert isinstance(dim, numbers.Integral)


class Test_Shape_Dim_Min_Values:
    '''
    Element >= 1: unpadded, window, stride (if mode is "user")
    Element = 1: stride (if mode is "full" or "same")
    Element >= 0: padding
    '''

    def test_unpadded_dim_min_value(self, sweeper):
        for dim in sweeper.unpadded.shape:
            assert dim >= 1

    def test_window_dim_min_value(self, sweeper):
        for dim in sweeper.window.shape:
            assert dim >= 1

    def test_padding_dim_min_value(self, sweeper):
        for dim in sweeper.padding.shape:
            assert dim >= 0

    def test_stride_dim_min_value(self, sweeper):
        if sweeper.mode == "user":
            for dim in sweeper.stride.shape:
                assert dim >= 1

        else:
            for dim in sweeper.stride.shape:
                assert dim == 1
