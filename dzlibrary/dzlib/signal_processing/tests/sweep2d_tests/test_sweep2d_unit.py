import pytest
from dzlib.signal_processing.sweep2d import Sweep2d
import numbers


# Test Inputs
unpaddeds = [(1, 1, 10, 10), (1, 3, 10, 10)]
windows = [(1, 1, 5, 5), (1, 3, 5, 5)]
paddings = [(0, 0), (0, 0)]
strides = [(1, 1), (1, 1)]
# modes = ['user', 'user']

params = [inputs for inputs in zip(unpaddeds, windows, paddings, strides)]
modes = ["user", "full", "same"]

# Test Fixtures
@pytest.fixture(scope="function", params=modes)
def mode(request):
    '''Fixture to iterate through each mode'''
    return request.param


@pytest.fixture(scope="function", params=params)
def sweeper(request, mode):
    '''Fixture to iterate through each set of inputs, including modes from mode fixture'''
    inputs = request.param
    return Sweep2d(*inputs, mode)


# Test Classes
class Test_Shape_Lengths:
    """Test Shape Lengths"""

    # ShapeNd objects
    def test_unpadded_length(self, sweeper):
        assert len(sweeper.unpadded.shape) == 4

    def test_window_length(self, sweeper):
        assert len(sweeper.window.shape) == 4

    def test_padding_length(self, sweeper):
        assert len(sweeper.padding.shape) == 2

    def test_stride_length(self, sweeper):
        assert len(sweeper.stride.shape) == 2

    def test_output_length(self, sweeper):
        assert len(sweeper.output.shape) == 4

    def test_padded_length(self, sweeper):
        assert len(sweeper.padded.shape) == 4

    # numpy.ndarray objects
    def test_padded_array_length(self, sweeper):
        assert len(sweeper.padded_array.shape) == 4

    def test_im2col_indices_length(self, sweeper):
        assert len(sweeper.im2col_indices.shape) == 2


class Test_Shape_Types:
    """Test Shape Types"""

    # ShapeNd objects
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
    """Test Shape Dimension Types"""

    # ShapeNd objects
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
    """Test Shape Dimension Min Values"""

    # ShapeNd objects
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

