import pytest
from dzlib.signal_processing.sweep2d import Sweep2d


# Test Inputs
unpaddeds = [(1, 1, 10, 10), (1, 3, 10, 10)]
windows = [(1, 1, 5, 5), (1, 3, 5, 5)]
paddings = [(0, 0), (0, 0)]
strides = [(1, 1), (1, 1)]
modes = ['user', 'user']

params = [inputs for inputs in zip(unpaddeds, windows, paddings, strides, modes)]

# Test Fixtures
@pytest.fixture(scope="class", params=params)
def sweeper(request):
    inputs = request.param
    sweeper = Sweep2d(*inputs)
    return sweeper


# Test Classes
class Test_Shape_Lengths():
    '''Test Shape Lengths'''
    def test_unpadded_length(self, sweeper):
        assert len(sweeper.unpadded.shape) == 4

    def test_window_length(self, sweeper):
        assert len(sweeper.window.shape) == 4

    def test_padding_length(self, sweeper):
        assert len(sweeper.padding.shape) == 3

    def test_stride_length(self, sweeper):
        assert len(sweeper.stride.shape) == 2

    def test_output_length(self, sweeper):
        assert len(sweeper.output.shape) == 4

class Test_Shape_Types():
    def test_unpadded_type(self, sweeper):
        assert isinstance(sweeper.unpadded.shape, tuple)

    def test_window_type(self, sweeper):
        assert isinstance(sweeper.window.shape, tuple)

    def test_padding_type(self, sweeper):
        assert isinstance(sweeper.padding.shape, tuple)

    def test_stride_type(self, sweeper):
        assert isinstance(sweeper.stride.shape, tuple)
