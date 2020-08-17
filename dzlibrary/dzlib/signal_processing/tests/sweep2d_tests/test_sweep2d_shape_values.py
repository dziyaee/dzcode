
# All of the tests in this module are for data computed during Sweep2d class instatiation / initialization.


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
