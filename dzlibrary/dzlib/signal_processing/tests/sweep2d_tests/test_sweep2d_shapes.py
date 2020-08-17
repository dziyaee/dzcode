import numbers
import pytest


# All of the tests in this module are for data computed during Sweep2d class instatiation / initialization.

# Test Classes
class Test_ShapeNd_Lengths:
    """ Tests all object instances of ShapeNd for correct lengths of their .shape attributes (indirectly testing the ndim).
    Length = 2: Padding, Stride
    Length = 4: Unpadded, Window, Output, Padded
    """

    def test_padding_length(self, sweeper):
        assert len(sweeper.padding.shape) == 2

    def test_stride_length(self, sweeper):
        assert len(sweeper.stride.shape) == 2

    def test_unpadded_length(self, sweeper):
        assert len(sweeper.unpadded.shape) == 4

    def test_window_length(self, sweeper):
        assert len(sweeper.window.shape) == 4

    def test_output_length(self, sweeper):
        assert len(sweeper.output.shape) == 4

    def test_padded_length(self, sweeper):
        assert len(sweeper.padded.shape) == 4


class Test_Numpy_NdArray_Shape_Lengths:
    """ Tests all object instances of numpy.ndarray for correct lengths of their .shape attributes (indirectly testing the ndim).
    Length = 2: Padding, Stride
    Length = 4: Unpadded, Window, Output, Padded
    """

    def test_padded_array_length(self, sweeper):
        assert len(sweeper.padded_array.shape) == 4

    def test_im2col_indices_length(self, sweeper):
        assert len(sweeper.im2col_indices.shape) == 2


class Test_ShapeNd_Types:
    ''' Tests all object instances of ShapeNd objects for correct .shape type
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


class Test_ShapeNd_Dim_Types:
    """ Tests all object instances of ShapeNd objects .shape attribute elements for correct type
    Element is instance of numbers.Integral: unpadded, window, padding (if mode is "user" or "full"), stride
    Element is instance of numbers.Real: padding (if mode is "same")
    """
    @pytest.mark.parametrize("sweeper_shape_dims", ["num", "depth", "height", "width"], indirect=True)
    def test_unpadded_dim_type(self, sweeper_shape_dims):
        unpadded_dim, *_ = sweeper_shape_dims
        assert isinstance(unpadded_dim, numbers.Integral)

    @pytest.mark.parametrize("sweeper_shape_dims", ["num", "depth", "height", "width"], indirect=True)
    def test_window_dim_type(self, sweeper_shape_dims):
        _, window_dim, *_ = sweeper_shape_dims
        assert isinstance(window_dim, numbers.Integral)

    @pytest.mark.parametrize("sweeper_shape_dims", ["height", "width"], indirect=True)
    def test_padding_dim_type(self, sweeper_shape_dims, mode):
        _, _, padding_dim, *_ = sweeper_shape_dims
        if mode == "same":
            assert isinstance(padding_dim, numbers.Real)

        else:
            assert isinstance(padding_dim, numbers.Integral)

    @pytest.mark.parametrize("sweeper_shape_dims", ["height", "width"], indirect=True)
    def test_stride_dim_type(self, sweeper_shape_dims):
        *_, stride_dim, _, _ = sweeper_shape_dims
        assert isinstance(stride_dim, numbers.Integral)

    @pytest.mark.parametrize("sweeper_shape_dims", ["num", "depth", "height", "width"], indirect=True)
    def test_padded_dim_type(self, sweeper_shape_dims):
        *_, padded_dim, _ = sweeper_shape_dims
        assert isinstance(padded_dim, numbers.Integral)

    @pytest.mark.parametrize("sweeper_shape_dims", ["num", "depth", "height", "width"], indirect=True)
    def test_output_dim_type(self, sweeper_shape_dims):
        *_, output_dim = sweeper_shape_dims
        assert isinstance(output_dim, numbers.Integral)


class Test_Shape_Dim_Min_Values:
    """ Tests all object instances of ShapeNd objects .shape attribute elements for greater than or equal to min value

    Element >= 1: unpadded, window, stride (if mode is "user")
    Element = 1: stride (if mode is "full" or "same")
    Element >= 0: padding
    """

    @pytest.mark.parametrize("sweeper_shape_dims", ["num", "depth", "height", "width"], indirect=True)
    def test_unpadded_dim_min_value(self, sweeper_shape_dims):
        unpadded_dim, *_ = sweeper_shape_dims
        assert unpadded_dim >= 1

    @pytest.mark.parametrize("sweeper_shape_dims", ["num", "depth", "height", "width"], indirect=True)
    def test_window_dim_min_value(self, sweeper_shape_dims):
        _, window_dim, *_ = sweeper_shape_dims
        assert window_dim >= 1

    @pytest.mark.parametrize("sweeper_shape_dims", ["height", "width"], indirect=True)
    def test_padding_dim_min_value(self, sweeper_shape_dims):
        _, _, padding_dim, *_ = sweeper_shape_dims
        assert padding_dim >= 0

    @pytest.mark.parametrize("sweeper_shape_dims", ["height", "width"], indirect=True)
    def test_stride_dim_min_value(self, sweeper_shape_dims, mode):
        *_, stride_dim, _, _ = sweeper_shape_dims
        if mode == "user":
            assert stride_dim >= 1

        else:
            assert stride_dim == 1

    @pytest.mark.parametrize("sweeper_shape_dims", ["num", "depth", "height", "width"], indirect=True)
    def test_padded_dim_min_value(self, sweeper_shape_dims):
        *_, padded_dim, _ = sweeper_shape_dims
        assert padded_dim >= 1

    @pytest.mark.parametrize("sweeper_shape_dims", ["num", "depth", "height", "width"], indirect=True)
    def test_output_dim_min_value(self, sweeper_shape_dims):
        *_, output_dim = sweeper_shape_dims
        assert output_dim >= 1
