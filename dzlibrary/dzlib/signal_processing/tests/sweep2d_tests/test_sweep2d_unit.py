import unittest
from dzlib.signal_processing.sweep2d import Sweep2d
import numbers


class Shape_Test_Templates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Inputs
        cls.unpadded_input = (1, 1, 10, 10)
        cls.window_input = (1, 1, 3, 3)
        cls.padding_input = (0, 0)
        cls.stride_input = (1, 1)
        cls.mode_input = 'user'

        # Instantiate & Init Sweep2d Class
        cls.sweep = Sweep2d(cls.unpadded_input, cls.window_input, cls.padding_input, cls.stride_input, cls.mode_input)

        # Init test input / shape
        cls.test_input = None
        cls.test_shape = None
        pass

    @classmethod
    def tearDownClass(cls):
        del cls.unpadded_input
        del cls.window_input
        del cls.padding_input
        del cls.stride_input
        del cls.mode_input
        del cls.sweep
        del cls.test_input
        del cls.test_shape
        pass

    def assert_shape_length(self, shape, length):
        self.assertEqual(len(shape), length)

    def assert_shape_type(self, shape, type_):
        self.assertIsInstance(shape, type_)

    def assert_shape_match_input(self, shape, input_):
        self.assertEqual(shape, input_)

    def assert_shape_dims_type(self, shape, type_):
        for i, dim in enumerate(shape):
            with self.subTest(shape=shape, element=dim, position=i, type=type_):
                self.assertIsInstance(dim, type_)

    def assert_shape_dims_min_value(self, shape, min_value):
        for i, dim in enumerate(shape):
            with self.subTest(shape=shape, element=dim, position=i, min_value=min_value):
                self.assertGreaterEqual(dim, min_value)

    def assert_shape_dims_multiple_of_value(self, shape, value):
        for i, dim in enumerate(shape):
            with self.subTest(shape=shape, element=dim, position=i, multiple=value):
                self.assertEqual(dim % value, 0)


class Test_Unpadded_Shape(Shape_Test_Templates):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_input = cls.unpadded_input
        cls.test_shape = cls.sweep.unpadded.shape
        pass

    def test_unpadded_type(self):
        super().assert_shape_type(self.test_shape, tuple)

    def test_unpadded_length(self):
        super().assert_shape_length(self.test_shape, 4)

    def test_unpadded_dims_type(self):
        super().assert_shape_dims_type(self.test_shape, numbers.Integral)

    def test_unpadded_dims_min_value(self):
        super().assert_shape_dims_min_value(self.test_shape, 1)

    def test_unpadded_match_input(self):
        super().assert_shape_match_input(self.test_shape, self.test_input)


class Test_Window_Shape(Shape_Test_Templates):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_input = cls.window_input
        cls.test_shape = cls.sweep.window.shape
        pass

    def test_window_type(self):
        super().assert_shape_type(self.test_shape, tuple)

    def test_window_length(self):
        super().assert_shape_length(self.test_shape, 4)

    def test_window_dims_type(self):
        super().assert_shape_dims_type(self.test_shape, numbers.Integral)

    def test_window_dims_min_value(self):
        super().assert_shape_dims_min_value(self.test_shape, 1)

    def test_window_match_input(self):
        super().assert_shape_match_input(self.test_shape, self.test_input)


class Test_Padding_Shape(Shape_Test_Templates):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_input = cls.padding_input
        cls.test_shape = cls.sweep.padding.shape
        pass

    def test_padding_type(self):
        super().assert_shape_type(self.test_shape, tuple)

    def test_padding_length(self):
        super().assert_shape_length(self.test_shape, 2)

    def test_padding_dims_type(self):
        super().assert_shape_dims_type(self.test_shape, numbers.Real)

    def test_padding_dims_min_value(self):
        super().assert_shape_dims_min_value(self.test_shape, 0)

    def test_padding_dims_multiple_of_value(self):
        super().assert_shape_dims_multiple_of_value(self.test_shape, 1)

    def test_padding_match_input(self):
        super().assert_shape_match_input(self.test_shape, self.test_input)


class Test_Stride_Shape(Shape_Test_Templates):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_input = cls.stride_input
        cls.test_shape = cls.sweep.stride.shape
        pass

    def test_stride_type(self):
        super().assert_shape_type(self.test_shape, tuple)

    def test_stride_length(self):
        super().assert_shape_length(self.test_shape, 2)

    def test_stride_dims_type(self):
        super().assert_shape_dims_type(self.test_shape, numbers.Integral)

    def test_stride_dims_min_value(self):
        super().assert_shape_dims_min_value(self.test_shape, 1)

    def test_stride_match_input(self):
        super().assert_shape_match_input(self.test_shape, self.test_input)


if __name__ == "__main__":
    try:
        # Create Test Suites for each Test Case
        testloader = unittest.TestLoader()
        suite1 = testloader.loadTestsFromTestCase(Test_Unpadded_Shape)
        suite2 = testloader.loadTestsFromTestCase(Test_Window_Shape)
        suite3 = testloader.loadTestsFromTestCase(Test_Padding_Shape)
        suite4 = testloader.loadTestsFromTestCase(Test_Stride_Shape)

        # Load all Test Suites into one
        suite = unittest.TestSuite()
        suite.addTest(suite1)
        suite.addTest(suite2)
        suite.addTest(suite3)
        suite.addTest(suite4)

    except NameError:
        print(f"Not all given tests are defined")

    else:
        # Run all Test Suites together
        runner = unittest.TextTestRunner()
        runner.run(suite)
