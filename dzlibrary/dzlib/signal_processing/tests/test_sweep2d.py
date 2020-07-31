import unittest
import numpy as np
from dzlib.signal_processing.sweep2d import Sweep2d
from dzlib.common.data import Shape


class Test__totuple(unittest.TestCase):
    # _totuple() is a static method that converts an input into a tuple of integers and validates the max length of the tuple and min value of the integers
    def setUp(self):
        self.sweep = Sweep2d

    def tearDown(self):
        del self.sweep

    def test__totuple_valid_types(self):
        sweep = self.sweep

        # integer input, output should be tuple of integer
        self.assertEqual(sweep._totuple(1, 1, 1), (1,))
        self.assertEqual(sweep._totuple(1, 0, 1), (1,))
        self.assertEqual(sweep._totuple(0, 1, 0), (0,))
        self.assertEqual(sweep._totuple(-1, 1, -2), (-1,))

        # tuple input, output should be equivalent tuple
        self.assertEqual(sweep._totuple((1,), 0, 1), (1,))
        self.assertEqual(sweep._totuple((1,), 1, 1), (1,))
        self.assertEqual(sweep._totuple((1, 2), 2, 1), (1, 2))
        self.assertEqual(sweep._totuple((0, -1, 1), 3, -1), (0, -1, 1))

        # list input, output should be equivalent but tuple
        self.assertEqual(sweep._totuple([1], 0, 1), (1,))
        self.assertEqual(sweep._totuple([1], 1, 1), (1,))
        self.assertEqual(sweep._totuple([1, 2], 2, 1), (1, 2))
        self.assertEqual(sweep._totuple([0, -1, 1], 3, -1), (0, -1, 1))

    def test__totuple_invalid_types(self):
        sweep = self.sweep

        # invalid input types
        self.assertRaises(TypeError, sweep._totuple, 1., 1, 0)
        self.assertRaises(TypeError, sweep._totuple, 1+1j, 1, 0)
        self.assertRaises(TypeError, sweep._totuple, {1}, 1, 0)
        self.assertRaises(TypeError, sweep._totuple, {1: 1}, 1, 0)
        self.assertRaises(TypeError, sweep._totuple, np.array([1]), 1, 0)

        # invalid element types
        self.assertRaises(TypeError, sweep._totuple, (1.,), 1, 0)
        self.assertRaises(TypeError, sweep._totuple, (1, 2, 3j), 3, 0)
        self.assertRaises(TypeError, sweep._totuple, (1, 2., 3), 3, 0)

    def test__totuple_invalid_lens(self):
        sweep = self.sweep

        # len(input) must be <= max len
        self.assertRaises(ValueError, sweep._totuple, (1, 2), 1, 0)
        self.assertRaises(ValueError, sweep._totuple, (1, 2, 3), 2, 0)

    def test__totuple_invalid_vals(self):
        sweep = self.sweep

        # elements must be >= min val
        self.assertRaises(ValueError, sweep._totuple, 1, 2, 2)
        self.assertRaises(ValueError, sweep._totuple, (1, 2), 2, 2)
        self.assertRaises(ValueError, sweep._totuple, (0, 1), 2, 1)
        self.assertRaises(ValueError, sweep._totuple, (-2, 1), 2, -1)


class Test__expandtuple(unittest.TestCase):
    # _expand() is a static method that expands a tuple to a specified length with extra elements as specified fill values
    def setUp(self):
        self.sweep = Sweep2d

    def tearDown(self):
        del self.sweep

    def test__expandtuple_valid_types(self):
        sweep = self.sweep

        # tuple input with len <= size
        self.assertEqual(sweep._expandtuple((1,), 2, 0), (0, 1))
        self.assertEqual(sweep._expandtuple((1, 2), 2, 0), (1, 2))
        self.assertEqual(sweep._expandtuple((3, 2), 4, 1), (1, 1, 3, 2))

        # tuple input with len > size
        self.assertEqual(sweep._expandtuple((1, 2, 3), 2, 0), (2, 3))
        self.assertEqual(sweep._expandtuple([1, 2], 1, 0), (2,))

        # output should always be tuple
        self.assertIsInstance(sweep._expandtuple((1, 2), 2, 2), tuple)
        self.assertIsInstance(sweep._expandtuple([1, 2], 2, 2), tuple)

    def test__expandtuple_invalid_types(self):
        sweep = self.sweep

        # first input must be tuple or list
        self.assertRaises(TypeError, sweep._expandtuple, 2, 1, 0)
        self.assertRaises(TypeError, sweep._expandtuple, 2., 1, 0)
        self.assertRaises(TypeError, sweep._expandtuple, '2', 1, 0)

        # second input must be int
        self.assertRaises(TypeError, sweep._expandtuple, (1, 2), 2., 0)
        self.assertRaises(TypeError, sweep._expandtuple, (1, 2), '2', 0)
        self.assertRaises(TypeError, sweep._expandtuple, (1, 2), [2], 0)

class Test__mode(unittest.TestCase):
    def setUp(self):
        self.sweep = Sweep2d
        self.xx = Shape((1, 3, 10, 10))
        self.kk = Shape((1, 3, 3, 3))
        self.padding = (0, 0)
        self.stride = (1, 1)

    def tearDown(self):
        del self.sweep

    def test__mode_invalid_inputs(self):
        xx, kk = self.xx, self.kk
        padding, stride = self.padding, self.stride
        sweep = Sweep2d

        # mode must be in modes
        self.assertRaises(ValueError, sweep._mode, xx, kk, padding, stride, 'use')
        self.assertRaises(ValueError, sweep._mode, xx, kk, padding, stride, '')
        self.assertRaises(ValueError, sweep._mode, xx, kk, padding, stride, 1)
        self.assertRaises(ValueError, sweep._mode, xx, kk, padding, stride, 1.)
        self.assertRaises(ValueError, sweep._mode, xx, kk, padding, stride, ['user'])
        self.assertRaises(ValueError, sweep._mode, xx, kk, padding, stride, {'user': 'user'})




class Test_Sweep2d(unittest.TestCase):
    def setUp(self):
        self.sweep = Sweep2d

    def tearDown(self):
        del self.sweep

    def test_dim_mismatch(self):
        sweep = self.sweep

        # image and kernel depths must match
        self.assertRaises(ValueError, sweep, (3, 10, 10), (2, 3, 3))
        self.assertRaises(ValueError, sweep, (1, 3, 10, 10), (3, 4, 3, 3))



if __name__ == "__main__":
    unittest.main()
