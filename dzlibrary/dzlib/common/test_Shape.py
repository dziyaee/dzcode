import unittest
from dzlib.common.data import Shape


class TestShape(unittest.TestCase):
    '''Unit Tests for Shape Class'''
    def setUp(self):
        self.withtuple = Shape((1, 2, 3))
        self.withlist = Shape([1, 2, 3])
        self.inputs = [self.withtuple, self.withlist]

    def tearDown(self):
        del self.withtuple
        del self.withlist
        del self.inputs

    # Test setting of shape attribute at Shape class instantiation
    def test_shape(self):
        # input must be tuple or list
        self.assertRaises(TypeError, Shape, None)
        self.assertRaises(TypeError, Shape, 1)
        self.assertRaises(TypeError, Shape, 1.)
        self.assertRaises(TypeError, Shape, {(0, 1): (2, 3)})
        self.assertRaises(TypeError, Shape, {1, 2, 3})

        # length of input must be > 0
        self.assertRaises(ValueError, Shape, ())
        self.assertRaises(ValueError, Shape, [])

        # all elements in input iterable must be integers
        self.assertRaises(TypeError, Shape, (1, 2, 3.))
        self.assertRaises(TypeError, Shape, (1, 2, 3j))
        self.assertRaises(TypeError, Shape, (1, 2, (3, 4)))
        self.assertRaises(TypeError, Shape, [1, 2, (3, 4)])

        # all elements must be positive integers > 0
        self.assertRaises(ValueError, Shape, (1, 2, -3))
        self.assertRaises(ValueError, Shape, (1, 0, 3))

    # Test Shape attribute types after instantiation
    def test_Shape_attr_types(self):
        for x in self.inputs:
            self.assertIsInstance(x.shape, tuple)
            self.assertIsInstance(x.ndim, int)
            self.assertIsInstance(x.width, int)
            self.assertIsInstance(x.height, int)
            self.assertIsInstance(x.depth, int)
            self.assertIs(x.num, None)

    # Test Shape attribute values after instantiation
    def test_Shape_attr_values(self):
        for x in self.inputs:
            self.assertEqual(x.shape, (1, 2, 3))
            self.assertEqual(x.ndim, 3)
            self.assertEqual(x.width, 3)
            self.assertEqual(x.height, 2)
            self.assertEqual(x.depth, 1)
            self.assertEqual(x.num, None)

    # Test setting of .shape attribute after Shape class instantiation
    def test_shape_set(self):
        x, y = self.inputs
        x.shape = (2, 3)
        y.shape = [3, 4]
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(y.shape, (3, 4))

    # Shape attributes other than .shape must not be settable
    def test_invalid_set(self):
        x, _ = self.inputs
        with self.assertRaises(AttributeError): x.ndim = 4
        with self.assertRaises(AttributeError): x.width = 4
        with self.assertRaises(AttributeError): x.height = 4
        with self.assertRaises(AttributeError): x.depth = 4
        with self.assertRaises(AttributeError): x.num = 4


if __name__ == "__main__":
    unittest.main()
