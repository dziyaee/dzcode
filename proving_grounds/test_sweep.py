import unittest
from dzlib.signal_processing.sweep import Sweep1d, Sweep2d

class TestSweep1d(unittest.TestCase):
    def test_valid_padding_values(self):
        ps = [0, 1]
        outs = [(0,), (1,)]
        for p, out in zip(ps, outs):
            sweep = Sweep1d(5, 3, p, 1)
            with self.subTest(padding=p):
                self.assertEqual(sweep.padding, out)

    def test_invalid_padding(self):
        self.assertRaises(TypeError, Sweep1d, 5, 3, 0., 1)
        self.assertRaises(TypeError, Sweep1d, 5, 3, 1., 1)
        self.assertRaises(TypeError, Sweep1d, 5, 3, 1j, 1)
        self.assertRaises(TypeError, Sweep1d, 5, 3, '1', 1)
        self.assertRaises(TypeError, Sweep1d, 5, 3, (1.,), 1)
        self.assertRaises(ValueError, Sweep1d, 5, 3, (1, 1), 1)
        self.assertRaises(ValueError, Sweep1d, 5, 3, -1, 1)
    pass

class TestSweep2d(unittest.TestCase):
    def test_valid_padding_values(self):
        # sweep = Sweep2d()
        pass

    pass
if __name__ == "__main__":
    unittest.main()
