import unittest
import numpy as np
from eval_sweep2d import generate_vs_torch, eval_vs_torch, generate_vs_scipy, eval_vs_scipy


class TestvsTorch(unittest.TestCase):
    def setUp(self):
        datas = generate_vs_torch(settings)
        self.data = iter(datas)
        self.n = n_tests
        self.precision = precision_vs_torch

    def tearDown(self):
        del self.data
        del self.n
        del self.precision

    def test_mode_user(self):
        mode = 'user'
        n = self.n
        precision = self.precision

        for i in range(n):
            x, k, p, s = next(self.data)
            seed = np.random.randint(0, int(1e3), 1)[0]
            out1, out2 = eval_vs_torch((x, k, p, s), mode, seed)

            with self.subTest(f"Test: {i+1}/{n}", seed = seed, x = x, k = k, p = p, s = s):
                np.testing.assert_almost_equal(out1, out2, decimal=precision, verbose=False)


class TestvsSciPy(unittest.TestCase):
    def setUp(self):
        datas = generate_vs_scipy(settings)
        self.data = iter(datas)
        self.n = n_tests
        self.precision = precision_vs_scipy

    def tearDown(self):
        del self.data
        del self.n
        del self.precision

    def test_mode_full(self):
        mode = 'full'
        n = self.n
        precision = self.precision

        for i in range(n):
            x, k, p, s = next(self.data)
            seed = np.random.randint(0, int(1e3), 1)[0]
            out1, out2 = eval_vs_scipy((x, k, p, s), mode, seed)

            with self.subTest(f"Test: {i+1}/{n}", seed = seed, x = x, k = k, p = p, s = s):
                np.testing.assert_almost_equal(out1, out2, decimal=precision, verbose=False)

    def test_mode_same(self):
        mode = 'same'
        n = self.n
        precision = self.precision

        for i in range(n):
            x, k, p, s = next(self.data)
            seed = np.random.randint(0, int(1e3), 1)[0]
            out1, out2 = eval_vs_scipy((x, k, p, s), mode, seed)

            with self.subTest(f"Test: {i+1}/{n}", seed = seed, x = x, k = k, p = p, s = s):
                np.testing.assert_almost_equal(out1, out2, decimal=precision, verbose=False)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", help="yaml settings file path", type=str)
    args = parser.parse_args()

    if args.filepath:
        settings_path = args.filepath

    else:
        settings_path = 'sweep_eval_settings.yml'

    with open(settings_path) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    n_tests = settings['Sweep2d']['n_tests']
    precision_vs_torch = settings['Sweep2d']['precisions']['vs_torch']
    precision_vs_scipy = settings['Sweep2d']['precisions']['vs_scipy']

    # Use this method instead of unittest.main() because it doesn't play nice with argparse
    runner = unittest.TextTestRunner()
    suite = unittest.TestSuite()
    test1 = unittest.TestLoader().loadTestsFromTestCase(TestvsTorch)
    test2 = unittest.TestLoader().loadTestsFromTestCase(TestvsSciPy)
    suite.addTests(test1)
    suite.addTests(test2)
    runner.run(suite)
