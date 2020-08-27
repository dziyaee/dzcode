import argparse
import unittest

import numpy as np
import yaml

from sweep2d_eval import (
    eval_vs_scipy,
    eval_vs_torch,
    generate_vs_scipy,
    generate_vs_torch,
)


class ParametrizedTestCase(unittest.TestCase):
    def __init__(self, methodName='runTest', settings=None, generator=None, evaluator=None):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.settings = settings
        self.generator = generator
        self.evaluator = evaluator

    @staticmethod
    def parametrize(testcase_class, settings=None, generator=None, evaluator=None):
        testloader = unittest.TestLoader()
        test_names = testloader.getTestCaseNames(testcase_class)
        suite = unittest.TestSuite()
        for test_name in test_names:
            suite.addTest(testcase_class(test_name, settings, generator, evaluator))
        return suite


class TestCaseCommon(ParametrizedTestCase):
    def setUp(self):
        self.n_tests = self.settings['n_tests']
        self.precision = self.settings['precision']
        self.datas = self.generator(self.settings)

    def tearDown(self):
        del self.n_tests
        del self.precision
        del self.datas

    def eval_with_mode(self, mode):
        n = self.n_tests
        precision = self.precision
        data = zip(*self.datas)

        for i in range(n):
            x, k, p, s = next(data)
            seed = np.random.randint(0, int(1e3), 1)[0]
            out1, out2 = self.evaluator((x, k, p, s), mode, seed)

            with self.subTest(f"Test: {i+1}/{n}", seed=seed, x=x, k=k, p=p, s=s, mode=mode, precision=precision):
                np.testing.assert_almost_equal(out1, out2, decimal=precision, verbose=False)


class TestvsTorch(TestCaseCommon):
    def test_mode_user(self):
        mode = 'user'  # vs_torch only works with mode 'user'
        super().eval_with_mode(mode)
        pass


class TestvsScipy(TestCaseCommon):
    def test_mode_full(self):
        mode = 'full'
        super().eval_with_mode(mode)
        pass

    def test_mode_same(self):
        mode = 'same'
        super().eval_with_mode(mode)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", help="file path to settings.yml", type=str)
    args = parser.parse_args()

    if args.filepath:
        settings_path = args.filepath

    else:
        settings_path = "settings.yml"

    with open(settings_path) as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    testcase1 = ParametrizedTestCase.parametrize(TestvsTorch, settings['Sweep2d']['Eval']['vs_torch'], generate_vs_torch, eval_vs_torch)
    testcase2 = ParametrizedTestCase.parametrize(TestvsScipy, settings['Sweep2d']['Eval']['vs_scipy'], generate_vs_scipy, eval_vs_scipy)

    suite = unittest.TestSuite()
    suite.addTest(testcase1)
    suite.addTest(testcase2)

    runner = unittest.TextTestRunner()
    runner.run(suite)
