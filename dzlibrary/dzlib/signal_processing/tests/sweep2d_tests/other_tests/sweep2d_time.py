import numpy as np
import torch


def get_shapes(settings):
    tests = settings['Sweep2d']['Time'].values()

    numbers = [test['n_tests'] for test in tests]

    shapes = [test['input_arg_shapes'].values() for test in tests]
    shapes = [[tuple(x) for x in shape] for shape in shapes]
    return numbers, shapes


def generate_data(shapes):
    np_datas = []
    pt_datas = []

    for x, k, _, _ in shapes:
        np_images = np.random.randn(*x).astype(np.float32)
        np_kernels = np.random.randn(*k).astype(np.float32)

        pt_images = torch.from_numpy(np_images).type(torch.FloatTensor)
        pt_kernels = torch.from_numpy(np_kernels).type(torch.FloatTensor)

        np_datas.append((np_images, np_kernels))
        pt_datas.append((pt_images, pt_kernels))
    return np_datas, pt_datas
