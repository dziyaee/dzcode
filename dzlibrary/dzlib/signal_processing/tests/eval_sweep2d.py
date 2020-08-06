import numpy as np
import torch
import torch.nn.functional as F
from dzlib.signal_processing.sweep2d import Sweep2d
import matplotlib.pyplot as plt


def gen_shape(*args):
    shape = tuple(np.random.randint(i, j, 1)[0].astype(np.int32) for x in args for i, j in x)
    return shape


def eval_user_mode(n_tests):
    dtype_np = np.float32
    dtype_pt = torch.FloatTensor

    # generate test data
    test_datas = []
    for i in range(n_tests):
        d = np.random.randint(1, 4)  # x depth must be equal to k depth
        xlims = [(1, 11), (d, d+1), (100, 501), (100, 501)]
        klims = [(1, 11), (d, d+1), (1, 6), (1, 6)]
        plims = [(0, 6), (0, 6)]
        slims = [(1, 4), (1, 4)]
        x, k, p, s = (gen_shape(lims) for lims in (xlims, klims, plims, slims))
        test_datas.append((x, k, p, s))

    mean_errors = []
    max_errors = []
    min_errors = []
    for test_data in test_datas:
        x, k, p, s = test_data

        images_np = np.random.randn(*x).astype(dtype_np)
        kernels_np = np.random.randn(*k).astype(dtype_np)

        images_pt = torch.from_numpy(images_np).type(dtype_pt)
        kernels_pt = torch.from_numpy(kernels_np).type(dtype_pt)

        sweep2d = Sweep2d(x, k, p, s, 'user')
        sweep_outputs_np = sweep2d.correlate2d(images_np, kernels_np)

        torch_outputs_pt = F.conv2d(images_pt, kernels_pt, bias=None, stride=s, padding=p)
        torch_outputs_np = np.asarray(torch_outputs_pt).astype(dtype_np)

        error = np.abs(sweep_outputs_np - torch_outputs_np)
        mean_errors.append(np.mean(error))

    mean_errors = np.asarray(mean_errors)
    return mean_errors


n = 100
mean_errors = eval_user_mode(n)
max_mean_error = np.max(mean_errors)

print(f"Eval User Mode")
print(f"Number of tests: {n}")
print(f"Max of Mean Errors: {max_mean_error}")

fig, ax1 = plt.subplots(ncols=1, nrows=1)
ax1.hist(mean_errors, bins=10)
ax1.set_title("Mean Errors")
ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))

plt.show()
