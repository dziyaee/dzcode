from Sweep2d import Sweep2d
import numpy as np
import torch
import torch.nn.functional as F
import scipy.signal as sp


x = (5, 3, 100, 100)
k = (5, 3, 3, 3)
p = (1, 1)
s = (1, 1)
m = 'user'

images_np = np.random.randn(*x).astype(np.float32)
kernels_np = np.random.randn(*k).astype(np.float32)

images_pt = torch.from_numpy(images_np).type(torch.FloatTensor)
kernels_pt = torch.from_numpy(kernels_np).type(torch.FloatTensor)

sweep2d = Sweep2d(x, k, p, s, m)
sweep_outputs_np = sweep2d.correlate(images_np, kernels_np)

torch_outputs_pt = F.conv2d(images_pt, kernels_pt, bias=None, stride=s, padding=p)
torch_outputs_np = np.asarray(torch_outputs_pt).astype(np.float32)

errors = sweep_outputs_np - torch_outputs_np
mean_error = np.mean(np.abs(errors))
max_error = np.max(np.abs(errors))

print("Sweeper Corr2d vs Pytorch Conv2d:")
print(f"mean_error: {mean_error}")
print(f"max_error:  {max_error}")
print()

x = (1, 1, 100, 100)
k = (1, 1, 3, 3)
p = (0, 0)
s = (1, 1)
m = 'full'

images_np = np.random.randn(*x).astype(np.float32)
kernels_np = np.random.randn(*k).astype(np.float32)

sweep2d = Sweep2d(x, k, p, s, m)
sweep_outputs_np = sweep2d.correlate(images_np, kernels_np)[0, 0, :, :]

scipy_outputs_np = sp.correlate2d(images_np[0, 0, :, :], kernels_np[0, 0, :, :], mode='full')
errors = sweep_outputs_np - scipy_outputs_np
mean_error = np.mean(np.abs(errors))
max_error = np.max(np.abs(errors))

print("Sweeper Corr2d vs Scipy Corr2d:")
print(f"mean_error: {mean_error}")
print(f"max_error:  {max_error}")
print()

x = (1, 1, 100, 100)
k = (1, 1, 3, 3)
p = (0, 0)
s = (1, 1)

images_np = np.random.randn(*x).astype(np.float32)
kernels_np = np.random.randn(*k).astype(np.float32)

images_pt = torch.from_numpy(images_np).type(torch.FloatTensor)
kernels_pt = torch.from_numpy(kernels_np).type(torch.FloatTensor)

scipy_outputs_np = sp.correlate2d(images_np[0, 0, :, :], kernels_np[0, 0, :, :], mode='valid')

torch_outputs_pt = F.conv2d(images_pt, kernels_pt, bias=None, stride=s, padding=p)
torch_outputs_np = np.asarray(torch_outputs_pt).astype(np.float32)

errors = scipy_outputs_np - torch_outputs_np
mean_error = np.mean(np.abs(errors))
max_error = np.max(np.abs(errors))

print("Scipy Corr2d vs Pytorch Conv2d:")
print(f"mean_error: {mean_error}")
print(f"max_error:  {max_error}")
print()
