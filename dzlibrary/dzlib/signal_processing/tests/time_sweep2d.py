import timeit
from dzlib.common.utils import timer, init_logger
from dzlib.signal_processing.sweep2d import Sweep2d
import numpy as np
import logging
logger = init_logger(filename=f'{Sweep2d.__name__}.log',
                     loggername=Sweep2d.__name__,
                     loggerlevel='INFO',
                     format='%(asctime)s %(name)s:: %(message)s',
                     dateformat='%Y-%m-%d %H:%M:%S')


# Inputs
images_shape = (10, 5, 500, 500)
kernels_shape = (50, 5, 5, 5)

padding = (1, 1)
stride = (1, 1)
repeat = 1
number = 10
test_names = ['Pytorch Conv2d', 'Sweep2d Corr2d w/o init', 'Sweep2d Corr2d w/ init']

# Timing
# Setups and Statements for timeit
setup_common = f'''
import numpy as np

# inputs
images_shape = {images_shape}
kernels_shape = {kernels_shape}
padding = {padding}
stride = {stride}
mode = "user"

# init
images = np.random.randn(*images_shape)
kernels = np.random.randn(*kernels_shape)

images_np = np.array(images, ndmin=4).astype(np.float32)
kernels_np = np.array(kernels, ndmin=4).astype(np.float32)
'''

stmt_common = f'''
'''

setup_torch = setup_common + '''
import torch
import torch.nn.functional as F
images_pt = torch.from_numpy(images_np).type(torch.FloatTensor)
kernels_pt = torch.from_numpy(kernels_np).type(torch.FloatTensor)
def time_torch_conv2d():
    F.conv2d(input=images_pt, weight=kernels_pt, padding=padding, stride=stride)
    return None
'''

stmt_torch = stmt_common + '''
time_torch_conv2d()
'''

setup_without = setup_common + '''
from dzlib.signal_processing.sweep2d import Sweep2d
sweeper_without = Sweep2d(images_np.shape, kernels_np.shape, padding, stride, mode)
def time_sweep2d_without_init():
    sweeper_without.correlate2d(images_np, kernels_np)
    return None
'''

stmt_without = stmt_common + '''
time_sweep2d_without_init()
'''

setup_with = setup_common + '''
from dzlib.signal_processing.sweep2d import Sweep2d
def time_sweep2d_with_init():
    sweeper_with = Sweep2d(images_np.shape, kernels_np.shape, padding, stride, mode)
    sweeper_with.correlate2d(images_np, kernels_np)
    return None
'''

stmt_with = stmt_common + '''
time_sweep2d_with_init()
'''

@timer
def timeit_tests(setups, stmts, repeat, number):
    '''Function to run a timeit test for each setup / stmt pair and return average times per test. @timer decorator to print total time'''
    avg_times = []
    for stmt, setup in zip(stmts, setups):
        times = timeit.repeat(stmt=stmt, setup=setup, repeat=repeat, number=number)
        times = np.mean(np.asarray(times))
        avg_time = times / number
        avg_times.append(avg_time)
    return avg_times


# run tests
setups = [setup_torch, setup_without, setup_with]
stmts = [stmt_torch, stmt_without, stmt_with]
avg_times = timeit_tests(setups=setups, stmts=stmts, repeat=repeat, number=number)

# log test results
message1 = f"timeit args: repeat: {repeat}, number: {number}"
message2 = f"{Sweep2d.__name__} args: images_shape: {images_shape}, kernels_shape: {kernels_shape}, padding: {padding}, stride: {stride}"
print(message1)
print(message2)
logger.info(message1)
logger.info(message2)

longest_name = np.max(np.asarray([len(name) for name in test_names]))
for avg_time, test_name in zip(avg_times, test_names):
    spaces = ' ' * (longest_name - len(test_name))
    message1 = f"{test_name}:{spaces} {(avg_time * 1e3):.3f} ms"
    if test_name == test_names[-1]:
        message1 += "\n"

    print(message1)
    logger.info(message1)
