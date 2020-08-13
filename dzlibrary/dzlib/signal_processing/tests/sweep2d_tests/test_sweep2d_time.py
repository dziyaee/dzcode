import time
import torch.nn.functional as F
from dzlib.signal_processing.sweep2d import Sweep2d
from dzlib.common.utils import timer


@timer
def torch2d(images_pt, kernels_pt, padding, stride):
    F.conv2d(input=images_pt, weight=kernels_pt, padding=padding, stride=stride)
    return None


@timer
def sweep2d(sweeper, images_np, kernels_np):
    sweeper.correlate2d(images_np, kernels_np)
    return None


def timing(n, datas):
    start_ = time.perf_counter()
    for i in range(n):
        number, (unpadded, window, padding, stride), (images_np, kernels_np), (images_pt, kernels_pt) = next(datas)

        sweeper = Sweep2d(unpadded, window, padding, stride, mode='user')

        start = time.perf_counter()
        time1 = sweep2d.timer(number, sweeper, images_np, kernels_np)
        time2 = torch2d.timer(number, images_pt, kernels_pt, padding, stride)
        stop = time.perf_counter()

        # time1 = time1 / number
        # time2 = time2 / number

        print()
        print(f"Test {i+1}/{n} finished in: {(stop - start):.0f} s")
        print(f"iters = {number}, unpadded = {unpadded}, window = {window}, padding = {padding}, stride = {stride}")
        print(f"Sweep2d: {time1:.7f} s")
        print(f"Torch2d: {time2:.7f} s")

    stop_ = time.perf_counter()
    print()
    print(f"Total Test Time: {(stop_ - start_):.0f} s")
    return None


if __name__ == "__main__":
    import yaml
    import argparse
    from sweep2d_time import get_shapes, generate_data

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", help="yaml settings file path", type=str)
    args = parser.parse_args()

    if args.filepath:
        settings_path = args.filepath

    else:
        settings_path = "settings.yml"

    with open(settings_path) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    numbers, shapes = get_shapes(settings)
    np_datas, pt_datas = generate_data(shapes)

    n = len(numbers)
    datas = iter(zip(numbers, shapes, np_datas, pt_datas))

    timing(n, datas)
