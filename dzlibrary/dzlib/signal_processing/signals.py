import numpy as np
from collections import namedtuple


class Signal():
    def __init__(self, array):
        self.set_data(array)

    def set_data(self, array):
        self.arr = array
        self._update()

    def _update(self):
        self.real = self.arr.real
        self.imag = self.arr.imag
        self.mag = np.abs(self.arr)
        self.phase = np.angle(self.arr)
        self.min = np.real_if_close(np.min(self.arr))
        self.max = np.real_if_close(np.max(self.arr))
        self.mean = np.real_if_close(np.mean(self.arr))
        self.std = np.std(self.arr)
        self.var = np.var(self.arr)





















# def signal(x):
#     Signal = namedtuple('Signal', ['arr', 'real', 'imag', 'mag', 'phase', 'min', 'max', 'mean', 'std', 'var'])
#     y = Signal(arr=x, real=x.real, imag=x.imag, mag=np.abs(x), phase=np.angle(x), min=np.min(x), max=np.max(x), mean=np.mean(x), std=np.std(x), var=np.var(x))
#     return y

# def printsignal(signal):
#     for field, data in zip(signal._fields, signal):
#         print(f"{field:5}: {data}")
#     return None


# if __name__ == '__main__':
#     x = np.array([1, 2, 3, 3, 2, 1])
#     x = signal(x)
#     printsignal(x)

#     y = np.array([-1, -2, -3, 3, 2, 1])
#     y = signal(y)
#     printsignal(y)
