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
