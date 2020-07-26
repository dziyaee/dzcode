import numpy as np


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


class Dims():
    attrs = ['num', 'depth', 'height', 'width']

    def __init__(self, shape):
        attrs = self.attrs

        # ndim is represented by length of shape tuple
        ndim = min(len(attrs), len(shape))

        # set valid attributes from names list
        attrs = attrs[-ndim:]
        for attr, dim in zip(attrs, shape):
            setattr(self, attr, dim)

        self.shape = shape


class Array(np.ndarray):
    '''Numpy Subclass to add predefined dimension attributes that update dynamically'''
    attrs = ['num', 'depth', 'height', 'width'] # desired dimension attributes

    def __new__(cls, input_array, ndmin=0):
        obj = np.array(input_array, ndmin=ndmin).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        ndim = self.ndim
        attrs = self.attrs
        shape = self.shape

        ndim = min(ndim, len(attrs)) # get minimum between current shape and number of dimension attributes
        attrs = attrs[-ndim:]
        shape = shape[-ndim:]

        for attr, dim in zip(attrs, shape):
            setattr(self, attr, dim)
