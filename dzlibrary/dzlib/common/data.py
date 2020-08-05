import numpy as np


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

