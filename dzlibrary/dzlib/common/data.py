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


class Shape:
    def __init__(self, shape):
        self.shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape_):
        # validations
        if not isinstance(shape_, (tuple, list)):
            raise TypeError(f"Expected tuple or list, got {type(shape_)}")

        if len(shape_) == 0:
            raise ValueError(f"Expected a tuple or list of length > 0, got {len(shape_)}")

        for x in shape_:
            if not isinstance(x, int):
                raise TypeError(f"elements in {shape_} must be of type (int), got {type(x)}")

            if not x > 0:
                raise ValueError(f"elements in {shape_} must be > 0, got {x}")

        self._shape = tuple(shape_)

    @property
    def ndim(self):
        return len(self.shape)

    def _get_dim(self, i):
        try:
            return self.shape[i]
        except IndexError:
            return None

    width = property(lambda self: self._get_dim(-1))
    height = property(lambda self: self._get_dim(-2))
    depth = property(lambda self: self._get_dim(-3))
    num = property(lambda self: self._get_dim(-4))


# # Alternate implementation of Shape Class using Dimension Class as Shape Class attributes
# class Dimension:
#     def __init__(self, n):
#         self.n = n

#     def __get__(self, obj, cls):
#         if obj is None:
#             return self
#         try:
#             return obj.shape[self.n]
#         except IndexError:
#             return None

#     def __set__(self, obj, value):
#         raise AttributeError("can't set attribute")


# class Shape:
#     width = Dimension(-1)
#     height = Dimension(-2)
#     depth = Dimension(-3)
#     num = Dimension(-4)

#     def __init__(self, shape):
#         self.shape = shape

#     @property
#     def shape(self):
#         return self._shape

#     @shape.setter
#     def shape(self, shape_):
#         if not isinstance(shape_, (tuple, list)):
#             raise TypeError(f"Expected tuple or list, got {type(shape_)}")

#         if len(shape_) == 0:
#             raise ValueError(f"Expected a tuple or list of length > 0, got {len(shape_)}")

#         if  not all(isinstance(dim, int) for dim in shape_):
#             raise TypeError(f"Expected only integer elements, got {shape_}")

#         if not all(dim > 0 for dim in shape_):
#             raise ValueError(f"Expected only positive integer elements, got {shape_}")

#         self._shape = tuple(shape_)

#     @property
#     def ndim(self):
#         return len(self.shape)
