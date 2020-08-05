

class ShapeNd():
    def __init__(self, shape):
        self.shape = shape

    @property
    def shape(self):
        return tuple(self._shape)

    @shape.setter
    def shape(self, shape):
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape,)

        self._shape = list(shape)

    def __len__(self):
        return len(self.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self.shape)})"

    def __copy__(self):
        return self.__class__(self.shape)

    copy = __copy__


class Dimension():
    def __init__(self, n):
        self.n = n

    def __get__(self, obj, cls):
        if obj is None:
            return self
        try:
            return obj._shape[self.n]
        except IndexError:
            return None

    def __set__(self, obj, val):
        obj._shape[self.n] = val
