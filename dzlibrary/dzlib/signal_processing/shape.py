

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


if __name__ == "__main__":
    # Example of subclassing ShapeNd and using the Dimension "helper" class as it's used in conjunction with Sweep2d class
    # from shape import ShapeNd, Dimension

    class Input4d(ShapeNd):
        def __init__(self, shape):
            super(Input4d, self).__init__(shape)

        width = Dimension(-1)
        height = Dimension(-2)
        depth = Dimension(-3)
        num = Dimension(-4)

    class Param2d(ShapeNd):
        def __init__(self, shape):
            super(Param2d, self).__init__(shape)

        width = Dimension(-1)
        height = Dimension(-2)


    images_shape = (1, 3, 10, 10)
    kernels_shape = (1, 3, 3, 3)
    padding = (2, 2)
    stride = (1, 1)

    images_shape = Input4d(images_shape)
    kernels_shape = Input4d(kernels_shape)
    padding = Param2d(padding)
    stride = Param2d(stride)

    print(images_shape)
    print(kernels_shape)
    print(padding)
    print(stride)
