

class ShapeNd():
    def __init__(self, shape):
        self.shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if len(shape) != self.ndim:
            raise ValueError(f"{self.__class__.__name__} only accepts tuples of {self.ndim} numbers, got {shape}")
        self._shape = list(shape)

    def _get_dim(self, i):
        return self._shape[i]

    def _set_dim(self, i, val):
        self._shape[i] = val

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self.shape)})"

    def __add__(self, other):
        try:
            i = min(self.ndim, other.ndim)
            d = max(self.ndim, other.ndim) - i
        except AttributeError:
            new = tuple(i + other for i in self.shape)
            return self.__class__(new)
        else:
            if not self.ndim >= other.ndim:
                raise ValueError(f"Unable to add lower to higher ndim shapes: {self.ndim}, {other.ndim}")
            new = tuple(i + j for i, j in zip(self.shape[-i:], other.shape[-i:]))
            return self.__class__((*self.shape[:d], *new))

    __radd__ = __add__

    def __sub__(self, other):
        try:
            i = min(self.ndim, other.ndim)
            d = max(self.ndim, other.ndim) - i
        except AttributeError:
            new = tuple(i - other for i in self.shape)
            return self.__class__(new)
        else:
            if not self.ndim >= other.ndim:
                raise TypeError(f"Cannot subtract {other} from {self}")
            new = tuple(i - j for i, j in zip(self.shape[-i:], other.shape[-i:]))
            return self.__class__((*self.shape[:d], *new))

    __rsub__ = __sub__

    def __mul__(self, other):
        try:
            i = min(self.ndim, other.ndim)
            d = max(self.ndim, other.ndim) - i
        except AttributeError:
            new = tuple(i * other for i in self.shape)
            return self.__class__(new)
        else:
            if not self.ndim >= other.ndim:
                raise TypeError(f"Cannot multiply {self} with {other}")
            new = tuple(i * j for i, j in zip(self.shape[-i:], other.shape[-i:]))
            return self.__class__((*self.shape[:d], *new))

    __rmul__ = __mul__

    def __truediv__(self, other):
        try:
            i = min(self.ndim, other.ndim)
            d = max(self.ndim, other.ndim) - i
        except AttributeError:
            new = tuple(i / other for i in self.shape)
            return self.__class__(new)
        else:
            if not self.ndim >= other.ndim:
                raise TypeError(f"Cannot divide {self} by {other}")
            new = tuple(i / j for i, j in zip(self.shape[-i:], other.shape[-i:]))
            return self.__class__((*self.shape[:d], *new))

    __rtruediv__ = __truediv__


class Shape4d(ShapeNd):
    ndim = 4
    def __init__(self, shape):
        super(Shape4d, self).__init__(shape)

    width =  property(lambda self: self._get_dim(-1), lambda self, val: self._set_dim(-1, val))
    height = property(lambda self: self._get_dim(-2), lambda self, val: self._set_dim(-2, val))
    depth =  property(lambda self: self._get_dim(-3), lambda self, val: self._set_dim(-3, val))
    num =    property(lambda self: self._get_dim(-4), lambda self, val: self._set_dim(-4, val))

class Shape3d(ShapeNd):
    ndim = 3
    def __init__(self, shape):
        super(Shape3d, self).__init__(shape)

    width =  property(lambda self: self._get_dim(-1), lambda self, val: self._set_dim(-1, val))
    depth =  property(lambda self: self._get_dim(-3), lambda self, val: self._set_dim(-2, val))
    num =    property(lambda self: self._get_dim(-4), lambda self, val: self._set_dim(-3, val))


class Params2d(ShapeNd):
    ndim = 2
    def __init__(self, shape):
        super(Params2d, self).__init__(shape)

    width =  property(lambda self: self._get_dim(-1), lambda self, val: self._set_dim(-1, val))
    height = property(lambda self: self._get_dim(-2), lambda self, val: self._set_dim(-2, val))


class Params1d(ShapeNd):
    ndim = 1
    def __init__(self, shape):
        super(Params1d, self).__init__(shape)

    width = property(lambda self: self._get_dim(-1), lambda self, val: self._set_dim(-1, val))





x4 = Shape4d((1, 3, 10, 10))
p2 = Params2d((1, 2))
p1 = Params1d((3,))

# print()
# y = x4 + 2 * p2
# print(x4)
# print(p2)
# print(y)

# print()
# y = x4 + 2 * p1
# print(x4)
# print(p1)
# print(y)


x3 = Shape3d((1, 3, 10))

print()
y = x3 + p1
print(x3)
print(p1)
print(y)
