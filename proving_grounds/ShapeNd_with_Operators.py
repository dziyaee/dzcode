

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


class Operators():
    @classmethod
    def use(self, operator1, operator2):
        def decorator(wrapped_func):
            def wrapper(operand1, operand2):
                n = operand1.naxes
                l = len(operand1)
                d = l - n
                try:
                    if n == operand2.naxes:
                        new = operator2(operand1, operand2, n)
                        return operand1.__class__((*operand1.shape[:d], *new))
                    else:
                        raise ValueError(f"Operand nds must be equal, got {operand1.naxes}, {operand2.naxes}")
                except AttributeError:
                    new = operator1(operand1, operand2, n)
                    return operand1.__class__((*operand1.shape[:d], *new))
            return wrapper
        return decorator

    add1 = lambda operand1, operand2, n: tuple(i + operand2 for i in operand1.shape[-n:])
    add2 = lambda operand1, operand2, n: tuple(i + j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))

    sub1 = lambda operand1, operand2, n: tuple(i - operand2 for i in operand1.shape[-n:])
    sub2 = lambda operand1, operand2, n: tuple(i - j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))

    mul1 = lambda operand1, operand2, n: tuple(i * operand2 for i in operand1.shape[-n:])
    mul2 = lambda operand1, operand2, n: tuple(i * j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))

    div1 = lambda operand1, operand2, n: tuple(i / operand2 for i in operand1.shape[-n:])
    div2 = lambda operand1, operand2, n: tuple(i / j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))

    fdiv1 = lambda operand1, operand2, n: tuple(i // operand2 for i in operand1.shape[-n:])
    fdiv2 = lambda operand1, operand2, n: tuple(i // j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))


class ShapeNd():
    def __init__(self, shape):
        self.shape = shape

    @property
    def naxes(self):
        return self._naxes

    @property
    def shape(self):
        return tuple(self._shape)

    @shape.setter
    def shape(self, shape):
        if len(shape) < self.naxes:
            raise ValueError(f"{self.__class__.__name__} only accepts tuples of >= {self.naxes} numbers, got {shape}")
        self._shape = list(shape)

    def __len__(self):
        return len(self.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}({tuple(self.shape)})"

    @Operators.use(Operators.add1, Operators.add2)
    def __add__(self, other):
        pass

    __radd__ = __add__

    @Operators.use(Operators.sub1, Operators.sub2)
    def __sub__(self, other):
        pass

    __rsub__ = __sub__

    @Operators.use(Operators.mul1, Operators.mul2)
    def __mul__(self, other):
        pass

    __rmul__ = __mul__

    @Operators.use(Operators.div1, Operators.div2)
    def __truediv__(self, other):
        pass

    __rtruediv__ = __truediv__

    @Operators.use(Operators.fdiv1, Operators.fdiv2)
    def __floordiv__(self, other):
        pass



if __name__ == "__main__":
    class Shape4d(ShapeNd):
        _naxes = 2
        def __init__(self, shape):
            super(Shape4d, self).__init__(shape)

        width = Dimension(-1)
        height = Dimension(-2)
        depth = Dimension(-3)
        num = Dimension(-4)

    x = Shape4d((1, 3, 10, 10))
    k = Shape4d((1, 2, 4, 5))
    p = Shape4d((2, 1))
    s = Shape4d((1, 2))

    p.height = (k.height - s.height + x.height * (s.height - 1)) / 2
    p.width = (k.width - s.width + x.width * (s.width - 1)) / 2
    print(p)

    x = Shape4d((1, 3, 10, 10))
    k = Shape4d((1, 2, 4, 5))
    p = Shape4d((2, 1))
    s = Shape4d((1, 2))

    z = Shape4d((0, 0))
    p = z + (k - s + x * (s - 1)) / 2
    print(p)

    p = z + k - 1
    print(p)
