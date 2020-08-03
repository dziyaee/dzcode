
class Operators():
    @classmethod
    def use(self, operator1, operator2):
        def decorator(wrapped_func):
            def wrapper(operand1, operand2):
                try:
                    n = min(operand1.ndim, operand2.ndim) # negative indices for both operands over which to operate
                    d = max(operand1.ndim, operand2.ndim) - n # positive indices for operand1 over which to not operate
                except AttributeError:
                    new = operator1(operand1, operand2)
                    return operand1.__class__(new)
                else:
                    if operand1.ndim >= operand2.ndim: # normal operand order
                        new = operator2(operand1, operand2, n)
                        return operand1.__class__((*operand1.shape[:d], *new))
                    else: # reverse operand order
                        new = operator2(operand2, operand1, n)
                        return operand2.__class__((*operand2.shape[:d], *new))
            return wrapper
        return decorator

    add1 = lambda operand1, operand2: tuple(i + operand2 for i in operand1.shape)
    add2 = lambda operand1, operand2, n: tuple(i + j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))

    sub1 = lambda operand1, operand2: tuple(i - operand2 for i in operand1.shape)
    sub2 = lambda operand1, operand2, n: tuple(i - j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))

    mul1 = lambda operand1, operand2: tuple(i * operand2 for i in operand1.shape)
    mul2 = lambda operand1, operand2, n: tuple(i * j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))

    div1 = lambda operand1, operand2: tuple(i / operand2 for i in operand1.shape)
    div2 = lambda operand1, operand2, n: tuple(i / j for i, j in zip(operand1.shape[-n:], operand2.shape[-n:]))


class Dimension():
    def __init__(self, n):
        self.n = n

    def __get__(self, obj, cls):
        if obj is None:
            return self
        return obj._shape[self.n]

    def __set__(self, obj, val):
        obj._shape[self.n] = val


class ShapeNd():
    def __init__(self, shape):
        self.shape = shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return tuple(self._shape)

    @shape.setter
    def shape(self, shape):
        if len(shape) != self.ndim:
            raise ValueError(f"{self.__class__.__name__} only accepts tuples of {self.ndim} numbers, got {shape}")
        self._shape = list(shape)

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


class Shape4d(ShapeNd):
    _ndim = 4
    def __init__(self, shape):
        super(Shape4d, self).__init__(shape)

    width = Dimension(-1)
    height = Dimension(-2)
    depth = Dimension(-3)
    num = Dimension(-4)

class Shape3d(ShapeNd):
    _ndim = 3
    def __init__(self, shape):
        super(Shape3d, self).__init__(shape)

    width = Dimension(-1)
    depth = Dimension(-2)
    num = Dimension(-3)

class Shape2d(ShapeNd):
    _ndim = 2
    def __init__(self, shape):
        super(Shape2d, self).__init__(shape)

    width = Dimension(-1)
    height = Dimension(-2)

class Shape1d(ShapeNd):
    _ndim = 1
    def __init__(self, shape):
        super(Shape1d, self).__init__(shape)

    width = Dimension(-1)





# x4 = Shape4d((1, 3, 10, 10))
# p2 = Params2d((1, 2))
# p1 = Params1d((3,))

# # print()
# # y = x4 + 2 * p2
# # print(x4)
# # print(p2)
# # print(y)

# # print()
# # y = x4 + 2 * p1
# # print(x4)
# # print(p1)
# # print(y)


# x3 = Shape3d((1, 3, 10))

# print()
# # y = x3 + p1
# y = x4 + x3
# print(x4)
# print(x3)

# # print(p1)
# print(y)
# print(y.width)
# y.width = 9
# print(y)
# print(y.width)
# print(y.shape)
# # print(y.ndim)
# y.shape = (3, 4, 1, 2)
# print(y)
# print(y.width)
# print(y.shape)
# print(y.ndim)


x = Shape4d((1, 3, 10, 10))
p = Shape2d((2, 3))
print(x)
print(p)

print()
y = p + x
print(y)

print()
y = x + p
print(y)
