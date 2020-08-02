
class SweepNd(object):
    def __init__(self, images_shape, kernels_shape, padding, stride, mode='user'):
        self.images_shape = images_shape
        self.kernels_shape = kernels_shape
        self.padding= padding
        self.stride = stride
        self.mode = mode
        print(f"HERE: {self.padding}")

        pass

    @staticmethod
    def _totuple(x, n):
        if isinstance(x, (tuple, list)):
            return x
        print(f"converting {[x]} to {tuple((x,) * n)}")
        return tuple((x,) * n)

    def _validate(x, max_len, min_val):
        # tuple max length validation
        if len(x) > max_len or len(x) < 1:
            raise ValueError(f"Expected tuple of {max_len} values, got {x} with {len(x)} values")
        for e in x:
            # element type int check
            if not isinstance(e, int):
                raise TypeError(f"Expected only integer elements in {x}, got {type(e)}")
            # element minimum value check
            if not e >= min_val:
                raise ValueError(f"Expected only values >= {min_val} in {x}, got {e}")

    def _expand(x, nd, f):
        f = (f,) * nd
        return tuple(*f, *x)



class Sweep2d(SweepNd):
    def __init__(self, images_shape, kernels_shape, padding, stride, mode='user'):
        totuple = SweepNd._totuple
        validate = SweepNd._validate
        expand = SweepNd._expand
        padding = totuple(padding, 2)
        stride = totuple(stride, 2)
        super(Sweep2d, self).__init__(images_shape, kernels_shape, padding, stride, mode)

class Sweep1d(SweepNd):
    def __init__(self, images_shape, kernels_shape, padding, stride, mode='user'):
        totuple = SweepNd._totuple
        padding = totuple(padding, 1)
        stride = totuple(stride, 1)
        super(Sweep1d, self).__init__(images_shape, kernels_shape, padding, stride, mode)


if __name__ == "__main__":
    pass

