
## programmatic class properties example
width =  property(lambda self: self._get_dim(-1), lambda self, val: self._set_dim(-1, val))
height = property(lambda self: self._get_dim(-2), lambda self, val: self._set_dim(-2, val))
depth =  property(lambda self: self._get_dim(-3), lambda self, val: self._set_dim(-3, val))
num =    property(lambda self: self._get_dim(-4), lambda self, val: self._set_dim(-4, val))


## slicing of arbitrary ndim array
cols = slice(1, 3)
rows = slice(1, 3)
i = np.s_[..., rows, cols]
# example input / output
x = np.zeros((1, 1, 4, 4))
x[i] = 1


## matplotlib tick labels w/ scientific notation, scilimits can be adjusted to determine range where scientific notation is used
fig, ax1 = plt.subplots(ncols=1, nrows=1)
ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))


## timer decorator with auto unit calc
def timer(func):
    units = ('s', 'ms', 'us', 'ns')
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        stop_time = time.perf_counter()
        run_time = stop_time - start_time
        m = int(np.abs(np.log10(run_time) // 3))
        run_time *= (1e3) ** m
        print(f"{func.__name__!r} time: {(run_time):.1f} {units[m]}")
        return value
    return wrapper


## ntimes decorator preserving normal function
def ntimes(func):
    def inner(n, *args, **kwargs):
        def wrapper():
            return [func(*args, **kwargs) for i in range(n)]

        result = wrapper()
        return result
    func.ntimes = inner
    return func
