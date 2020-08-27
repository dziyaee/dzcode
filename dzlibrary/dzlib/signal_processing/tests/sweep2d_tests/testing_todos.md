TESTING THE INIT:
VARS:
- padded array dimension (px)
- unpadded array dimension (x)
- padding dimension (p)
- kernel dimension (k)
- stride dimension (s)
- output dimension (y)
- im2col dimension (im)

<!-- 1) INPUT -->
<!-- - any mode: x = x -->
<!-- - tuple of 4 ints >= 1 -->

<!-- 2) PADDING -->
<!-- a) padding vals -->
<!-- - user mode: p = p -->
<!-- - user mode: p % 1 = 0 -->
<!-- - full mode: p = k - 1 -->
<!-- - same mode: p = (k - s + x * (s - 1)) / 2 -->
<!-- - .shape = tuple of 2 numbers >= 0 -->

<!-- b) padded array dims
- user mode: px = x + 2p
- full mode: px = x + 2k - 2
- same mode: px = x + k - 1
- padded array shape matches padded array dims and input array shape
 -->
<!-- c) padding indices -->
<!-- - lower: int(math.ceil(p)) -->
<!-- - high: int(math.ceil(x - p)) -->
<!-- - edge case: 'same' mode can result in uneven padding, panding indices should yield left-top bias padding -->


3) KERNEL:
a) raise exception for kh > pxh or kw > pxw
b) raise exception for kd != xd
<!-- - any mode: k = k -->
<!-- - .shape = tuple of 4 ints >= 1 -->

<!-- 4) STRIDE: -->
<!-- - user mode: s = user input -->
<!-- - full mode: s = 1 -->
<!-- - same mode: s = 1 -->
<!-- - .shape = tuple of 2 ints >= 1 -->

5) MODE:
- raise exception if mode not in modes

<!-- 6) OUTPUT: -->
<!-- - yw = int(((pxw - kw) / sw) + 1)
- yh = int(((pxh - kh) / sh) + 1)
- yd = kn
- yn = xn
- .shape = tuple of 4 ints >= 1
 -->
7) IM2COL:
- imd = xn
<!-- - imh = kd * kh * kw
- imw = yh * yw
- .shape = tuple of 3 ints >= 1
 -->
8) MISC:
a) INPUT_NDIM = 4
b) PARAM_NDIM = 2
c) SWEEP_AXES = (2, 3)
