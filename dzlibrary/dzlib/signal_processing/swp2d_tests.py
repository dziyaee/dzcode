import numpy as np
from Class_Signal_Processing import SWP2d

def parity(integer):

    return "even" if integer % 2 == 0 else "odd"



modes = ["user", "full", "none", "auto", "keep"]


# mode = "auto"
# mode = "full"
mode = "none"

for i in range(10):

    Nx = 1
    Dx = np.random.randint(1, 10)
    (Hx, Wx) = np.random.randint(5, 20, 2)

    Nk = np.random.randint(1, 10)
    Dk = Dx
    (Hk, Wk) = np.random.randint(1, 6, 2)

    Sk = np.random.randint(1, 4)
    (Hp, Wp) = np.random.randint(0, 10, 2)

    x = np.random.uniform(0, 1, (Nx, Dx, Hx, Wx))
    k = np.random.uniform(-1, 1, (Nk, Dk, Hk, Wk))

    sweeper = SWP2d(image = x, kernel_size = (Hk, Wk), pad = (Hp, Wp), stride = Sk, mode = mode)

    sweeper.print_attrs()
    print("-" * 100)
