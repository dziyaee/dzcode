import numpy as np
from numpy import log10
from numpy import pi, exp, cos, sin
from scipy.fftpack import fft2
import matplotlib.pyplot as plt
import os
from PIL import Image


# WHAT THIS IS:
# My first implementation of a 2d DFT operation via constructing a dft matrix to be used in the a matrix multiplication with an input vector
# x: (N1 x N2) input vector (space domain)
# X: (M1 x M2) DFT Output Matrix (frequency domain)
# dft_matrix: (M1.M2 x N1.N2) matrix representation of the complex exponentials used in the DFT operation. The rows represent separate complex exponentials at the different DFT analysis frequencies. The columns represent each time sample of each complex exponential. Note that in this case M1 and M2, the number of analysis frequencies or frequency pair 'samples', is equal to N1 and N2, the number of time-domain pair samples of the input signal

# I am checking my work against the DFT implementation used by the SciPy library. Specifically, I will check my output X against sp_X, which calculates the DFT of the input x via the scipy.fft.fft2 function. Note, unlike the 1d implementation, I will not be checking my work against a Scipy 2d DFT Matrix function, as none exists at this time.

# This function is simply an expression of the complex exponential e^-j(2pi(m1*n1/N1 + m2*n2/N2)). Note that this expression does accept numpy arrays as inputs for the m and n variables. In particular, both m and n are both (M1.M2 x N1.N2) 'meshgrid' matricies which, when combined, represent all possible (m1, m2, n1, n2) index groups used in the construction of the DFT Matrix.
def dft_matrix2d(Ns, ns, ms):
    N1, N2 = Ns
    n1, n2 = ns
    m1, m2 = ms
    dft_matrix = exp(-2j * pi * ((m1 * n1 / N1) + (m2 * n2 / N2)))
    return dft_matrix

# Create the input x
image_path = '/'.join((os.getcwd(), 'image1.jpg'))
x = np.asarray(Image.open(image_path).resize((64, 80)))

# Meshgrids: It was a process of trial and error to get the meshgrid matrices to be correct by transposing them as needed. I already worked out what they needed to look like. The final DFT Matrix is an (M1.M2 x N1.N2) matrix. Note, I plan to study how the np.meshgrid function works in more detail so that the next time I don't have to rely on trial and error.

# meshgrids for n1 and n2
# Going across a columns:
# The n1 pattern is (0 N2 times, 1 N2 times, ..., N1-1 N2 times)
# The n2 pattern is (0, 1, ..., N2-1 N1 times)
N1, N2 = x.shape
n1, n2 = np.arange(N1), np.arange(N2)
(n1, n2) = np.meshgrid(n1, n2)
n1 = n1.T
n2 = n2.T
(n1, n2) = np.meshgrid(n1.flatten(), n2.flatten())
n2 = n2.T

# meshgrids for m1 and m2
# Going across rows:
# The m1 pattern is (0 M2 times, 1 M2 times, ..., M1-1 M2 times)
# The m2 pattern is (0, 1, ..., M2-1 M1 times)
M1, M2 = N1, N2
m1, m2 = np.arange(M1), np.arange(M2)
(m1, m2) = np.meshgrid(m1, m2)
m1 = m1.T
m2 = m2.T
(m1, m2) = np.meshgrid(m1, m2)
m1 = m1.T

# Construct DFT Matrix out of the n1, n2, m1, m2 meshgrids and the complex exponential 2d function defined above
Ns = [N1, N2]
ns = [n1, n2]
ms = [m1, m2]
dft_matrix = dft_matrix2d(Ns, ns, ms)

# Compute DFT and Scipy DFT
X = np.matmul(dft_matrix, x.flatten()).reshape(N1, N2)
sp_X = fft2(x)

# Compute Error
errors = X - sp_X
avg_error = np.mean(np.abs(errors))
if np.all(np.isclose(X, sp_X)):
    print(f"X and sp_X are close, Average Error = {avg_error}")
else:
    print(f"X and sp_X are not close, Average Error = {avg_error}")

# Plot the test image, my DFT, and SciPy's DFT
datas = [x, log10(np.abs(X)), log10(np.abs(sp_X))]
for i in range(1, 3, 1):
    datas[i] = (datas[i] - np.min(datas[i])) / np.max(datas[i])

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.subplots_adjust(wspace=0, left=0.01, right=0.99)
for axis, data in zip(axes, datas):
    axis.imshow(data)
    axis.set_xticks([])
    axis.set_yticks([])
plt.show(block=True)
