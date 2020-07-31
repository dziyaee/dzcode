import numpy as np
from numpy import pi, sin, cos, exp
from scipy.fft import fft, fft2
from scipy.linalg import dft as sp_dft_matrix1d
from dzlib.common.utils import info


# WHAT THIS IS:
# My first implementation of a 1d DFT operation via constructing a dft matrix to be used in a matrix multiplcation with an input vector
# x: (N x 1) input vector (time domain)
# X: (N x 1) output vector (frequency domain)
# dft_matrix: (M x N) matrix representation of the complex exponentials used in the DFT operation. The rows represent separate complex exponentials at the different DFT analysis frequencies. The columns represent each time sample of each complex exponential. Note that in this case M, the number of analysis frequencies or frequency 'samples', is equal to N, the number of time-domain samples of the input signal

# I am checking my work against the DFT implementation used by the SciPy library. Specifically, I will check my output X against sp_X, which calculates the DFT of the input x via the scipy.fft.fft function, and dft_matrix against sp_dft_matrix, which returns the DFT Matrix via the scipy.linalg.dft function

# This lambda function is simply an expression of the complex exponential e^-j(2pi*m*n/N). Note that this expression does accept numpy arrays as inputs for the m and n variables. In particular, both m and n are both (NxN) 'meshgrid' matricies which, when combined, represent all possible (m, n) index pairs used in the construction of the DFT Matrix.
dft_matrix1d = lambda  N, m, n : exp(-1j * ((2 * pi / N) * m * n))

# Input x, number of samples N, time sample indices n, frequency sample indices m
x = np.array([2, 1, 0, 1])
N1 = x.size
n1 = np.arange(N1)
m1 = np.arange(N1)

# Convert n and m to time sample and frequency sample meshgrids
(n1, m1) = np.meshgrid(n1, m1)

# Calculate dft_matrix and check against sp_dft_matrix. Note that np.array_equal will check for *exact* equality
dft_matrix = dft_matrix1d(N1, m1, n1)
sp_dft_matrix = sp_dft_matrix1d(N1)
if np.array_equal(dft_matrix, sp_dft_matrix):
    print(f"dft_matrix and sp_dft_matrix are equal")
else:
    print(f"dft_matrix and sp_dft_matrix are not equal")

# Calculate DFT Output X and check against sp_X. Note that here np.isclose is used to calculate equality to allow for very small, near-zero errors. I am unsure of the cause of these errors yet. Because of these errors, I have also calculated an average error by taking the mean of the absolute value of the differences in both outputs.
X = np.matmul(dft_matrix, x)
sp_X = fft(x)
errors = X - sp_X
avg_error = np.mean(np.abs(errors))

if np.all(np.isclose(X, sp_X)):
    print(f"X and sp_X are close, Average Error = {avg_error}")
else:
    print(f"X and sp_X are not close, Average Error = {avg_error}")
