import numpy as np
from numpy import pi, exp, cos, sin
from scipy.fftpack import fft, fft2
from scipy.linalg import dft
from dzlib.common.utils import info


# WHAT THIS IS
# My second implementation of a 2d DFT operation via a series of 1d DFTs.

# Variables:
# x: (N1 x N2) Image
# dft1: (M1 x N1) Columns DFT Matrix, DFT Matrix for 1d Columns DFT
# dft2: (M2 x N2) Rows DFT Matrix, DFT Matrix for 1d Rows DFT
# Y: (M1 x N2) Columns DFT, DFT Output of Image columns
# Z: (M2 x M1) Rows DFT, DFT Output of Columns DFT
# X: (M1 x M2) Image DFT, DFT Output of Image

# Equation:
# X = (dft2*(dft1*x).T).T

# Method:
# The initial image x can be thought of as N2 columns of 1d images each of size N1
# The 1d DFT of each of these images can be computed by matrix multiplying the 1d DFT Matrix of shape (M1 x N1) with the image x
# The resulting output can be thought of as N1 columns of 1d DFT images each of size M1
# This DFT output image is then transposed in order to DFT across rows, and thus 'correlating' the original image's columns and rows
# Again, the final 1d DFT of these images can be computed by matrix multiplying the 1d DFT Matrix of shape (M2 x N2) with the transposed DFT Output
# The final resulting DFT output image is of shape (M2 x M1)
# To obtain the desired result of the original image's DFT, we transpose the final DFT output to obtain a DFT output of shape (M1 x M2)

# Random input image
N1 = np.random.randint(1, 11)
N2 = np.random.randint(1, 11)
x = np.random.randn(N1, N2)

# Function to implement the steps outlined above. I have purposefully kept some of the steps separate to have a clearer view of what's happening
def dft2d(x):
    assert x.ndim == 2
    N1, N2 = x.shape
    dft1, dft2 = dft(N1),  dft(N2)
    X1 = np.matmul(dft1, x).T
    X2 = np.matmul(dft2, X1).T
    return X2

# Compute and compare my 2d DFT implementation with Scipy's
X = dft2d(x)
sp_X = fft2(x)

errors = X - sp_X
avg_error = np.mean(np.abs(errors))

if np.all(np.isclose(X, sp_X)):
    print(f"X is close to sp_X, Average Error = {avg_error}")
else:
    print(f"X is not close to sp_X, Average Error = {avg_error}")
