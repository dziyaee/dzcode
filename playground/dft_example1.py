import numpy as np
from numpy import pi, sin, cos, arctan2
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")
mpl.style.use("default")


# WHAT THIS IS:
# My attempt to implement a DFT using matrix operations. From Book: Understanding-Digital-Signal-Processing/3.1.1/DFT-Example-1 by Richard Lyons

# 8-point DFT
N = 8
nn = mm = np.arange(0, N)
n, m = np.meshgrid(np.arange(0, N), np.arange(0, N))

# Input Signal
fs = 8e3
fa = 1e3
fb = 2e3
an = 1 * sin(2 * pi * fa/fs * nn)
bn = 0.5 * sin(2 * pi * fb/fs * nn + (3 * pi / 4))
xn = an + bn

# DFT Real and Imaginary Components
real = np.sum(xn * cos(2 * pi * n * m / N), axis=1)
imag = np.sum(xn * -sin(2 * pi * n * m / N), axis=1)

# Mag, Power
mag = ((real ** 2) + (imag ** 2)) ** 0.5
power = mag ** 2

# Phase
eps = 1e-15
real[(real > -eps) & (real < eps)] = 0
imag[(imag > -eps) & (imag < eps)] = 0
phase = arctan2(imag, real) * (360 / (2 * pi))

# Plot
titles = ['X(m) Magnitude', 'X(m) Real', 'X(m) Phase', 'X(m) Imaginary']
color = 'black'
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
fig.subplots_adjust(hspace=0.5)

for axis, data, title in zip(axes.flatten(), [mag, real, phase, imag], titles):
    axis.axhline(0, lw=1, color=color)
    axis.plot(mm, data, 's', color=color)
    axis.vlines(mm, ymin=0, ymax=data, ls='--', color=color)
    axis.set_title(title)
    axis.set_xlabel("Frequency (kHz)")
    axis.xaxis.set_label_coords(0.85, -0.15)
    ticks = [np.round(x, 1) if x % 1 != 0 else x for x in data]
    axis.set_yticks(ticks)

fig.suptitle("DFT Results")
plt.show(block=True)
