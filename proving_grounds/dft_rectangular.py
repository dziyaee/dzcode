import numpy as np
from scipy.fftpack import fft, fftshift
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

mpl.use("Qt5Agg")
mpl.style.use("default")


def make_window(num_samples, n0, K, A):
    window = np.zeros(num_samples)
    window[n0:n0+K] = A
    center_offset = int(np.floor(K / 2))
    center = (n0 + center_offset) % num_samples
    return window


# Window (n0, K, A: starting index, number and value of non-zero values, respectively)
num_samples = 64
n0 = 0
K = 2
# assert K % 2 == 1
A = 1
window = make_window(num_samples, n0, K, A)
center_offset = int(np.floor(K / 2))
center = (n0 + center_offset) % num_samples

# DFT (N: number of points of DFT)
N = 64
if N % 2 == 0:
    lowest_m = -(N/2) + 1
    highest_m = (N/2)
elif N % 2 == 1:
    lowest_m = -(N-1)/2
    highest_m = (N-1)/2

# sample indices
# n = np.arange(0-(N/2), N-(N/2), 1)
n = np.arange(N)

# m = np.arange(lowest_m-1, highest_m, 1)
m = np.arange(N)

dft = fft(window, N)
# dft = fftshift(dft)
real, imag, mag, phase = dft.real, dft.imag, np.abs(dft), np.angle(dft)

# Figure & Axes inits
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(nrows=3, ncols=2)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :1])
ax3 = plt.subplot(gs[1, 1:])
ax4 = plt.subplot(gs[2, :1])
ax5 = plt.subplot(gs[2, 1:])
fig.subplots_adjust(top=0.95)
axes = [ax1, ax2, ax3, ax4, ax5]
fig.suptitle(f"{N}-point DFT of a Rectangular Function, $x(n)$, with {K} non-zero values. $x(n)$ is currently centered about n = {center} ({n0 + center_offset})")

# Slider for Rectangle starting n, n0
left = 0.125
bot = 0.04
width = 0.775
height = 0.03
slider1 = plt.axes([left, bot, width, height])
n_shift = Slider(ax=slider1, label='Rectangle Shift', valmin=0, valmax=2*N, valinit=n0, valstep=1, valfmt='%.0f')

# Slider for Rectangle Number of Non-Zero Values
left = 0.125
bot = 0.005
width = 0.775
height = 0.03
slider2 = plt.axes([left, bot, width, height])
k_num = Slider(ax=slider2, label='Non-Zero Values', valmin=0, valmax=N, valinit=K, valstep=1, valfmt='%.0f')

# Axes labels
xlabels = ['n', 'm', 'm', 'm', 'm']
ylabels = ['$x(n)$', '$X_{real}(m)$', '$X_{imag}(m)$', '$|X(m)|$', '$X_{\phi}(m)$']
for axis, xlabel, ylabel in zip(axes, xlabels, ylabels):
    axis.set_xlabel(xlabel, fontsize=12)
    axis.xaxis.set_label_coords(1.00, -0.025)
    axis.set_ylabel(ylabel, fontsize=12)

# Initial Plots
lines1 = []
for axis, xdata, ydata in zip(axes, [n, m, m, m, m], [window, real, imag, mag, phase]):
    # lines1.append(axis.plot(xdata, ydata, marker='s'))
    lines1.append(axis.plot(xdata, ydata, marker='.', markersize=5, markerfacecolor='r', markeredgecolor='r'))
    axis.axhline(0, color='k', lw=1)
    axis.axvline(0, color='k', lw=1)
    axis.set_ylim(np.min(ydata)-1, np.max(ydata)+1)

# Updating Plots
def update(val):
    global lines1

    K = int(k_num.val)
    window = make_window(num_samples, n0, K, A)

    shift = int(n_shift.val)
    shifted_window = np.roll(window, shift)
    center_offset = int(np.floor(K / 2))
    center = (shift + center_offset) % num_samples


    fig.suptitle(f"{N}-point DFT of a Rectangular Function with {K} non-zero values. $x(n)$ is currently centered about n = {center} ({shift + center_offset})")

    dft = fft(shifted_window, N)
    # dft = fftshift(dft)
    real, imag, mag, phase = dft.real, dft.imag, np.abs(dft), np.angle(dft)

    for line, axis, xdata, ydata in zip(lines1, axes, [n, m, m, m, m], [shifted_window, real, imag, mag, phase]):
        line[0].set_data(xdata, ydata)
        axis.set_ylim(np.min(ydata)-1, np.max(ydata)+1)

    fig.canvas.draw_idle()

n_shift.on_changed(update)
k_num.on_changed(update)

plt.show(block=True)
