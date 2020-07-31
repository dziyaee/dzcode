import numpy as np
from numpy import pi, sin
from collections import namedtuple
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

mpl.use("Qt5Agg")
mpl.style.use('default')


# v = Amplitude, n = Sample time indices, fo = Signal Frequency, fs = Sample Frequency
Signal = namedtuple("Signal", 'a n fo fs')


def sample_sinewave(original_freq, sample_freq):
    """Function to create a sinewave with original_freq sampled at sample_freq

    Args:
        original_freq (int): original signal's frequency
        sample_freq (int): frequency with which to sample the signal

    Returns:
        x (Signal namedtuple):
            x.a (np array): 1d vector containing the amplitudes of the sampled signal
            x.n (np array): 1d vector containing the sampling time indices of the sampled signal
            x.fo (int): original signal's frequency
            x.fs (int): sampling frequency used to sample the original signal
    """
    fo, fs = original_freq, sample_freq
    ts = 1/fs
    n = np.arange(0, 1+ts, ts)
    x = sin(2*pi*fo*n)
    x = Signal(x, n, fo, fs)
    return x


fig, ax = plt.subplots(figsize=(14, 5))
fig.subplots_adjust(top=0.95, bottom=0.2)
ax.set_xlim(0, 1)

# Initial Plot
fo = 2
fs = 100
x = sample_sinewave(original_freq=fo, sample_freq=fs)
line = ax.plot(x.n, x.a)

# Slider 1
left = 0.125
bot = 0.1
width = 0.775
height = 0.03
slider1 = plt.axes([left, bot, width, height])
signal_freq = Slider(ax=slider1, label='Signal Frequency', valmin=0, valmax=30, valinit=fo, valstep=0.1, valfmt='%.1fHz')

# Slider 2
left = 0.125
bot = 0.05
width = 0.775
height = 0.03
slider2 = plt.axes([left, bot, width, height])
sample_freq = Slider(ax=slider2, label='Sample Frequency', valmin=1, valmax=100, valinit=fs, valstep=1, valfmt='%.0fHz')


def update(val):
    fo = signal_freq.val
    fs = sample_freq.val
    x = sample_sinewave(original_freq=fo, sample_freq=fs)
    line[0].set_data(x.n, x.a)
    fig.canvas.draw_idle()

signal_freq.on_changed(update)
sample_freq.on_changed(update)


plt.show(block=True)
