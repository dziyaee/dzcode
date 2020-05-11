import numpy as np
from numpy import pi, sin
from collections import namedtuple
from dzlib.plotting.animation import Animate
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot as plt

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


# Plotting 2 signals per frame to show aliasing effects at varying sampling frequencies
# x(t), 7Hz Sinewave sampled at 1000Hz, (connected markers)
# x(n), 7Hz Sinewave sampled at fs for fs from 1Hz to 20Hz (connected markers)
xt = sample_sinewave(original_freq=7, sample_freq=1000)
xn = []
sample_freqs = list(np.arange(1, 21))
for fs in sample_freqs:
    xn.append(sample_sinewave(xt.fo, fs))


# Animation functions and calls
def init():
    ax.plot(xt.n, xt.a, lw='3')
    ax.set_xlim(0, 1)
    ax.set_title(f"Sine wave with frequency {xt.fo}Hz sampled at varying sampling frequencies")
    return None


def update(i):
    # line = ax.plot(xn[i].n, xn[i].a, color=colors[1], marker="o")
    line = ax.plot(xn[i].n, xn[i].a, color='C1', marker="o")
    ax.legend([f'x(t), fs={xt.fs}Hz', f'x(n), fs={xn[i].fs}Hz'], loc='upper right', title='Sampling Frequencies')
    return line


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
frames = np.arange(len(sample_freqs))
anim = Animate(fig=fig, ax=ax, frames=frames, update_func=update, init_func=init, pause=1, hold=True)
anim.animate()
