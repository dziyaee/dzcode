import numpy as np
from numpy import pi, sin
import matplotlib as mpl
import matplotlib.style
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")
mpl.style.use('default')
from collections import namedtuple
from dzlib.plotting.animation import Animate



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

# Plotting 3 signals per frame to show sampled signal frequency ambiguity at a particular sampling frequency
# x(t), 7Hz Sinewave sampled at 1000Hz, (connected markers)
# x(n), 7Hz Sinewave sampled at 6Hz, (discrete markers)
# z(t), 7Hz + k*6Hz Sinewave sampled at 1000Hz for k from -10 to 10, (connected markers)
xt = sample_sinewave(original_freq=7, sample_freq=1000)
xn = sample_sinewave(original_freq=7, sample_freq=6)
zt = []
ks = list(np.arange(-10, 11, 1))
for k in ks:
    fo = xt.fo + k*xn.fs
    zt.append(sample_sinewave(original_freq=fo, sample_freq=1000))

def init():
    ax.plot(xt.n, xt.a)
    ax.plot(xn.n, xn.a, 'o')
    ax.set_xlim(0, 1)
    ax.set_title(f"Orange Markers show samples from {xn.fs}Hz sampling frequency")
    return None

def update(i):
    line = ax.plot(zt[i].n, zt[i].a, color='C2')
    ax.legend([f'x(t): {xt.fo:3d}Hz', f'x(n): {xn.fo:3d}Hz', f'z(n): {zt[i].fo:3d}Hz'], loc='upper right', title='Signal Frequencies')
    return line

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
frames = np.arange(len(ks))
anim = Animate(fig=fig, ax=ax, frames=frames, update_func=update, init_func=init, pause=1, hold=True)
anim.animate()
