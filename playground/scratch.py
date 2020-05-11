import numpy as np
from numpy import pi, sin
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt



def dsp_plot(nrows, ncols, plots, n, titles, subplots_kwargs):
    assert nrows * ncols == len(plots) == len(titles)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **subplots_kwargs)
    if not isinstance(axes, list):
        axes = [axes]
    for ax, points, title in zip(axes, plots, titles):
        ax.plot(n, points, 'ks')
        ax.vlines(n, ymin=0, ymax=points, ls='--')
        ax.axhline(0, color='k')
        # ax.set_xticks(np.arange(0, 1+0.25, 0.25))
        ax.set_title(title)
        ax.yaxis.set_tick_params(labelleft=True)
        # for x, y in zip(n, points):
        #     ax.annotate(f"{y:.2f}", (x+0.05, y+0.03))
        #     ax.annotate(f"{x}", (x+0.05, -0.1))
        # ax.xaxis.set_tick_params(labelbottom=False)
        # ax.set_xticks([])

    plt.show(block=True)
    return fig

# ts = 1
# Nt = 5
# To = ts * Nt
# fo = 1 / To

# n = np.arange(0, 2*Nt+2, 1)
# x = sin(2 * pi * fo * n * ts)
# plots = [x]
# titles = ['x(n)']

A = 5
f = 0.25
n = np.arange(3, 7+1, 2)
times = np.arange(0, 10, 0.1)
s = []
for t in times:
    s.append((4 * A / pi) * np.sum(np.sin(2 * pi * n * f * t) / n))

plots = [s]
n = times
titles = ['square']
dsp_plot(nrows=1, ncols=1, plots=plots, n=n, titles=titles, subplots_kwargs={'sharey': True, 'figsize': (14, 5)})



