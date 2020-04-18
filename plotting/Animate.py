import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")


class Animate():

    def __init__(self, fig, ax, init_func, update_func, frames, pause=0.1, hold=True):

        self.fig = fig
        self.ax = ax
        self.init = init_func
        self.update = update_func
        self.frames = frames
        self.pause = pause
        self.hold = hold

    def animate(self):

        self.init()

        for i in self.frames:

            lines = self.update(i)
            plt.pause(self.pause)

            if i != len(self.frames) - 1:
                self.remove(lines)
                block=False

            else:
                block=self.hold

            plt.show(block=block)

        return None

    def remove(self, lines):

        if not isinstance(lines, (list, tuple)):
            lines = [lines]

        for line in lines:

            if isinstance(line, list):
                line[0].remove()

            else:
                line.remove()

        return None


# Example
def init():

    ax[0].plot(x, lw=3, ls='--', c='k')
    ax[0].set_ylim(0, 11)
    ax[0].set_xlim(0, 30)
    ax[1].set_ylim(30, 0)
    ax[1].set_xlim(0, 30)

    return None


def update(i):

    line1 = ax[0].plot(x[:i+1], c='orange', lw=2)
    line3 = ax[0].axvline(i)
    line2 = ax[1].imshow(y[i], cmap='Greys')

    ax[0].set_title(i)
    ax[1].set_title(i)

    return line1, line2, line3


fig, ax = plt.subplots(2)
fig.tight_layout()

x = np.random.randint(1, 10, 30)
y = np.random.uniform(0, 1, (30, 30, 30))
f = np.arange(x.shape[0])

anim = Animate(fig, ax, init_func=init, update_func=update, frames=f, pause=0.1, hold=False)
anim.animate()
