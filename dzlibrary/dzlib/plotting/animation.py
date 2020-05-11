import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")


def janimate(fig, plot_list, method_list, kwargs_list, remove=True):
    """Function to animate matplotlib plots in Jupyter Notebooks. The function takes as arguments a figure object to update, a list of objects to plot, a list of bound axis methods, a list of kwargs for each respective axis, and a remove boolean to clear the axis objects each frame which speeds up performance. The remove boolean can be set to False on the last animation iteration in order to freeze the last frame on the figure.
    Note: The magic command "%matplotlib notebook" must be used for this to work.

    Args:
        fig (matplotlib figure): the figure object to update
        plot_list (list): list of objects to plot
        method_list (list): list of axis-bound methods used to plot each respective plot object
        kwargs_list (list): list of kwargs to pass to each axis-bound method call
        remove (bool, optional): toggle the removal of all axis objects at the end of the frame

    Returns:
        None
    """
    # Call each bound method on each plot object with each set of kwargs if available and append to a lines list
    lines = []
    for plot, method, kwargs in zip(plot_list, method_list, kwargs_list):
        if kwargs is not None:
            lines.append(method(plot, **kwargs))
        else:
            lines.append(method(plot))

    # Update figure
    fig.canvas.draw()

    # Iterate through lines list and remove objects if remove is True
    if remove:
        for line in lines:
            if isinstance(line, list):
                line[0].remove()
            else:
                line.remove()
    return None


class Animate():
    '''Class to animate matplotlib plots

    Attributes:
        ax (matplotlib axis): axis to update
        fig (matplotlib figure): figure to update
        frames (int): number of frames in animation
        hold (bool): toggle the removal of the last frame
        init (user-created function): function to set up the initial and/or unchanging figure and axis objects
        pause (float): time (in seconds) to wait between each frame
        update (user-created function): function to call when updating each frame. Takes the frame variable as an input argument to be used as the iterator variable. This function *must* return all axis objects that are intended to be updated (example provided in animation.py)
    '''

    def __init__(self, fig, ax, frames, update_func, init_func=None, pause=0.1, hold=True):
        '''
        Animate instance is initialized with the following:

        Args:
            fig: matplotlib figure object

            ax: matplotlib axis object

            frames: iteration variable used with the update function 'update_func'

            update_func: user-defined function that takes the iteration variable 'frames'. This variable can be used to update any matplotlib attributes or methods within the function. The function should return any matplotlib axis objects that the user desired to be removed per update. Doing this will help ensure that the animation does not slow down.

            init_func: user-defined function that takes no input arguments and returns None. Within this function, define the starting and/or static matplotlib axis objects

            pause: time (in seconds) to wait between each frame

            hold: boolean to control whether or not the last frame is held by passing block=True to plt.show()
        '''

        self.fig = fig
        self.ax = ax
        self.init = init_func
        self.update = update_func
        self.frames = frames
        self.pause = pause
        self.hold = hold

    def animate(self):
        """Method to begin the animation using arguments provided during instantiation of Animate

        Returns:
            None
        """
        # Call init function if available
        if self.init:
            self.init()

        # Iterate through frames
        for i in self.frames:

            # Break out of animation loop if figure no longer exists (closed by user)
            if not plt.get_fignums():
                break

            # Call update function then pause
            lines = self.update(i)
            plt.pause(self.pause)

            # Remove method is called if not on last frame
            if (i != len(self.frames) - 1) and (lines is not None):
                self._remove(lines)
                block=False

            else:
                # If on last frame, hold is checked to toggle figure blocking on or off
                block=self.hold

            plt.show(block=block)

        return None

    def _remove(self, lines):
        """Method to remove all axis objects stored in the lines variable. lines may be a list or tuple of multiple axis objects, a list of a single axis object, or a single axis object.

        Args:
            lines (list, tuple, axis object): a single or collection of axis objects

        Returns:
            None
        """
        if not isinstance(lines, (list, tuple)):
            lines = [lines]

        for line in lines:
            if isinstance(line, list):
                line[0].remove()

            else:
                line.remove()
        return None


# Example
if __name__ == "__main__":

    def init():

        # Axis 0 static settings / objects
        ax[0].plot(x, lw=3, ls='--', c='k')
        ax[0].set_ylim(0, 11)
        ax[0].set_xlim(0, 30)

        # Axis 1 static settings / objects
        ax[1].set_ylim(30, 0)
        ax[1].set_xlim(0, 30)

        return None

    def update(i):

        # Axis 0 dynamic settings / objects
        line1 = ax[0].plot(x[: i + 1], c='orange', lw=2)
        line2 = ax[0].axvline(i)
        ax[0].set_title(i)

        # Axis 1 dynamic settings / objects
        line3 = ax[1].imshow(y[i], cmap='Greys')
        ax[1].set_title(i)

        return line1, line2, line3

    fig, ax = plt.subplots(2)
    fig.tight_layout()

    x = np.random.randint(1, 10, 30)
    y = np.random.uniform(0, 1, (30, 30, 30))
    f = np.arange(x.shape[0])

    anim = Animate(fig, ax, init_func=init, update_func=update, frames=f, pause=0.1, hold=False)
    anim.animate()
