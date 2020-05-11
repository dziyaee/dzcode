import numpy as np
import torch
import matplotlib.pyplot as plt


def npshow(x, ax):
    '''Function to convert and imshow a numpy array. Expected input shape is (C x H x W) or (H x W), which will be converted to (H x W x C) to make it suitable for imshow. C must be 3 or 1

    Args:
        x (Numpy Array or Torch Tensor): x must be of shape (3 x H x W), (1 x H x W), or (H x W)
        ax (matplotlib axis object): axis on which to call the .imshow() method

    Returns:
        ax (matplotlib axis object)

    Raises:
        Exception: x must be of type numpy.ndarray or torch.Tensor
        Exception: x must be of shape (3 x H x W), (1 x H x W), or (H x W)
    '''

    if isinstance(x, torch.Tensor):
        x = np.asarray(x)

    if isinstance(x, np.ndarray):

        # Input shape must be (C x H x W), where C = 3 or 1
        if (x.ndim == 3) and (x.shape[0] == 3 or x.shape[0] == 1):

            # Change input shape to (H x W x C), where C = 3 or 1
            x = x.transpose(1, 2, 0)

        # Input shape must be (H x W)
        elif x.ndim == 2:

            # Change input shape to (H x W x 1)
            x = x.reshape(x.shape[0], x.shape[1], 1)

        else:
            raise Exception(f"Expected array of shape (3 x H x W), (1 x H x W) or (H x W), got {x.shape} instead")

    else:
        raise Exception(f"Expected input of type numpy.ndarray or torch.Tensor, got {type(x)} instead")

    ax.imshow(x)

    return ax
