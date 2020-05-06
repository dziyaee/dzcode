import numpy as np
import torch
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")


# Numpy / Torch Stuff
def info(x, var_name="var"):
    ''' Helper function to print useful information (dtype, shape) of a numpy or pytorch array.
    '''

    try:
        print(f"\n{var_name} info:")
        print(f"dtype: {x.dtype}")
        print(f"shape: {x.shape}")

    except Exception as ex:
        print(ex)

    return None


def stats(x, var_name="var", ddof=1):
    ''' Helper function to print statistics (max, min, mean, std) of an array after converting it to numpy. For std, torch.std() uses unbiased estimation as default. In order to mimic this behavior with numpy.std(), input arg ddof is set to 1 by default.
    '''

    try:
        x = np.asarray(x).astype(np.float32)

        print(f"\n{var_name} stats:")
        print(f"max:  {np.max(x):.4f}")
        print(f"min:  {np.min(x):.4f}")
        print(f"mean: {np.mean(x):.4f}")
        print(f"std:  {np.std(x, ddof=ddof):.4f}")

    except Exception as ex:
        print(ex)

    return None


# nn.Modules Stuff



def calc_padding(x_dim, k_dim, stride, depth):
    x, k, s, d = x_dim, k_dim, stride, depth
    sd = s ** d

    xmin_spacing = (sd - 1) / (s - 1)
    xmin = 1 + (k - 1) * xmin_spacing

    n = np.ceil((x - xmin) / sd)
    validx = xmin + n * sd

    newx = max(xmin, validx)
    padding = int(newx - x)

    return padding


def final_dim(x, k, s, n):
    """Formula to calculate the final output size "y" after "n" valid convolution operations defined by a kernel size "k" and a stride "s" on an input of size "x":

    y = (x + (s - k) * Sum from (d = 0) to (d = n-1) of s^d) / s^n

    Args:
        x (int): Initial input size
        k (int): kernel size
        s (int): stride
        n (int): number of convolution operations

    Returns:
        y (float): Final output size

        Note:
            floor(y) will give the actual output size after valid convolutions
            ceil(y) will give the next valid output size and can be used to calculate the next valid input size
    """
    # Formula to calculate the sum of powers, defined as sum from i=0 to i=n-1 of s^i
    sum_of_powers = lambda s, n: np.sum(s ** np.arange(0, n))
    sop = sum_of_powers(s, n)
    y = (x + (s - k) * (sop)) / s ** n

    return y

def valid_dim(y, k, s, n):
    """Formula to calculate the initial input size "x" that will result in a final output size "y" after "n" valid convolution operations defined by a kernel size "k" and a stride "s". A valid convolution operation is one where no padding is added to the input along any dimension.

    For n = 1:
        x = (s^n * (y - 1)) + (k * Sum from (d = 0) to (d = n-1) of s^d)

    For n > 1:
        x = (s^n * (y - 1)) + (k * Sum from (d = 0) to (d = n-1) of s^d) - (s * Sum from (d = 0) to (d = n-2) of s^d)

    Args:
        y (int): final dimension y. If not a whole number, ceiling(y) will result in the next valid integer y being used for the calculation. This will give the next valid x
        k (int): kernel size
        s (int): stride
        n (int): number of valid convolution operations

    Returns:
        x (int): first valid initial input size x
    """

    # Formula to calculate the sum of powers, defined as sum from i=0 to i=n-1 of s^i
    sum_of_powers = lambda s, n: np.sum(s ** np.arange(0, n))
    sop1 = sum_of_powers(s, n)

    if n == 1:
        x = ((s ** n) * (y - 1)) + (k * sop1) - (s*0)
    else:
        sop2 = sum_of_powers(s, n-1)
        x = ((s**n) * (y - 1)) + (k * sop1) - (s * sop2)

    return x



# Matplotlib Stuff



def npshow(x, ax):
    ''' Function to convert and imshow a numpy array. Expected input shape is (C x H x W) or (H x W), which will be converted to (H x W x C) to make it suitable for imshow. C must be 3 or 1'''

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
        raise Exception("Expected input of type numpy.ndarray or torch.Tensor, got {type(x)} instead")

    ax.imshow(x)

    return ax


# OS Stuff
def get_filenames(data_dir, extension):
    # Fix starting dot for extension
    if extension.startswith('.'):
        extension = extension[1:]
    # sorted list of filenames ending with extension, if blank, returns all files (and folders)
    filenames = sorted([fn for fn in os.listdir(data_dir) if fn.endswith('.'+extension)])
    return filenames

def make_filepaths(data_dir, filenames):
    # Fix starting & ending slashes for filenames and data_dir
    filenames = [fn[1:] if fn.startswith('/') else fn for fn in filenames]
    if not data_dir.endswith('/'):
        data_dir += '/'

    # append data_dir to filenames
    filepaths = [data_dir + fn for fn in filenames]
    return filepaths


# PIL Stuff
def ccrop_pil(image_pil, factor):
    '''
    Function to center crop a PIL image from (W x H) to (W' x H'), where each new dimension is the next smallest integer such that a given number is a factor of it as follows: W' mod number = 0 and H' mod number = 0.

    Args:
        image_pil: PIL Image
        factor: int, integer that should be a factor of the new dimensions W' and H'

    Returns:
        image_pil: center-cropped PIL Image
    '''

    # factor needs to be an int
    assert isinstance(factor, int)

    # PIL Image gives size as (W x H). Also important to note that PIL Image is structured (left > right) and (top > bottom)
    width, height = image_pil.size

    # Starting indices for center crop height / width respectively
    left = (width % factor) / 2
    top = (height % factor) / 2

    # Ending indices for center crop height / width respectively
    right = width - left
    bot = height - top

    # Convert all indices to ints for valid indices
    left, top, right, bot  = int(left), int(top), int(right), int(bot)

    # Center Crop
    image_pil = image_pil.crop((left, top, right, bot))

    return image_pil


def resize_pil(image_pil, factor, resample=Image.NEAREST):
    '''
    Function to resize a PIL Image from (W x H) to (W' x H') by a factor f, such that W' = W*f, and H' = H*f.

    Args:
        image_pil: PIL Image
        factor: int, integer to scale the old dimensions W and H by
        resample: Image.resize resample argument: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS

    Returns:
        image_pil: resized PIL Image
    '''

    # Convert old dimensions to new integer dimensions
    width, height = image_pil.size
    width = int(width * factor)
    height = int(height * factor)

    # Resize
    image_pil = image_pil.resize((width, height), resample)

    return image_pil




