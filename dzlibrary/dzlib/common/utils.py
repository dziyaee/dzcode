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
