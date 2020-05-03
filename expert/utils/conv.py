"""
The :mod:`expert.utils.conv` module holds util functions for convolutions, like
padding to maintain the size of the image.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['pad']


def pad(im_size, filt_size, stride):
    """
    Returns the amount of padding needed on [height, width] to maintain image
    size.

    This function calculates the amount of padding needed to keep the output
    image shape the same as the input image shape.

    Parameters
    ----------
    im_size : List[int]
        List of [height, width] of the image to pad.
    filt_size : int
        The width of the filter being used in the convolution, assumed to be
        square.
    stride : int
        Amount of stride in the convolution.

    Returns
    -------
    padding : List[int]
        Amount of padding needed for []
    """
    padding = [int(((stride-1)*i-stride+filt_size)/2) for i in im_size]

    # Append lists to each other for use in func:`torch.nn.functional.pad`.
    return padding*2
