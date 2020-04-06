"""
Expert
=============
Expert is a Python module that implements models for perceptual experiments
using pytorch. These models can usually be separated into two classes,
deep learning architectrures and traditional perceptual literature models.
Expert aims to bridge the gap in the construction of these.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

from typing import Optional

import sys

# Author and license information
__author__ = 'Alex Hepburn'
__email__ = 'alex.hepburn@bristol.ac.uk'
__license__ = 'new BSD'

# The current package version
__version__ = '0.0.1'

__all__ = ['setup_random_seed']


def setup_random_seed(seed: Optional[int] = None) -> None:
    """
    Set's up Python's, numpy's and torch's random seeds.

    Parameters
    ----------
    seed : integer
        An integer used to seed Python's, numpy's and torch's random number
        generators.

    Raises
    ------
    TypeError
        The ``seed`` input parameter is not an integer.
    """
    import numpy as np
    import random
    import torch

    if seed is None:
        random_seed = int(np.random.uniform() * (2**31 - 1))
    else:
        if isinstance(seed, int):
            random_seed = seed
        else:
            raise TypeError('The seed parameter is not an integer.')

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if torch.backends.cudnn.version() is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
