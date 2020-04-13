"""
The :mod:`expert.models` module holds a number of models that have been used
in perception or deep learning. These can either be randomly initialised
or pre-trained weights can be used.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

from expert.models.perceptnet import PerceptNet

__all__ = ['PerceptNet']
