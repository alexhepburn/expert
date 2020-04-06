"""
Functions and classes for correlation loss functions.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import numpy as np

import torch
import torch.nn as nn

import expert
import expert.losses.correlation as ec


class TestPearsonCorrelation():
    """
    Tests :class:`expert.losses.correlation.PearsonCorrelation` class.
    """
    correlation = ec.PearsonCorrelation()

    def test_init(self):
        """
        Tests :class:`expert.losses.correlation.PearsonCorrelation` class init.
        """
        assert (issubclass(ec.PearsonCorrelation, nn.Module))
        assert (self.correlation.__class__.__bases__[0].__name__
                == 'Module')

    def test_forward(self):
        """
        Tests :func:`expert.losses.correlation.PearsonCorrelation.forward`
        function.

        TODO: test dimensions of the input data.
        """
        expert.setup_random_seed()

        true_results = torch.from_numpy(np.array([[[0.2445, -0.2465],
                                                  [-2.1353, -1.0341]]],
                                                  dtype=np.float32))

        ones_1 = torch.rand((1, 1, 2, 2), dtype=torch.float)
        ones_2 = torch.rand((1, 1, 2, 2), dtype=torch.float)
        ones_3 = torch.squeeze(ones_2)

        corr_12 = self.correlation(ones_1, ones_2)
        corr_13 = self.correlation(ones_1, ones_3)

        assert torch.allclose(corr_12, true_results, atol=0.1)
        assert torch.allclose(corr_13, true_results, atol=0.1)
