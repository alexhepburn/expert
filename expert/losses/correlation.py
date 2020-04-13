"""
The :mod:`expert.losses.correlation` implements loss functions that are
some form of correlation.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import torch
import torch.nn as nn

__all__ = ['PearsonCorrelation']


class PearsonCorrelation(nn.Module):
    """
    Module for measuring Pearson correlation.
    Given samples (x, y), the Pearson correlation coefficient is given by:
    .. math::
        r = \frac{{}\sum_{i=1}^{n} (x_i - \overline{x})(y_i - \overline{y})}
        {\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2(y_i - \overline{y})^2}}
    """
    def __init__(self) -> None:
        super(PearsonCorrelation, self).__init__()

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        Override the forward function in nn.Modules. Performs Pearson
        Correlation calculation between ``score`` and ``mos``

        Parameters
        ----------
        x : torch.Tensor
            First input vector.
        y : torch.Tensor
            Second input vector.
        TODO: dimensions of the input data have to agree, have to be batch wise
              and check for NaN values in the result.
        """
        x_n = x - torch.mean(x)
        y_n = y - torch.mean(y)
        x_n_norm = torch.sqrt(torch.mean(x_n**2))
        y_n_norm = torch.sqrt(torch.mean(y_n**2))
        normaliser = x_n_norm * y_n_norm
        return -torch.mean(x_n*y_n.squeeze(), dim=0) / normaliser
