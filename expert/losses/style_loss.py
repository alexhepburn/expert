"""
The :mod:`expert.losses.style_loss` implements loss functions that are
inspired by the style loss.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StyleLoss']


class StyleLoss(nn.Module):
    """
    Module for measuring the style loss with learnable lambda parameters.

    TODO: insert citation

    Parameters
    ----------
    network : torch.nn.Sequential
        A torch.nn.Sequential object that contains the layers in a network.
    layer_type : torch.nn.Module, optional (default=torch.nn.Conv2d)
        What type of layers will be used in the style loss computation. This is
        in order to avoid taking the style loss between a convolutional layer,
        and also the next layer which is the same convolutional layer with a
        relu activation applied.
    initial_lambda : float, optional
        The initial lambda to use to multiply each gram matrix for each layer.

    Raises
    ------

    Attributes
    ----------

    """
    def __init__(self,
                 network: nn.Module,
                 layer_type: nn.Module = nn.Conv2d,
                 initial_lambda: float = 1.0) -> None:
        super(StyleLoss, self).__init__()
        self.network = network
        self.layer_type = layer_type

        # How many parameters needed
        self.number_lambdas = 0
        for layer in self.network:
            if isinstance(layer, self.layer_type):
                self.number_lambdas += 1

        # Initialies lambda parameters
        self.lambdas = nn.Parameter(
            torch.ones([self.number_lambdas], dtype=torch.float32))

    def gram_matrix(self, x):
        """
        Calculates gram matrix for ``x`` per image in batch.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of images to calculate gram matrix for.

        Returns
        -------
        gram : torch.Tensor
            Gram matrix of ``x``.
        """
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        G = torch.bmm(features, torch.transpose(features, 1, 2))
        gram = G.div(b * c * d)
        return gram

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                reduce_layer_dim: bool = True,
                reduce_mean: bool = True) -> torch.Tensor:
        """
        Calculates style loss between ``x`` and ``y``

        Parameters
        ----------
        x : torch.Tensor
            First input tensor.
        y : torch.Tensor
            Second input tensor.
        reduce_layer_dim : boolean (optional, default=True)
            Whether to take the mean over the layer loss for each image in
            batch, if ``True`` then the loss will be averaged over the
            layer axis.
        reduce_mean : boolean (optional, default=True)
            If ``True``, the mean of the loss will be returned.

        Returns
        -------
        style : torch.Tensor
            Tensor containing style losses.
        """
        styles = []
        for layer in self.network:
            x, y = layer(x), layer(y)
            if isinstance(layer, self.layer_type):
                G1, G2 = self.gram_matrix(x), self.gram_matrix(y)
                styles.append(F.mse_loss(
                    G1, G2, reduction='none').mean(axis=(1, 2)))
        style = torch.stack(styles).transpose(0, 1)
        style = style * self.lambdas
        if reduce_layer_dim:
            style = torch.mean(style, axis=1)
        if reduce_mean:
            style = torch.mean(style)
        return style
