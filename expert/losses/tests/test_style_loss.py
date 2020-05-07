"""
Tests for style loss functions.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import numpy as np

import torch
import torch.nn as nn

import expert
import expert.losses.style_loss as elsl


class _TestNetwork(nn.Module):
    """
    Network used for testing the style loss functions.

    Simple network with 2 convolutional layers with fixed weights
    """
    def __init__(self):
        super(_TestNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        self.conv1.weight.data = torch.ones_like(self.conv1.weight.data).div(10)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, bias=False)
        self.conv2.weight.data = torch.ones_like(self.conv2.weight.data).div(10)
        self.features = nn.Sequential(self.conv1, self.conv2)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.l1(x)
        x = self.l2(x)
        return x

class TestStyleLoss():
    """
    Tests :class:`expert.losses.style_loss.StyleLoss` class.
    """
    net = _TestNetwork()
    net.eval()
    style_loss = elsl.StyleLoss(net.features, layer_type=nn.Conv2d)

    def test_init(self):
        """
        Tests :class:`expert.losses.style_loss.StyleLoss` class init.
        """
        assert (issubclass(elsl.StyleLoss, nn.Module))
        assert (self.style_loss.__class__.__bases__[0].__name__
                == 'Module')

    def test_forward(self):
        """
        Tests :func:`expert.losses.style_loss.StyleLoss.forward`
        function.
        """
        expert.setup_random_seed()
        x = torch.ones((2, 1, 5, 5))
        y = torch.ones((2, 1, 5, 5)) - 0.5

        true_loss = torch.from_numpy(np.array([[0.2421, 0.0], [0.2421, 0.0]],
                                     dtype=np.float32))
        loss = self.style_loss(x, y, reduce_layer_dim=False, reduce_mean=False)
        assert torch.allclose(loss, true_loss, atol=0.01)

        loss = self.style_loss(x, y, reduce_layer_dim=True, reduce_mean=False)
        true_loss_layer_dim = torch.mean(true_loss, axis=1)
        assert torch.allclose(loss, true_loss_layer_dim, atol=0.01)

        loss = self.style_loss(x, y, reduce_layer_dim=True, reduce_mean=True)
        loss_mean = torch.mean(true_loss)
        assert torch.allclose(loss, loss_mean, atol=0.01)

        lambdas = torch.from_numpy(np.array([0.1, 0.8], dtype=np.float32))
        # change lambda
        self.style_loss.lambdas.data = torch.from_numpy(
            np.array([0.1, 0.8], dtype=np.float32))

        loss = self.style_loss(x, y, reduce_layer_dim=False, reduce_mean=False)
        lambda_loss = torch.from_numpy(np.array([[0.0242, 0.0], [0.0242, 0.0]],
                                       dtype=np.float32))
        assert torch.allclose(loss, lambda_loss, atol=0.01)
