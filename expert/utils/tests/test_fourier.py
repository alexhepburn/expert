"""
Tests for Fourier utility functions.
TODO: tests for harmonic column
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import pytest

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import expert.utils.fourier as fourier


def test_steer_to_harmonics():
    """
    Tests :func:`expert.utils.fourier.steer_to_harmonics`
    function.
    """

    # All good
    # Test with zero
    harmonics = torch.Tensor([0., 1., 2.])
    angles = torch.arange(0, 3+1) * math.pi / 4
    steer = fourier.steer_to_harmonics(harmonics, angles, phase='sin')
    correct = torch.Tensor([[0.2384, 0.1260, 0.1260, 0.2384],
                            [-0.3371, 0.5289, -0.1782, 0.3700],
                            [-0.8604, 0.7809, -0.9262, 0.8467],
                            [-0.6084, 1.0522, -0.6549, 0.0987],
                            [0.0987, -0.6549, 1.0522, -0.6084]])
    assert torch.allclose(correct, steer, atol=1e-4)

    harmonics = torch.Tensor([1., 2., 3.])
    steer = fourier.steer_to_harmonics(harmonics, angles, phase='cos')
    correct = torch.Tensor([[0.3750, 0.1768, 0.1250, -0.1768],
                            [0.1250, 0.3536, 0.3750, 0.3536],
                            [0.2500, 0.0000, -0.2500, 0.0000],
                            [0.000, 0.2500, 0.0000, -0.2500],
                            [0.3750, -0.1768, 0.1250, 0.1768],
                            [-0.1250, 0.3535, -0.3750, 0.3536]])
    assert torch.allclose(correct, steer, atol=1e-4)

def test_raised_cosine():
    """
    Tests :func:`expert.utils.fourier.raised_cosine` function
    """
    # All good
    X, Y = fourier.raised_cosine(size=10)
    correct_X = torch.Tensor([-1.1, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4,
                                -0.3, -0.2, -0.1, 0.0, 0.1])
    correct_Y = torch.Tensor(
        [0.0000, 0.0000, 0.0245, 0.0955, 0.2061, 0.3455, 0.5, 0.6545,
            0.7939, 0.9045, 0.9755, 1.0, 1.0])
    assert torch.allclose(correct_X, X, atol=1e-4)
    assert torch.allclose(correct_Y, Y, atol=1e-4)

    # Change default values
    X, Y = fourier.raised_cosine(
        width=2, position=1, func_min=-2, func_max=0, size=10)
    correct_X = torch.Tensor([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
                                1.6, 1.8, 2.0, 2.2])
    correct_Y = torch.Tensor(
        [-2.0, -2.0, -1.9511, -1.809, -1.5878, -1.3090, -1.0000, -0.6910,
        -0.4122, -0.1910, -0.0489, 0.0, 0.0])
    assert torch.allclose(correct_X, X, atol=1e-4)
    assert torch.allclose(correct_Y, Y, atol=1e-4)

def test_point_operation_filter():
    """
    Tests :func:`expert.utils.fourier.point_operation_filter` function.
    """
    # All good
    # Simulate using function in Steerable Wavelet to create low pass mask.
    log_rad = torch.Tensor([[0.1781, -0.1610, -0.3219, -0.1610, 0.1781],
                            [-0.1610, -0.8219, -1.3219, -0.8219, -0.1610],
                            [-0.3219, -1.3219, -1.3219, -1.3219, -0.3219],
                            [-0.1610, -0.8219, -1.3219, -0.8219, -0.1610],
                            [0.1781, -0.1610, -0.3219, -0.1610, 0.1781]])
    YIrcos = torch.Tensor([1.0, 1.0, 0.9877, 0.9511, 0.8910, 0.8090, 0.7071,
                           0.5878, 0.4540, 0.3090, 0.1564, 0.0, 0.0])
    origin = -1.10
    increment = 0.1000
    mask = fourier.point_operation_filter(
        log_rad, YIrcos, origin, increment)
    true_mask = torch.Tensor([[0.0, 0.2495, 0.4833, 0.2495, 0.0],
                              [0.2495, 0.9591, 1.0, 0.9591, 0.2495],
                              [0.4833, 1.0, 1.0, 1.0, 0.4833],
                              [0.2495, 0.9591, 1.0, 0.9591, 0.2495],
                              [0.0, 0.2495, 0.4833, 0.2495, 0.0]])
    assert torch.allclose(true_mask, mask, atol=1e-4)

def test_roll_n():
    """
    Tests :func:`expert.utils.fourier.roll_n` function.
    TODO: testing
    """
    return True

def test_fftshift():
    """
    Tests :func:`expert.utils.fourier.fftshift` function.
    TODO: testing
    """
    return True

def test_ifftshift():
    """
    Tests :func:`expert.utils.fourier.ifftshift` function.
    TODO: testing
    """
    return True
