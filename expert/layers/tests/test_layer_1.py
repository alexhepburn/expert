"""
Tests for layer 1
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import torch
import torch.nn as nn

import expert
import expert.layers.layer_1 as ell1


class TestLayer1():
    """
    Tests :class:`expert.layers.layer_1.Layer1` class.
    """
    expert.setup_random_seed()
    layer1 = ell1.Layer1()

    def test_init(self):
        """
        Test :class:`expert.layers.layer_1.Layer1` class init.
        """
        return True

    def test_forward(self):
        """
        Test :func:`expert.layers.layer_1.Layer1.forward` function.
        """
        x = torch.zeros(9, 6)
        x[4, :] = torch.arange(0, 0.6, 0.1)
        [y, y_dn] = self.layer1.forward(x)

        correct_y = torch.ones(9, 6)*1e-6
        correct_y[4, :] = torch.arange(0, 0.6, 0.1)

        correct_y_dn = torch.Tensor([
            [4.0819e-4, 1.9524e-4, 1.1203e-4, 7.5627e-5, 5.6173e-5, 4.4355e-5],
            [4.0819e-4, 1.9524e-4, 1.1203e-4, 7.5627e-5, 5.6173e-5, 4.4355e-5],
            [4.0819e-4, 1.9524e-4, 1.1203e-4, 7.5627e-5, 5.6173e-5, 4.4355e-5],
            [4.0819e-4, 1.9524e-4, 1.1203e-4, 7.5627e-5, 5.6173e-5, 4.4355e-5],
            [4.0819e-4, 183.1477, 202.9864, 212.1136, 218.9164, 225.0190],
            [4.0819e-4, 1.9524e-4, 1.1203e-4, 7.5627e-5, 5.6173e-5, 4.4355e-5],
            [4.0819e-4, 1.9524e-4, 1.1203e-4, 7.5627e-5, 5.6173e-5, 4.4355e-5],
            [4.0819e-4, 1.9524e-4, 1.1203e-4, 7.5627e-5, 5.6173e-5, 4.4355e-5],
            [4.0819e-4, 1.9524e-4, 1.1203e-4, 7.5627e-5, 5.6173e-5, 4.4355e-5]])

        assert torch.allclose(correct_y, y, atol=1e-6)
        correct_y_dn_small = correct_y_dn[[0, 1, 2, 3, 5, 6, 7, 8], :]
        y_dn_small = y_dn[[0, 1, 2, 3, 5, 6, 7, 8], :]
        assert torch.allclose(correct_y_dn_small, y_dn_small, atol=1e-7)
        assert torch.allclose(correct_y_dn[4, :], y_dn[4, :], atol=1e-2)

        x = torch.zeros(9, 6) + 0.125
        x[4, :] = torch.arange(0, 0.6, 0.1)
        [y, y_dn] = self.layer1.forward(x)

        correct_y = torch.ones(9, 6) * 0.125
        correct_y[4, :] = torch.arange(0, 0.6, 0.1)

        correct_y_dn = torch.Tensor([
            [98.3638, 93.9948, 88.3901, 82.7941, 77.5418, 72.7431],
            [98.3638, 93.9948, 88.3901, 82.7941, 77.5418, 72.7431],
            [98.3638, 93.9948, 88.3901, 82.7941, 77.5418, 72.7431],
            [98.3638, 93.9948, 88.3901, 82.7941, 77.5418, 72.7431],
            [3.5350e-5, 76.5588, 130.0170, 162.9414, 185.2633, 201.8940],
            [98.3638, 93.9948, 88.3901, 82.7941, 77.5418, 72.7431],
            [98.3638, 93.9948, 88.3901, 82.7941, 77.5418, 72.7431],
            [98.3638, 93.9948, 88.3901, 82.7941, 77.5418, 72.7431],
            [98.3638, 93.9948, 88.3901, 82.7941, 77.5418, 72.7431]])

        assert torch.allclose(correct_y, y, atol=1e-6)
        assert torch.allclose(correct_y_dn, y_dn, atol=1e-2)


class TestLayer2():
    """
    Tests :class:`expert.layers.layer_1.Layer2` class.
    """
    expert.setup_random_seed()
    layer2 = ell1.Layer2()

    def test_init(self):
        """
        Test :class:`expert.layers.layer_1.Layer2` class init.
        """
        return True

    def test_forward(self):
        """
        Test :func:`expert.layers.layer_1.Layer2.forward` function.
        """
        x = torch.Tensor([54.2545, 54.4370, 52.5493, 66.0231, 67.9172, 72.2044,
                          82.3919, 82.1441, 84.4466, 74.9919, 69.7561,
                          51.4902, 63.9187, 59.8488, 64.1332, 78.2811])
        [y, y_dn] = self.layer2(x)
        correct_y = torch.Tensor([46.2944, 46.0439, 44.1495, 58.0438, 59.5077,
                                  63.3382, 73.5195, 73.7167, 76.0166, 66.1050,
                                  60.8637, 43.0445, 55.9005, 51.3966, 55.6764,
                                  70.2498])
        correct_y_dn = torch.Tensor([1.2196, 1.1993, 1.1497, 1.5283, 1.5493,
                                     1.6296, 1.8913, 1.9183, 1.9781, 1.7000,
                                     1.5649, 1.1196, 1.4704, 1.3366, 1.4478,
                                     1.8472])

        assert torch.allclose(correct_y, y.squeeze(), atol=1e-3)
        assert torch.allclose(correct_y_dn, y_dn.squeeze(), atol=1e-3)


def test_make_2d_gauss_kernel():
    """
    """
    fs = 64
    N = 2
    sigma = 0.0660
    (H, dHds) = ell1.make_2d_gauss_kernel(fs, N, sigma)
    correct_H = torch.Tensor([
        [0.0089, 0.0087, 0.0087, 0.0084],
        [0.0087, 0.0089, 0.0084, 0.0087],
        [0.0087, 0.0084, 0.0089, 0.0087],
        [0.0084, 0.0087, 0.0087, 0.0089]])
    correct_dHds = torch.Tensor([
        [-0.2703, -0.2555, -0.2555, -0.2413],
        [-0.2555, -0.2703, -0.2413, -0.2555],
        [-0.2555, -0.2413, -0.2703, -0.2555],
        [-0.2413, -0.2555, -0.2555, -0.2703]])
    assert torch.allclose(correct_H, H, atol=1e-4)
    assert torch.allclose(correct_dHds, dHds, atol=1e-4)


class TestLayer3():
    """
    Tests :class:`expert.layers.layer_1.Layer3` class.
    """
    expert.setup_random_seed()
    layer3 = ell1.Layer3()

    def test_init(self):
        """
        Test :class:`expert.layers.layer_1.Layer3` class init.
        """
        return True

    def test_forward(self):
        """
        Test :func:`expert.layers.layer_1.Layer3.forward` function.
        """
        x = torch.Tensor([1.5060, 1.6206, 1.4715, 1.6843, 1.4584, 1.4719,
                          1.5168, 1.7102, 1.9397, 1.8849, 1.8148, 1.7293,
                          1.4438, 1.4452, 1.3708, 1.5931]).unsqueeze(1)
        [y, y_dn] = self.layer3(x)
        correct_y = torch.Tensor([1.5195, 1.5270, 1.5604, 1.6281, 1.5977,
                                  1.5977, 1.6018, 1.6437, 1.6722, 1.6785,
                                  1.6869, 1.7224, 1.5938, 1.5659, 1.5527,
                                  1.6013]).unsqueeze(1)
        correct_y_dn = torch.Tensor([2.1141, 1.6176, 1.6502, 2.2674, 1.7090,
                                     1.2973, 1.2899, 1.7375, 1.8055, 1.3816,
                                     1.3827, 1.8530, 2.1958, 1.6351, 1.6074,
                                     2.1848]).unsqueeze(1)
        assert torch.allclose(correct_y, y, atol=1e-3)
        assert torch.allclose(correct_y_dn, y_dn, atol=1e-3)


class TestLayer4():
    """
    Tests :class:`expert.layers.layer_1.Layer4` class.
    """
    expert.setup_random_seed()
    layer4 = ell1.Layer4()

    def test_init(self):
        """
        Test :class:`expert.layers.layer_1.Layer4` class init.
        """
        return True

    def test_forward(self):
        """
        Test :func:`expert.layers.layer_1.Layer4.forward` function.
        """
        assert True


def test_make_wavelet_kernel_2():
    """
    TODO: testing
    """
    w, ind = ell1.make_wavelet_kernel_2(16, 2, 4, 1, 0, 1)
    print(ind)
    sys.exit(0)


def test_kernel_s_wavelet_spatial():
    """
    """
    ind = [[16, 16]]*5 + [[8, 8]]*4 + [[4, 4]]
    H_spatial = ell1.kernel_s_wavelet_spatial(ind, 64, [0.24])
    true_upper_right = torch.Tensor([
        [0.6746, 0.6732, 0.6689, 0.6618, 0.6521, 0.6398, 0.6250, 0.6080,
         0.5890],
        [0.6732, 0.6746, 0.6732, 0.6689, 0.6618, 0.6521, 0.6398, 0.6250,
         0.6080],
        [0.6689, 0.6732, 0.6746, 0.6732, 0.6689, 0.6618, 0.6521, 0.6398,
         0.6250],
        [0.6618, 0.6689, 0.6732, 0.6746, 0.6732, 0.6689, 0.6618, 0.6521,
         0.6398],
        [0.6521, 0.6618, 0.6689, 0.6732, 0.6746, 0.6732, 0.6689, 0.6618,
         0.6521],
        [0.6398, 0.6521, 0.6618, 0.6689, 0.6732, 0.6746, 0.6732, 0.6689,
         0.6618],
        [0.6250, 0.6398, 0.6521, 0.6618, 0.6689, 0.6732, 0.6746, 0.6732,
         0.6689],
        [0.6080, 0.6250, 0.6398, 0.6521, 0.6618, 0.6689, 0.6732, 0.6746,
         0.6732],
        [0.5890, 0.6080, 0.6250, 0.6398, 0.6521, 0.6618, 0.6689, 0.6732,
         0.6746]])
    true_bottom_right = torch.Tensor([
        [0.0108, 0.0077, 0.0091, 0.0101, 0.0104, 0.0069, 0.0082, 0.0091,
         0.0094],
        [0.0077, 0.0108, 0.0104, 0.0094, 0.0080, 0.0104, 0.0101, 0.0091,
         0.0077],
        [0.0091, 0.0104, 0.0108, 0.0104, 0.0094, 0.0101, 0.0104, 0.0101,
         0.0091],
        [0.0101, 0.0094, 0.0104, 0.0108, 0.0104, 0.0091, 0.0101, 0.0104,
         0.0101],
        [0.0104, 0.0080, 0.0094, 0.0104, 0.0108, 0.0077, 0.0091, 0.0101,
         0.0104],
        [0.0069, 0.0104, 0.0101, 0.0091, 0.0077, 0.0108, 0.0104, 0.0094,
         0.0080],
        [0.0082, 0.0101, 0.0104, 0.0101, 0.0091, 0.0104, 0.0108, 0.0104,
         0.0094],
        [0.0091, 0.0091, 0.0101, 0.0104, 0.0101, 0.0094, 0.0104, 0.0108,
         0.0104],
        [0.0094, 0.0077, 0.0091, 0.0101, 0.0104, 0.0080, 0.0094, 0.0104,
         0.0108]])
    assert torch.allclose(true_upper_right, H_spatial[0:9, 0:9], atol=1e-3)
    assert torch.allclose(true_bottom_right, H_spatial[-9:, -9:], atol=1e-3)


def test_spatio_temp_freq_domain():
    """
    """
    (x, y, t, fx, fy, ft) = ell1.spatio_temp_freq_domain(10, 10, 1, 64, 64, 1)
    correct_x = torch.Tensor([0, 0.0156, 0.0313, 0.0469, 0.0625, 0.0781, 0.0938,
                              0.1094, 0.1250, 0.1406])
    correct_x = correct_x.repeat(10, 1)
    correct_y = correct_x.T
    correct_t = torch.zeros(10, 10)
    correct_fx = torch.Tensor([-32, -25.6, -19.2, -12.8, -6.4, 0, 6.4, 12.8,
                               19.2, 25.6])
    correct_fx = correct_fx.repeat(10, 1)
    correct_fy = correct_fx.T
    correct_ft = torch.zeros(10, 10)

    assert torch.allclose(correct_x, x, atol=1e-4)
    assert torch.allclose(correct_y, y, atol=1e-4)
    assert torch.allclose(correct_t, t, atol=1e-4)
    assert torch.allclose(correct_fx, fx, atol=1e-4)
    assert torch.allclose(correct_fy, fy, atol=1e-4)
    assert torch.allclose(correct_ft, ft, atol=1e-4)


def test_metefot():
    """
    """
    sec = torch.zeros(10, 10)
    fot_x = torch.Tensor([0, 0.0156, 0.0313, 0.0469, 0.0625, 0.0781, 0.0938,
                          0.1094, 0.1250, 0.1406])
    foto = fot_x.repeat(10, 1)
    N = 1
    ma = 1
    sec = ell1.metefot(sec, foto, N, ma)

    correct_sec = torch.allclose(foto, sec, atol=1e-4)


def test_freqspace():
    """
    """
    N = [10, 10]
    flag = 'meshgrid'
    [fx, fy] = ell1.freqspace(N, flag)

    correct_fx = torch.Tensor([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6,
                               0.8])
    correct_fx = correct_fx.repeat(N[0], 1)
    correct_fy = torch.Tensor([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6,
                               0.8])
    correct_fy = correct_fy.repeat(N[1], 1).T

    assert torch.allclose(correct_fx, fx, atol=1e-4)
    assert torch.allclose(correct_fy, fy, atol=1e-4)


def test_make_csf_kernel():
    """
    """
    hh, gn = ell1.make_csf_kernel(64, 4, 1, 1)
    correct_hh = torch.Tensor([
        [0.4194, 0.2456, 0., 0., 0.2456, 0.0895, 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.],
        [0.1840, 0.3141, 0.1840, 0., 0.0670, 0.1840, 0.0670, 0., 0., 0., 0.,
         0., 0., 0., 0., 0.],
        [0.0829, 0.1732, 0.2958, 0.1732, -0.025, 0.0631, 0.1732, 0.0631, 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0.1085, 0.2268, 0.3873, 0., -0.0321, 0.0826, 0.2268, 0., 0., 0.,
         0., 0., 0., 0., 0.],
        [0.1840, 0.0670, 0., 0., 0.3141, 0.1840, 0., 0., 0.1840, 0.0670, 0.,
         0., 0., 0., 0., 0.],
        [0.0508, 0.1396, 0.0508, 0., 0.1396, 0.2383, 0.1396, 0., 0.0508,
         0.1396, 0.0508, 0., 0., 0., 0., 0.],
        [-0.0192, 0.0495, 0.1359, 0.0495, 0.0650, 0.1359, 0.2320, 0.1359,
         -0.0192, 0.0495, 0.1359, 0.0495, 0., 0., 0., 0.],
        [0., -0.0251, 0.0647, 0.1776, 0., 0.0849, 0.1776, 0.3032, 0., -0.0251,
         0.064, 0.1776, 0., 0., 0., 0.],
        [0.0829, -0.0245, 0., 0., 0.1732, 0.0631, 0., 0., 0.2958, 0.1732, 0.,
         0., 0.1732, 0.0631, 0., 0.],
        [-0.0192, 0.0650, -0.0192, 0., 0.0495, 0.1359, 0.0495, 0., 0.1359,
         0.2320, 0.1359, 0., 0.0495, 0.1359, 0.0495, 0.],
        [-0.1002, -0.0206, 0.0697, -0.0206, -0.0206, 0.0530, 0.1456, 0.0530,
         0.0697, 0.1456, 0.2487, 0.1456, -0.0206, 0.0530, 0.1456, 0.0530],
        [0., -0.1303, -0.0268, 0.0906, 0., -0.0268, 0.0690, 0.1894, 0., 0.0906,
         0.1894, 0.3234, 0., -0.0268, 0.0690, 0.1894],
        [0., 0., 0., 0., 0.1085, -0.0321, 0., 0., 0.2268, 0.0826, 0., 0.,
         0.3873, 0.2268, 0., 0.],
        [0., 0., 0., 0., -0.0251, 0.0849, -0.0251, 0., 0.0647, 0.1776, 0.0647,
         0., 0.1776, 0.3032, 0.1776, 0.],
        [0., 0., 0., 0., -0.1303, -0.0268, 0.0906, -0.0268, -0.0268, 0.0690,
         0.1894, 0.0690, 0.0906, 0.1894, 0.3234, 0.1894],
        [0., 0., 0., 0., 0., -0.1695, -0.0349, 0.1179, 0., -0.0349, 0.0898,
         0.2465, 0., 0.1179, 0.2465, 0.4209]])
    correct_gn = torch.Tensor([[0.0530, 0.1456, 0.0530, -0.0206],
                               [0.1456, 0.2487, 0.1456, 0.0697],
                               [0.0530, 0.1456, 0.0530, -0.0206],
                               [-0.0206, 0.0697, -0.0206, -0.1002]])
    assert torch.allclose(correct_hh, hh, atol=1e-3)
    assert torch.allclose(correct_gn, gn, atol=1e-3)


def test_csfsso():
    """
    """
    (h, cssfo, csft, oe) = ell1.csfsso(64, 4, 330.74, 7.28, 0.837, 1.908, 1,
                                       6.664)
    correct_h = torch.Tensor([[2.8594, 7.8499, 2.8594, -1.1117],
                              [7.8499, 13.4051, 7.8499, 3.7551],
                              [2.8594, 7.8499, 2.8594, -1.1117],
                              [-1.1117, 3.7551, -1.1117, -5.3994]])
    correct_cssfo = torch.Tensor([[0.0007, 0.8812, 4.0785, 0.8812],
                                  [0.8812, 0.4954, 36.7277, 0.4954],
                                  [4.0785, 36.7277, 53.9061, 36.7277],
                                  [0.8812, 0.4954, 36.7277, 0.4954]])
    correct_csft = torch.Tensor([[0.6603, 2.4276, 4.0785, 2.4276],
                                 [2.4276, 14.7785, 36.7277, 14.7785],
                                 [4.0785, 36.7277, 53.9061, 36.7277],
                                 [2.4276, 14.7785, 36.7277, 14.7785]])
    correct_oe = torch.Tensor([[0.0011, 0.3630, 1.0000, 0.3630],
                               [0.3630, 0.0335, 1.0000, 0.0335],
                               [1.0000, 1.0000, 1.0000, 1.0000],
                               [0.3630, 0.0335, 1.0000, 0.0335]])
    assert torch.allclose(correct_h, h, atol=1e-3)
    assert torch.allclose(correct_cssfo, cssfo, atol=1e-3)
    assert torch.allclose(correct_csft, csft, atol=1e-3)
    assert torch.allclose(correct_oe, oe, atol=1e-3)


def test_fsamp2():
    """
    """
    f1 = torch.Tensor([[7.4217e-4, 0.8812, 4.0785, 0.8812],
                       [0.8812, 0.4954, 36.7277, 0.4954],
                       [4.0785, 36.7277, 53.9061, 36.7277],
                       [0.8812, 0.4954, 36.7277, 0.4954]])
    h = ell1.fsamp2(f1)
    correct_h = torch.Tensor([[2.8594, 7.8499, 2.8594, -1.1117],
                              [7.8499, 13.4051, 7.8499, 3.7551],
                              [2.8594, 7.8499, 2.8594, -1.1117],
                              [-1.1117, 3.7551, -1.1117, -5.3994]])
    assert torch.allclose(correct_h, h, atol=1e-3)


def test_convmtx2():
    """
    """
    kernel = torch.Tensor([[1., 2.], [3., 4.]])
    N = 3
    convMat, blockMat, convVec = ell1.convmtx2(kernel, N, N)

    correct_convMat = torch.Tensor([
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [3., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 3., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 3., 0., 0., 0., 0., 0., 0.],
        [2., 0., 0., 1., 0., 0., 0., 0., 0.],
        [4., 2., 0., 3., 1., 0., 0., 0., 0.],
        [0., 4., 2., 0., 3., 1., 0., 0., 0.],
        [0., 0., 4., 0., 0., 3., 0., 0., 0.],
        [0., 0., 0., 2., 0., 0., 1., 0., 0.],
        [0., 0., 0., 4., 2., 0., 3., 1., 0.],
        [0., 0., 0., 0., 4., 2., 0., 3., 1.],
        [0., 0., 0., 0., 0., 4., 0., 0., 3.],
        [0., 0., 0., 0., 0., 0., 2., 0., 0.],
        [0., 0., 0., 0., 0., 0., 4., 2., 0.],
        [0., 0., 0., 0., 0., 0., 0., 4., 2.],
        [0., 0., 0., 0., 0., 0., 0., 0., 4.]])
    assert torch.allclose(correct_convMat, convMat, atol=1e-1)
