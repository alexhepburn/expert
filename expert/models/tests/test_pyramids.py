"""
Tests for pyramid based models
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import pytest

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import expert
import expert.models.pyramids as emp
import expert.utils.pyramid_filters as filt_utils

TEST_INPUT = torch.ones((1, 1, 5, 5))
TEST_INPUT = F.pad(TEST_INPUT, [5, 5, 5, 5], value=0.5)


class TestSteerableWavelet():
    """
    Tests :class:`expert.models.pyramids.SteerableWavelet` class.
    """
    expert.setup_random_seed()
    sw = emp.SteerableWavelet(stages=2, order=3, twidth=1)
    sw_small = emp.SteerableWavelet(stages=1, order=2, twidth=1)

    def test_init(self):
        """
        Tests :class:`expert.models.pyramids.SteerableWavelet` class init.
        """
        assert (issubclass(emp.SteerableWavelet, nn.Module))
        assert (self.sw.__class__.__bases__[0].__name__ == 'Module')

        assert self.sw.stages == 2
        assert self.sw.num_orientations == 4
        assert self.sw.twidth == 1

        harmonics = torch.Tensor([1., 3.])
        angles = torch.Tensor([0.0000, 0.7854, 1.5708, 2.3562])
        assert torch.allclose(self.sw.harmonics, harmonics)
        assert torch.allclose(self.sw.angles, angles)

    def test_meshgrid_angle(self):
        """
        Tests :func:`expert.models.pyramids.SteerableWavelet.meshgrid_angle`
        function.
        """

        # All good
        # Dims 2x2
        dims = [2, 2]
        angle, log_rad = self.sw.meshgrid_angle(dims=dims)
        correct_angle = torch.Tensor([[-2.3562, -1.5708], [3.1416, 0.0]])
        correct_log_rad = torch.Tensor([[0.5, 0.0], [0.0, 0.0]])
        assert torch.allclose(angle, correct_angle, atol=1e-4)
        assert torch.allclose(log_rad, correct_log_rad, atol=1e-4)

        # Dims 5x5
        dims = [5, 5]
        angle, log_rad = self.sw.meshgrid_angle(dims=dims)
        correct_angle = torch.Tensor([
            [-2.3562, -2.0344, -1.5708, -1.1071, -0.7854],
            [-2.6779, -2.3562, -1.5708, -0.7854, -0.4636],
            [3.1416, 3.1416, 0.0, 0.0, 0.0],
            [2.6779, 2.3562, 1.5708, 0.7854, 0.4636],
            [2.3562, 2.0344, 1.5708, 1.1071, 0.7854]])
        correct_log_rad = torch.Tensor([
            [0.1781, -0.1610, -0.3219, -0.1610, 0.1781],
            [-0.1610, -0.8219, -1.3219, -0.8219, -0.1610],
            [-0.3219, -1.3219, -1.3219, -1.3219, -0.3219],
            [-0.1610, -0.8219, -1.3219, -0.8219, -0.1610],
            [0.1781, -0.1610, -0.3219, -0.1610, 0.1781]])
        assert torch.allclose(angle, correct_angle, atol=1e-4)
        assert torch.allclose(log_rad, correct_log_rad, atol=1e-4)

    def test_validate_input(self):
        """
        Tests :func:`expert.models.pyramids.SteerableWavelet._validate_input`
        function.
        """
        return True

    def test_check_height(self):
        """
        Tests :func:`expert.models.pyramids.SteerableWavelet._check_height`
        function.
        """
        dims_msg = ('Input maximum number of stages is %d but number of '
                    'pyramid stages is %d. Please use larger input images or '
                    'initialise pyramid with different number of stages.')

        # Check stages=1
        dims = torch.Size((1, 7, 7))
        with pytest.raises(ValueError) as exin:
            self.sw_small._check_height(dims)
        assert str(exin.value) == dims_msg % (0, 1)

        dims = torch.Size((1, 8, 8))
        assert self.sw_small._check_height(dims)

        dims = torch.Size((1, 64, 64))
        assert self.sw_small._check_height(dims)

        # Check stages=4
        dims = torch.Size((1, 15, 15))
        with pytest.raises(ValueError) as exin:
            self.sw._check_height(dims)
        assert str(exin.value) == dims_msg % (1, 2)

        dims = torch.Size((1, 16, 16))
        assert self.sw._check_height(dims)

        dims = torch.Size((1, 128, 128))
        assert self.sw._check_height(dims)

    def test_forward(self):
        """
        Tests :func:`expert.models.pyramids.SteerableWavelet.forward` function.
        """
        expert.setup_random_seed()

        # Check Tensors of 1
        x = torch.ones(1, 8, 8)
        pyr = self.sw_small.forward(x)
        high_pass, pyr = pyr[0], pyr[1:]
        # Correct Tensors
        low_pass_residual = torch.ones(1, 4, 4) * 4.0
        true_high_pass = 1.61e-16 * torch.ones(1, 8, 8)
        band_1 = -1.57e-16 * torch.ones(1, 8, 8)
        band_2 = -1.14e-16 * torch.ones(1, 8, 8)
        band_3 = -1.14e-16 * torch.ones(1, 8, 8)
        bands = torch.stack([band_1, band_2, band_3])
        assert torch.allclose(true_high_pass, high_pass, atol=1e-4)
        assert torch.allclose(bands, pyr[0], atol=1e-4)
        assert torch.allclose(low_pass_residual, pyr[-1], atol=1e-4)

        # Matrix of 0's with 1's in the diagonal elements
        x = torch.zeros(8, 8)
        x = x.fill_diagonal_(1.0).unsqueeze(0)
        pyr = self.sw_small.forward(x)
        high_pass, pyr = pyr[0], pyr[1:]
        # Correct Tensors
        low_pass_residual = torch.Tensor([[1.207, 0.500, -0.207, 0.500],
                                          [0.500, 1.207, 0.500, -0.207],
                                          [-0.207, 0.500, 1.207, 0.500],
                                          [0.500, -0.207, 0.500, 1.207]])
        true_high_pass = torch.Tensor([
            [0.552, -0.302, -0.052, 0.052, 0.052, 0.052, -0.052, -0.302],
            [-0.302, 0.552, -0.302, -0.052, 0.052, 0.052, 0.052, -0.052],
            [-0.052, -0.302, 0.552, -0.302, -0.052, 0.052, 0.052, 0.052],
            [0.052, -0.052, -0.302, 0.552, -0.302, -0.052, 0.052, 0.052],
            [0.052, 0.052, -0.052, -0.302, 0.552, -0.302, -0.052, 0.052],
            [0.052, 0.052, 0.052, -0.052, -0.302, 0.552, -0.302, -0.052],
            [-0.052, 0.052, 0.052, 0.052, -0.052, -0.302, 0.552, -0.302],
            [-0.302, -0.052, 0.052, 0.052, 0.052, -0.052, -0.302, 0.552]])
        band_1 = torch.Tensor([
            [-0.167, -0.059, 0.083, 0.059, 0.000, 0.059, 0.083, -0.059],
            [-0.059, -0.167, -0.059, 0.083, 0.059, 0.000, 0.059, 0.083],
            [0.083, -0.059, -0.167, -0.059, 0.083, 0.059, 0.000, 0.059],
            [0.059, 0.083, -0.059, -0.167, -0.059, 0.083, 0.059, 0.000],
            [0.000, 0.059, 0.083, -0.059, -0.167, -0.059, 0.083, 0.059],
            [0.059, 0.000, 0.059, 0.083, -0.059, -0.167, -0.059, 0.083],
            [0.083, 0.059, 0.000, 0.059, 0.083, -0.059, -0.167, -0.059],
            [-0.059, 0.083, 0.059, 0.000, 0.059, 0.083, -0.059, -0.167]])
        band_2 = torch.Tensor([
            [-0.022, -0.008, 0.011, 0.008, 0.000, 0.008, 0.011, -0.008],
            [-0.008, -0.022, -0.008, 0.011, 0.008, 0.000, 0.008, 0.011],
            [0.011, -0.008, -0.022, -0.008, 0.011, 0.008, 0.000, 0.008],
            [0.008, 0.011, -0.008, -0.022, -0.008, 0.011, 0.008, 0.000],
            [0.000, 0.008, 0.011, -0.008, -0.022, -0.008, 0.011, 0.008],
            [0.008, 0.000, 0.008, 0.011, -0.008, -0.022, -0.008, 0.011],
            [0.011, 0.008, 0.000, 0.008, 0.011, -0.008, -0.022, -0.008],
            [-0.008, 0.011, 0.008, 0.000, 0.008, 0.011, -0.008, -0.022]])
        band_3 = torch.Tensor([
            [-0.311, -0.110, 0.156, 0.110, 0.000, 0.110, 0.156, -0.110],
            [-0.110, -0.311, -0.110, 0.156, 0.110, 0.000, 0.110, 0.156],
            [0.156, -0.110, -0.311, -0.110, 0.156, 0.110, 0.000, 0.110],
            [0.110, 0.156, -0.110, -0.311, -0.110, 0.156, 0.110, 0.000],
            [0.000, 0.110, 0.156, -0.110, -0.311, -0.110, 0.156, 0.110],
            [0.110, 0.000, 0.110, 0.156, -0.110, -0.311, -0.110, 0.156],
            [0.156, 0.110, 0.000, 0.110, 0.156, -0.110, -0.311, -0.110],
            [-0.110, 0.156, 0.110, 0.000, 0.110, 0.156, -0.110, -0.311]])
        bands = torch.stack([band_1, band_2, band_3])
        assert torch.allclose(true_high_pass, high_pass, atol=1e-3)
        assert torch.allclose(bands, pyr[0], atol=1e-3)
        assert torch.allclose(low_pass_residual, pyr[-1], atol=1e-3)
        assert torch.allclose(true_high_pass, high_pass, atol=1e-3)

        # Test pyramid of height more than 1
        x = torch.zeros(16, 16)
        x = x.fill_diagonal_(1.0).unsqueeze(0)
        pyr = self.sw.forward(x)
        high_pass, pyr = pyr[0], pyr[1:]
        # Correct Tensors
        low_pass_residual = torch.Tensor([[2.4142, 1., -0.4142, 1.],
                                          [1., 2.4142, 1., -0.4142],
                                          [-0.4142, 1., 2.4142, 1.],
                                          [1., -0.4142, 1.,  2.4142]])
        # we'll check the first row and first column and size since the
        # matrices are quite big.
        true_high_pass_row_column = torch.Tensor(
            [0.5377, -0.3060, -0.0342,  0.0737,  0.0259, -0.0219, -0.0176,
             0.0042,  0.0141,  0.0042, -0.0176, -0.0219,  0.0259,  0.0737,
             -0.0342, -0.3060])
        band_1_1_row = torch.Tensor(
            [0., -0.094, -0.0479, 0.0274, 0.0283, 0.011, 0.008, 0.0014,  0.,
             -0.0014, -0.008, -0.011, -0.0283, -0.0274, 0.0479, 0.094])
        band_1_1_column = torch.Tensor(
            [0., 0.094, 0.0479, -0.0274, -0.0283, -0.011, -0.008, -0.0014, 0.,
             0.0014, 0.008, 0.011, 0.0283, 0.0274, -0.0479, -0.094])
        band_1_2_column = torch.Tensor(
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        band_1_2_row = torch.Tensor(
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        band_1_3_column = torch.Tensor(
            [0., -0.094, -0.0479, 0.0274, 0.0283, 0.011, 0.008, 0.0014, 0.,
             -0.0014, -0.008, -0.011, -0.0283, -0.0274, 0.0479, 0.094])
        band_1_3_row = torch.Tensor(
            [0., 0.094, 0.0479, -0.0274, -0.0283, -0.011, -0.008, -0.0014, 0.,
             0.0014, 0.008, 0.011, 0.0283, 0.0274, -0.0479, -0.094])
        band_1_4_column = torch.Tensor(
            [0., -0.2658, -0.1356, 0.0774, 0.0799, 0.0311, 0.0225, 0.0041,
             0., -0.0041, -0.0225, -0.0311, -0.0799, -0.0774, 0.1356, 0.2658])
        band_1_4_row = torch.Tensor(
            [0., 0.2658, 0.1356, -0.0774, -0.0799, -0.0311, -0.0225, -0.0041,
             0., 0.0041, 0.0225, 0.0311, 0.0799, 0.0774, -0.1356, -0.2658])
        bands_1 = [(band_1_1_row, band_1_1_column),
                   (band_1_2_row, band_1_2_column),
                   (band_1_3_row, band_1_3_column),
                   (band_1_4_row, band_1_4_column)]
        band_2_1_column = torch.Tensor(
            [0., 0.1909, 0.1118, -0.0327, 0., 0.0327, -0.1118, -0.1909])
        band_2_1_row = torch.Tensor(
            [0., -0.1909, -0.1118, 0.0327, 0., -0.0327, 0.1118, 0.1909])
        band_2_2_column = torch.Tensor([0., 0., 0., 0., 0., 0., 0., 0.])
        band_2_2_row = torch.Tensor([0., 0., 0., 0., 0., 0., 0., 0.])
        band_2_3_column = torch.Tensor(
            [0., -0.1909, -0.1118, 0.0327, 0., -0.0327, 0.1118, 0.1909])
        band_2_3_row = torch.Tensor(
            [0., 0.1909, 0.1118, -0.0327, 0., 0.0327, -0.1118, -0.1909])
        band_2_4_column = torch.Tensor(
            [0., -0.5398, -0.3162, 0.0926, 0., -0.0926, 0.3162, 0.5398])
        band_2_4_row = torch.Tensor(
            [0., 0.5398, 0.3162, -0.0926, 0., 0.0926, -0.3162, -0.5398])
        bands_2 = [(band_2_1_row, band_2_1_column),
                   (band_2_2_row, band_2_2_column),
                   (band_2_3_row, band_2_3_column),
                   (band_2_4_row, band_2_4_column)]

        assert torch.allclose(true_high_pass_row_column,
                              high_pass[0, 0, :], atol=1e-3)
        assert torch.allclose(true_high_pass_row_column,
                              high_pass[0, :, 0], atol=1e-3)
        assert torch.allclose(low_pass_residual, pyr[-1], atol=1e-4)
        for i in range(4):
            row, col = bands_1[i]
            assert torch.allclose(row, pyr[0][:, i, 0, :], atol=1e-3)
            assert torch.allclose(col, pyr[0][:, i, :, 0], atol=1e-3)

            row, col = bands_2[i]
            assert torch.allclose(row, pyr[1][:, i, 0, :], atol=1e-3)
            assert torch.allclose(col, pyr[1][:, i, :, 0], atol=1e-3)


class TestSteerablePyramid():
    """
    Tests :class:`expert.models.pyramids.SteerablePyramid` class.
    """
    expert.setup_random_seed()
    sw = emp.SteerablePyramid(stages=5, num_orientations=10)
    sw_pretrained = emp.SteerablePyramid(pretrained=True)

    def test_init(self):
        """
        Tests :class:`expert.models.pyramids.SteerablePyramid` class init.
        """
        assert (issubclass(emp.SteerablePyramid, nn.Module))
        assert (self.sw.__class__.__bases__[0].__name__ == 'Module')

        assert self.sw.stages == 5
        assert self.sw.num_orientations == 10
        assert self.sw.lo0filt.size() == torch.Size([1, 1, 9, 9])
        assert self.sw.hi0filt.size() == torch.Size([1, 1, 9, 9])
        assert self.sw.lofilt.size() == torch.Size([1, 1, 17, 17])
        assert self.sw.bfilts.size() == torch.Size([10, 1, 9, 9])

        # Test pretrained
        filters = filt_utils.STEERABLE_SPATIAL_FILTERS_1
        assert torch.allclose(
            self.sw_pretrained.lo0filt.data, filters['lo0filt'])
        assert torch.allclose(
            self.sw_pretrained.hi0filt.data, filters['hi0filt'])
        assert torch.allclose(
            self.sw_pretrained.lofilt.data, filters['lofilt'])
        assert torch.allclose(
            self.sw_pretrained.bfilts.data, filters['bfilts'])

    def test_validate_input(self):
        """
        Tests :func:`expert.models.pyramids.SteerablePyramid._validate_input`
        function.
        """
        stages_msg = ('stages parameter must be an integer greater than 0.')
        num_orientations_msg = (
            'num_orientations parameter must be an integer greater than 0.')
        pretrained_msg = ('pretrained parameter must be a boolean.')
        pretrained_err_msg = (
            'To use the pretrained network, num_orientations must be 2.')

        with pytest.raises(TypeError) as exin:
            self.sw._validate_input('k', 1, True)
        assert str(exin.value) == stages_msg

        with pytest.raises(TypeError) as exin:
            self.sw._validate_input(-1, 1, True)
        assert str(exin.value) == stages_msg

        with pytest.raises(TypeError) as exin:
            self.sw._validate_input(1, 'k', True)
        assert str(exin.value) == num_orientations_msg

        with pytest.raises(TypeError) as exin:
            self.sw._validate_input(1, -1, True)
        assert str(exin.value) == num_orientations_msg

        with pytest.raises(TypeError) as exin:
            self.sw._validate_input(1, 1, 'k')
        assert str(exin.value) == pretrained_msg

        with pytest.raises(ValueError) as exin:
            self.sw._validate_input(1, 1, True)
        assert str(exin.value) == pretrained_err_msg

        # All good
        self.sw._validate_input(1, 2, True)
        self.sw._validate_input(1, 1, False)

    def test_forward(self):
        """
        Tests :func:`expert.models.pyramids.SteerablePyramid.forward` function.
        """
        expert.setup_random_seed()


class TestLaplacianPyramid():
    """
    Tests :class:`expert.models.pyramids.LaplacianPyramid` class.
    """
    expert.setup_random_seed()
    lp = emp.LaplacianPyramid(k=5)

    def test_init(self):
        """
        Tests :class:`expert.models.pyramids.LaplacianPyramid` class init.
        """
        assert (issubclass(emp.LaplacianPyramid, nn.Module))
        assert (self.lp.__class__.__bases__[0].__name__ == 'Module')

    def test_validate_input(self):
        """
        Tests :func:`expert.models.pyramids.LaplacianPyramid._validate_input`
        function.
        """
        return True

    def test_forward(self):
        """
        Tests :func:`expert.models.pyramids.LaplacianPyramid.forward` function.
        """
        expert.setup_random_seed()

        test_1 = torch.ones((1, 3, 200, 200))
        test_2 = torch.ones((1, 3, 200, 200)) - 0.1
        diff = self.lp.compare(test_1, test_2)
        assert True
