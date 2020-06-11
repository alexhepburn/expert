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
    sw = emp.SteerableWavelet()
    sw_small = emp.SteerableWavelet(stages=1, order=2, twidth=1)

    def test_init(self):
        """
        Tests :class:`expert.models.pyramids.SteerableWavelet` class init.
        """
        assert (issubclass(emp.SteerableWavelet, nn.Module))
        assert (self.sw.__class__.__bases__[0].__name__ == 'Module')

        assert self.sw.stages == 4
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
        dims=torch.Size((1, 7, 7))
        with pytest.raises(ValueError) as exin:
            self.sw_small._check_height(dims)
        assert str(exin.value) == dims_msg%(0, 1)

        dims = torch.Size((1, 8, 8))
        assert self.sw_small._check_height(dims)

        dims = torch.Size((1, 64, 64))
        assert self.sw_small._check_height(dims)

        # Check stages=4
        dims = torch.Size((1, 63, 63))
        with pytest.raises(ValueError) as exin:
            self.sw._check_height(dims)
        assert str(exin.value) == dims_msg%(3, 4)

        dims = torch.Size((1, 64, 64))
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
        pyr, high_pass = self.sw_small.forward(x)
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

        # Check diagonal matrix
        # TODO: check diagonal matrix

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
        filters = filt_utils.STEERABLE_SPATIAL_FILTERS_0
        assert torch.equal(self.sw_pretrained.lo0filt.data, filters['lo0filt'])
        assert torch.equal(self.sw_pretrained.hi0filt.data, filters['hi0filt'])
        assert torch.equal(self.sw_pretrained.lofilt.data, filters['lofilt'])
        assert torch.equal(self.sw_pretrained.bfilts.data, filters['bfilts'])

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
