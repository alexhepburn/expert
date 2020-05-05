"""
Tests for pyramid based models
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import expert
import expert.models.pyramids as emp
import expert.utils.filters as filt_utils

TEST_INPUT = torch.ones((1, 1, 5, 5))
TEST_INPUT = F.pad(TEST_INPUT, [5, 5, 5, 5], value=0.5)


class TestSteerableWavelet():
    """
    Tests :class:`expert.models.pyramids.SteerableWavelet` class.
    """
    expert.setup_random_seed()
    sw = emp.SteerableWavelet()

    def test_init(self):
        """
        Tests :class:`expert.models.pyramids.SteerableWavelet` class init.
        """
        assert (issubclass(emp.SteerableWavelet, nn.Module))
        assert (self.sw.__class__.__bases__[0].__name__ == 'Module')

    def test_validate_input(self):
        """
        Tests :func:`expert.models.pyramids.SteerableWavelet._validate_input`
        function.
        """
        return True

    def test_forward(self):
        """
        Tests :func:`expert.models.pyramids.SteerableWavelet.forward` function.
        """
        expert.setup_random_seed()


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
        filters = filt_utils.STEERABLE_SPATIAL_FILTERS
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
