"""
Tests for correlation PerceptNet classes.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import torch
import torch.nn as nn

import expert
import expert.models.perceptnet as emp
import expert.layers.divisive_normalisation as eldn


class TestPerceptNet():
    """
    Tests :class:`expert.models.perceptnet.PerceptNet` class.
    """
    expert.setup_random_seed()
    net1 = emp.PerceptNet()

    def test_init(self):
        """
        Tests :class:`exper.models.perceptnet.PerceptNet` class init.
        """
        assert (issubclass(emp.PerceptNet, nn.Module))
        assert (self.net1.__class__.__bases__[0].__name__ == 'Module')

        assert isinstance(self.net1.normalisation_1, eldn.GDN)
        # check apply_independently
        assert self.net1.normalisation_1.groups == 3
        assert isinstance(self.net1.conv1, nn.Conv2d)
        assert isinstance(self.net1.maxpool, nn.MaxPool2d)
        assert isinstance(self.net1.normalisation_2, eldn.GDN)
        assert isinstance(self.net1.conv2, nn.Conv2d)
        assert isinstance(self.net1.normalisation_3, eldn.GDN)
        assert isinstance(self.net1.conv3, nn.Conv2d)
        assert isinstance(self.net1.normalisation_4, eldn.GDN)

        def test_validate_input(self):
            """
            Tests :func:`expert.models.perceptnet.PerceptNet._validate_input`
            function.
            """
            dims_err = ('dims parameter must be an integer greater than 0.')
            normalisation_err = ('normalisation parameter must be a string.')
            normalisation_str_err = ('normalisation %s not defined. Please '
                                     'see PerceptNet documentation for '
                                     'possible options.')
            pretrained_err = ('pretrained parameter must be a boolean.')
            network_err = ('The pretrained network uses gdn as the '
                           'normalisation layer. If using a pretrained '
                           'network, please selects gdn as the normalisation.')

            with pytest.raises(TypeError) as exin:
                self.net1._validate_input('k', 1, 1)
            assert str(exin.value) == dims_err

            with pytest.raises(TypeError) as exin:
                self.net1._validate_input(1, 1, 1)
            assert str(exin.value) == normalisation_err

            with pytest.raises(ValueError) as exin:
                self.net1._validate_input(1, 'test', 1)
            assert str(exin.value) == normalisation_str_err.format('test')

            with pytest.raises(TypeError) as exin:
                self.net1._validate_input(1, 'gdn', 'k')
            assert str(exin.value) == pretrained_err

            with pytest.raises(ValueError) as exin:
                self.net1._validate_input(1, 'batch_norm', True)
            assert str(exin.value) == network_err

        def test_forward():
            """
            Tests :func:`expert.models.perceptnet.PerceptNet.forward` function.
            TODO : finish forward tests.
            """
            expert.setup_random_seed()

            assert True
