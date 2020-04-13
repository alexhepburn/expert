"""
Tests for divisive normalisation classes
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import torch
import torch.nn as nn

import expert
import expert.layers.divisive_normalisation as edn


class TestGDN():
    """
    Tests :class:`expert.layers.divisive_normalisation.GDN` class.
    """
    expert.setup_random_seed()
    gdn = edn.GDN(2)
    gdn_apply_independently = edn.GDN(2, apply_independently=True)

    def test_init(self):
        """
        Tests :class:`expert.layers.divisive_normalisation.GDN` class init.
        """
        assert (issubclass(edn.GDN, nn.Module))
        assert (self.gdn.__class__.__bases__[0].__name__
                == 'Module')
        assert np.isclose(self.gdn.reparam_offset, 3.81e-06)
        assert np.isclose(self.gdn.beta_reparam, 0.001)
        assert self.gdn.groups == 1
        assert self.gdn_apply_independently.groups == 2

        initial_gamma = torch.from_numpy(np.array([[[[0.1]], [[1.4552e-11]]],
                                                   [[[1.4552e-10]], [[0.1]]]],
                                                   dtype=np.float32))
        initial_beta = torch.from_numpy(np.array([1., 1.], dtype=np.float32))
        assert torch.allclose(self.gdn.gamma, initial_gamma)
        assert torch.allclose(self.gdn.beta, initial_beta)

        initial_gamma_indep = torch.from_numpy(np.array([[[[0.1]]],
                                                         [[[0.1]]]],
                                                         dtype=np.float32))
        initial_beta_indep = torch.from_numpy(np.array([1.], dtype=np.float32))
        assert torch.allclose(self.gdn_apply_independently.gamma,
                              initial_gamma_indep)
        assert torch.allclose(self.gdn_apply_independently.beta,
                              initial_beta_indep)

    def test_validate_input(self):
        """
        Tests :func:`expert.layers.divisive_normalisation.GDN._validate_input`
        function.
        """
        n_channel_error = ('n_channels parameter must be an integer greater '
                           'than 0.')
        kernel_size_error = ('kernel_size parameter must be an integer '
                             'greater than 0.')
        stride_error = ('stride parameter must be an integer greater than 0.')
        padding_error = ('padding parameter must be a positive integer.')
        gamma_init_error = ('gamma_init parameter must be a positive float.')
        reparam_offset_error = ('reparam_offset parameter must be a positive '
                                'float.')
        beta_min_error = ('beta_min parameter must be a positive float.')
        apply_independently_error = ('apply_independently parameter must be '
                                     'a boolean.')

        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input('k', 1, 1, 1, 1, 1, 1, 1)
        assert str(exin.value) == n_channel_error
        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(-1, 1, 1, 1, 1, 1, 1, 1)
        assert str(exin.value) == n_channel_error

        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 'k', 1, 1, 1, 1, 1, 1)
        assert str(exin.value) == kernel_size_error
        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, -1, 1, 1, 1, 1, 1, 1)
        assert str(exin.value) == kernel_size_error

        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1.2, 1, 1, 1, 1, 1)
        assert str(exin.value) == stride_error
        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, -1, 1, 1, 1, 1, 1)
        assert str(exin.value) == stride_error

        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, 0.1, 1, 1, 1, 1)
        assert str(exin.value) == padding_error
        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, -0.1, 1, 1, 1, 1)
        assert str(exin.value) == padding_error

        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, 1, 'k', 1, 1, 1)
        assert str(exin.value) == gamma_init_error
        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, 1, -0.1, 1, 1, 1)
        assert str(exin.value) == gamma_init_error

        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, 1, 0.1, 'k', 1, 1)
        assert str(exin.value) == reparam_offset_error
        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, 1, 0.1, -0.1, 1, 1)
        assert str(exin.value) == reparam_offset_error

        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, 1, 0.1, 0.1, 'k', 1)
        assert str(exin.value) == beta_min_error
        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, 1, 0.1, 0.1, -0.1, 1)
        assert str(exin.value) == beta_min_error

        with pytest.raises(TypeError) as exin:
            self.gdn._validate_input(1, 1, 1, 1, 0.1, 0.1, 0.1, 'k')
        assert str(exin.value) == apply_independently_error

        # All good
        self.gdn._validate_input(1, 1, 1, 1, 0.1, 0.1, 0.1, True)

    def test_forward(self):
        """
        Tests :func:`expert.layers.divisive_normalisation.GDN.forward`
        function.
        """
        expert.setup_random_seed()

        # Check dtype of tensor
        type_error = ('Input x must be of type torch.float32.')
        ones_bool = torch.ones((1, 2, 2, 2), dtype=torch.bool)
        with pytest.raises(TypeError) as exin:
            y = self.gdn(ones_bool)
        assert str(exin.value) == type_error

        ones = torch.ones((1, 2, 2, 2), dtype=torch.float32)

        y = self.gdn(ones)
        true_output = ones-0.0465
        assert torch.allclose(y, true_output, rtol=0.01)

        y = self.gdn_apply_independently(ones)
        assert torch.allclose(y, true_output, rtol=0.01)
