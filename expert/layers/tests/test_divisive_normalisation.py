"""
Tests for divisive normalisation classes
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

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
                                                         [[[1.4552e-11]]]],
                                                         dtype=np.float32))
        initial_beta_indep = torch.from_numpy(np.array([1.], dtype=np.float32))
        assert torch.allclose(self.gdn_apply_independently.gamma,
                              initial_gamma_indep)
        assert torch.allclose(self.gdn_apply_independently.beta,
                              initial_beta_indep)

    def test_validate_input(self):
        assert True

    def test_forward(self):
        """
        Tests :func:`expert.layers.divisive_normalisation.GDN.forward`
        function.
        """
        expert.setup_random_seed()
        assert True
