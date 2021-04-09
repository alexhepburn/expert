"""
The :mod:`expert.models.networks.perceptnet` implements classes of networks
relating to PerceptNet.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>

import torch
import torch.nn as nn

import expert.layers.divisive_normalisation as expert_divisive_normalisation

__all__ = ['PerceptNet']

_NORMALISATIONS = {
    'batch_norm': nn.BatchNorm2d,
    'instance_norm': nn.InstanceNorm2d,
    'gdn': expert_divisive_normalisation.GDN}


class PerceptNet(nn.Module):
    """
    Neural network Network that follows the structure of human visual system
    introduced in [HEPBURN2019PER]_.

    PerceptNet is a neural network where the architecture takes inspiration
    from the various stages in the human visual system. The structure that the
    network mimics is as followed: gamma correction -> opponent colour space ->
    Von Kries transform -> center-surround filters -> LGN normalisation ->
    orientation sensitive and multiscale in V1 -> divisive normalisation in V1.

    If parameter ``pretrained`` is ``True``, then the network will be
    initialised with weights attained by maximising Pearson correlation between
    the $\ell_2$ distance in the transformed domain between an original image
    and a distorted image, and the mean opinion score (MOS) of the distorted
    image. The pretrained network was trained using a training split of the
    TID2008 This training procedure is detailed in [HEPBURN2019PER]_.

    If parameter ``normalisation`` is `'gdn'`, then the default values are
    used, which includes a `kernel_size` of 1, which means that there is
    no spatial element to the divisive normalisation.

    .. [HEPBURN2019PER] Hepburn, Alexander, et al. “PerceptNet: A Human Visual
       System Inspired Neural Network for Estimating Perceptual Distance.”
       ArXiv:1910.12548 [Cs, Eess, Stat], Oct. 2019. arXiv.org,
       http://arxiv.org/abs/1910.12548.

    Parameters
    ----------
    dims : int, optional (default=3)
        The number of dimensions of the input in the channel dimension. Usually
        either 3 for RGB images or 1 for Greyscale.
    normalisation : string, optional (default='gdn')
        The normalisation to be used in the network. The possible values are
        'batch_norm' for batch normalisation, 'instance_norm' for instance
        normalisation, and 'gdn' for general divisive normalisation.
    pretrained : boolean, optional (default=False)
        Whether to load network weights or not. The weights loaded were
        optimised for a training set of the TID2008 dataset.

    Raises
    ------
    ValueError
        ``dims`` parameter is not an integer. ``normalisation`` paramter is not
        a string or not in the known normalisation strings. ``pretrained`` is
        not a boolean.

    Attributes
    ----------
    normalisation_1 : nn.Module
        The first normalisation layer. If it is general divisive normalisation,
        the ``apply_independently`` paramter is `True` to simulate gamma
        correction being the first stage of the human visual system.
    conv_1 : nn.Conv2d
        The first convolutional layer. If default PerceptNet, this represents
        a transformation into opponent colour space.
    maxpool : nn.Module
        The max pool layer.
    normalisation_2 : nn.Module
        The second normalisation layer. If default PerceptNet, this represents
        a Von-Kries transform.
    conv_2 : nn.Conv2d
        The second convolutional layer. If default PerceptNet, this represents
        center-surround filters.
    normalisation_3 : nn.Module
        The third normalisation layer. If default PerceptNet, this represents
        LGN normalisation.
    conv_3 : nn.Conv2d
        The third convolutional layer. If default PerceptNet, this represents
        orientation sensitive and multiscale in V1.
    normalisation_4 : nn.Module
        The fourth normalistaion layer. If default PerceptNet, this represents
        the divisive normalisation in V1.
    features : nn.Sequential
        A sequential object of all the layers that create the network.
    """
    def __init__(self,
                 dims: int = 3,
                 normalisation: str = 'gdn',
                 pretrained: bool = False) -> None:
        """
        Constructs a ``PerceptNet`` class.
        """
        super(PerceptNet, self).__init__()
        # TODO: pretrained weights - look where to save the weights
        assert self._validate_input(dims, normalisation, pretrained)

        normalisation_layer = _NORMALISATIONS[normalisation]

        if normalisation_layer == expert_divisive_normalisation.GDN:
            # If GDN then make first layer channel independent
            normalisation_1 = normalisation_layer(
                dims, apply_independently=True)
        else:
            normalisation_1 = normalisation_layer(dims)
        normalisation_2 = normalisation_layer(dims)
        normalisation_3 = normalisation_layer(6)
        normalisation_4 = normalisation_layer(128)
        conv1 = nn.Conv2d(dims, dims, kernel_size=1, stride=1, padding=1)
        maxpool = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(dims, 6, kernel_size=5, stride=1, padding=1)
        conv3 = nn.Conv2d(6, 128, kernel_size=5, stride=1, padding=1)

        # Called features to be used as feature extraction just like the
        # modles in the torchvision package.
        self.features = nn.Sequential(
            normalisation_1,
            conv1,
            maxpool,
            normalisation_2,
            conv2,
            maxpool,
            normalisation_3,
            conv3,
            normalisation_4)

    def _validate_input(self,
                        dims: int,
                        normalisation: str,
                        pretrained: bool):
        """
        Validates input of the generalised divisive normalisation class.

        For the description of the input parameters and exceptions raised by
        this function, please see the documentation of the
        :class:`expert.models.networks.perceptnet.PerceptNet` class.

        Returns
        -------
        is_valid
            ``True`` if input is valid, ``False`` otherwise.
        """
        is_valid = False

        if not isinstance(dims, int) or dims <= 0:
            raise TypeError('dims parameter must be an integer greater than '
                            '0.')

        if not isinstance(normalisation, str):
            raise TypeError('normalisation parameter must be a string.')

        if normalisation not in _NORMALISATIONS:
            raise ValueError('normalisation %s not defined. Please see'
                             'PerceptNet documentation for possible options.')

        if not isinstance(pretrained, bool):
            raise TypeError('pretrained parameter must be a boolean.')

        if pretrained and normalisation is not "gdn":
            raise ValueError('The pretrained network uses gdn as the '
                             'normalisation layer. If using a pretrained '
                             'network, please selects gdn as the '
                             'normalisation.')

        is_valid = True
        return is_valid

    def forward(self,
                x: torch.Tensor):
        """
        Forward Pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the layer. Must be of shape [batch_size, channels,
            height, width].

        Raises
        ------
        TypeError:
            Input parameter ``x`` is not of dtype torch.float.

        Returns
        -------
        output : torch.Tensor
            Output of the network, the inputs representation in a more
            perceptually meaningful space.
        """
        if x.dtype != torch.float32:
            raise TypeError('Input x must be of type torch.float32.')

        output = self.features(x)
        return output
