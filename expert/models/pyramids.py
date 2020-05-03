"""
The :mod:`expert.layers.wavelet_transformation` module holds classes of
layers for wavelet transformations.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import expert.utils.filters as filt_utils
import expert.utils.conv as conv_utils

__all__ = ['SteerableWavelet', 'SteerablePyramid']


class SteerableWavelet(nn.Module):
    """
    Steerable wavelet layer.

    Performs a high-pass and low-pass filtering. The low-pass filtered image
    is then passed through ``wavelets`` wavelet transformations, each
    with a different orientation. This is done at ``scales`` dfferent scales
    where each scale is downsampled by ``downsampled``.
    """
    def __init__(self):
        """
        Constructs a ``SteerableWavelet`` class.
        """
        super(SteerableWavelet, self).__init__()

    def _validate_input(self):
        """
        Validates input of the steerable wavelet class.

        For the description of the input parameters and exceptions raised by
        this function, please see the documentation of the
        :class:`expert.models.pyramids.SteerableWavelet` class.

        Returns
        -------
        is_valid
            ``True`` if input is valid, ``False`` otherwise.
        """
        is_valid = False

        is_valid = True
        return is_valid

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """

        """
        return x


class SteerablePyramid(nn.Module):
    """
    Steerable pyramid model implemented in spatial domain, introduced in
    [SIMON1995PYR]_.

    Parameters
    ----------
    stages : int, optional (default=4)
        Number of stages to be used in the pyramid.
    num_orientations: int, optional (default=2)
        Number of orientations to be used at each stage of the pyramid (number
        of subbands). If ``pretrained`` is ``True`` then this must be 2 as the
        pretarined weights from the original implementation use 2 subbands.
    pretrained : bool, optional (default=False)
        Whether to load the pretrained filters, specified in the original paper
        [SIMON1995PYR]_.

    Raises
    ------


    Attributes
    ----------

    . [SIMON1995PYR] E P Simoncelli and W T Freeman, "The Steerable Pyramid:
       A Flexible Architecture for Multi-Scale Derivative Computation," Second
       Int'l Conf on Image Processing, Washington, DC, Oct 1995.

    TODO: implement an initialisation that sets ``num_orientations`` filters
          that are rotated a different amounts, and optimise only over the
          amount of rotation for each filter.
    """
    def __init__(self,
                 stages: int = 4,
                 num_orientations: int = 2,
                 pretrained: bool = False) -> None:
        """
        Constructs a ``SteerablePyramid`` class.
        """
        super(SteerablePyramid, self).__init__()
        assert self._validate_input(stages, num_orientations, pretrained)
        self.stages = stages
        self.num_orientations = num_orientations

        self.lo0filt = nn.Parameter(torch.ones(1, 1, 9, 9))
        self.hi0filt = nn.Parameter(torch.ones(1, 1, 9, 9))
        self.lofilt = nn.Parameter(torch.ones(1, 1, 17, 17))
        self.bfilts = nn.Parameter(torch.ones(self.num_orientations, 1, 9, 9))

        for param in self.parameters():
            torch.nn.init.normal_(param)

        if pretrained:
            filters = filt_utils.STEERABLE_SPATIAL_FILTERS
            with torch.no_grad():
                self.lo0filt.data = filters['lo0filt']
                self.hi0filt.data = filters['hi0filt']
                self.lofilt.data = filters['lofilt']
                self.bfilts.data = filters['bfilts']

    def _validate_input(self, stages: int, num_orientations: int,
                        pretrained: bool) -> bool:
        """
        Validates input of the steerable pyramid class.

        For the description of the input parameters and exceptions raised by
        this function, please see the documentation of the
        :class:`expert.models.pyramids.SteerablePyramid` class.

        Returns
        -------
        is_valid
            ``True`` if input is valid, ``False`` otherwise.
        """
        is_valid = False

        if not isinstance(stages, int) or stages <= 0:
            raise TypeError('stages parameter must be an integer greater than '
                            '0.')

        if not isinstance(num_orientations, int) or num_orientations <= 0:
            raise TypeError('num_orientations parameter must be an integer '
                            'greater than 0.')

        if not isinstance(pretrained, bool):
            raise TypeError('pretrained parameter must be a boolean.')

        if pretrained and num_orientations != 2:
            raise ValueError('To use the pretrained network, num_orientations '
                             'must be 2.')

        is_valid = True
        return is_valid

    def forward(
        self,
        x: torch.tensor,
        upsample_output: bool = False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the pyramid.

        This function returns a lit of Tensors for the subbands in each stage
        of the pyramid or, if ``upsample_output`` is ``True``, a Tensor
        containing every subband at every level that has been upsampled to be
        the same size as the input Tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input to the pyramid.
        upsample_output : boolean
            If ``True`` then the every subband will be upsampled to be the same
            size as the input and then stacked to be a singular tensor.

        Raises
        ------
        TypeError:
            Input parameter ``x`` is not of dtype torch.float.

        Returns
        -------
        pyramid : Union[List[torch.Tensor], Tensor]
            List of tensors, where each entry contains the subbands at each
            stage of the pyramid. The low pass residual is the last element
            in the pyramid. If ``upsample_output`` is ``True`` then this will
            be one Tensor where each subband has been upsample to be the same
            dimensions as the input.
        hi_pass : torch.Tensor
            Tensor containing the high pass residual.

        TODO: whether to return padded tensor or list of tensors
        """
        if x.dtype != torch.float32:
            raise TypeError('Input x must be of type torch.float32.')

        pyramid = []

        padded_x = F.pad(x,
                         pad=conv_utils.pad([x.size(2), x.size(3)], 9, 1),
                         mode='reflect')
        low_pass = F.conv2d(padded_x, self.lo0filt)
        high_pass = F.conv2d(padded_x, self.hi0filt)

        for h in range(0, self.stages):
            image_size = [low_pass.size(2), low_pass.size(3)]
            padded_lowpass = F.pad(low_pass,
                                   pad=conv_utils.pad(image_size, 9, 1),
                                   mode='reflect')
            subbands = F.conv2d(padded_lowpass, self.bfilts, groups=1)
            padded_lowpass = F.pad(low_pass,
                                   pad=conv_utils.pad(image_size, 17, 1),
                                   mode='reflect')
            low_pass = F.conv2d(padded_lowpass, self.lofilt, stride=[2, 2])
            pyramid.append(subbands)
        pyramid.append(low_pass)

        if upsample_output:
            original_size = [high_pass.size(2), high_pass.size(3)]
            pyr_upsample = [
                F.interpolate(stage, size=original_size) for stage in pyramid
            ]
            pyramid = torch.cat(pyr_upsample, dim=1)

        return pyramid, high_pass
