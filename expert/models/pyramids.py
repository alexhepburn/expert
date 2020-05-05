"""
The :mod:`expert.layers.wavelet_transformation` module holds classes of
layers for wavelet transformations.
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

from typing import List, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import expert.layers.divisive_normalisation as expert_divisive_normalisation

import expert.utils.filters as filt_utils
import expert.utils.conv as conv_utils

__all__ = ['SteerableWavelet', 'SteerablePyramid', 'LaplacianPyramid',
           'LaplacianPyramidGDN']

LAPLACIAN_FILTER = np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                             dtype=np.float32)


class LaplacianPyramid(nn.Module):
    def __init__(self, k, dims=3, filt=None, trainable=False):
        super(LaplacianPyramid, self).__init__()
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (dims, 1, 1)),
                              (dims, 1, 5, 5))
        self.k = k
        self.trainable = trainable
        self.dims = dims
        self.filt = nn.Parameter(torch.Tensor(filt), requires_grad=False)
        self.dn_filts, self.sigmas = self.DN_filters()

    def DN_filters(self):
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        dn_filts = []
        dn_filts.append(torch.Tensor(np.reshape([[0, 0.1011, 0],
                                    [0.1493, 0, 0.1460],
                                    [0, 0.1015, 0.]]*self.dims,
                                   (self.dims,  1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0757, 0],
                                    [0.1986, 0, 0.1846],
                                    [0, 0.0837, 0]]*self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0.0477, 0],
                                    [0.2138, 0, 0.2243],
                                    [0, 0.0467, 0]]*self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2503, 0, 0.2616],
                                    [0, 0, 0]]*self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2598, 0, 0.2552],
                                    [0, 0, 0]]*self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
                                    [0.2215, 0, 0.0717],
                                    [0, 0, 0]]*self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32)))
        dn_filts = nn.ParameterList([nn.Parameter(x, requires_grad=self.trainable)
                                     for x in dn_filts])
        sigmas = nn.ParameterList([nn.Parameter(torch.Tensor(np.array(x)),
                                  requires_grad=self.trainable) for x in sigmas])
        return dn_filts, sigmas

    def pyramid(self, im):
        out = []
        J = im
        pyr = []
        for i in range(0, self.k):
            J_padding_amount = conv_utils.pad([J.size(2), J.size(3)],
                                            self.filt.size(3), stride=2)
            I = F.conv2d(F.pad(J, J_padding_amount, mode='reflect'), self.filt,
                         stride=2, padding=0, groups=self.dims)
            I_up = F.interpolate(I, size=[J.size(2), J.size(3)],
                                 align_corners=True, mode='bilinear')
            I_padding_amount = conv_utils.pad([I_up.size(2), I_up.size(3)],
                                              self.filt.size(3), stride=1)
            I_up_conv = F.conv2d(F.pad(I_up, I_padding_amount, mode='reflect'),
                                 self.filt, stride=1, padding=0,
                                 groups=self.dims)
            out = J - I_up_conv
            out_padding_amount = conv_utils.pad(
                [out.size(2), out.size(3)], self.dn_filts[i].size(2), stride=1)
            out_conv = F.conv2d(
                F.pad(torch.abs(out), out_padding_amount, mode='reflect'),
                self.dn_filts[i],
                stride=1,
                groups=self.dims)
            out_norm = out / (self.sigmas[i]+out_conv)
            pyr.append(out_norm)
            J = I
        return pyr

    def compare(self, x1, x2):
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        total = []
        # Calculate difference in perceptual space (Tensors are stored
        # strangley to avoid needing to pad tensors)
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        return torch.norm(torch.stack(total), 0.6)


class LaplacianPyramidGDN(nn.Module):
    def __init__(self, k, dims=3, filt=None):
        super(LaplacianPyramidGDN, self).__init__()
        if filt is None:
            filt = np.tile(LAPLACIAN_FILTER, (dims, 1, 1))
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (dims, 1, 1)),
                              (dims, 1, 5, 5))
        self.k = k
        self.dims = dims
        self.filt = nn.Parameter(torch.Tensor(filt))
        self.filt.requires_grad = False
        self.gdns = nn.ModuleList([expert_divisive_normalisation.GDN(
            dims, apply_independently=True) for i in range(self.k)])
        self.pad_one = nn.ReflectionPad2d(1)
        self.pad_two = nn.ReflectionPad2d(2)
        self.mse = nn.MSELoss(reduction='none')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

    def pyramid(self, im):
        J = im
        pyr = []
        for i in range(0, self.k):
            I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0,
                         groups=self.dims)
            I_up = self.upsample(I)
            I_up_conv = F.conv2d(self.pad_two(I_up), self.filt, stride=1,
                                 padding=0, groups=self.dims)
            if J.size() != I_up_conv.size():
                I_up_conv = torch.nn.functional.interpolate(I_up_conv, [J.size(2), J.size(3)])
            pyr.append(self.gdns[i](J - I_up_conv))
            J = I
        return pyr

    def compare(self, x1, x2):
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        total = []
        # Calculate difference in perceptual space (Tensors are stored
        # strangley to avoid needing to pad tensors)
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        return torch.mean(torch.stack(total), dim=0)


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
