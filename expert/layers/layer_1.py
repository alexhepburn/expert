"""
"""
# Author: Alex Hepburn <alex.hepburn@bristol.ac.uk>
# License: new BSD

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import expert.utils.fourier as fourier
import expert.models as expert_models

__all__ = ['Layer1']


class Layer1(nn.Module):
    """
    """

    def __init__(self):
        """
        """
        super(Layer1, self).__init__()

        self.N = 9
        self.Npatch = 6
        self.d = 9
        self.H = torch.ones(9, 9) * 0.2222
        self.H.fill_diagonal_(1.2222)
        self.b = torch.ones(self.d, self.Npatch) * 0.01
        self.gamma = 1.3
        self.beta = 2
        self.scale = 255
        self.L = 1

    def forward(self, x):
        """
        """
        x[x == 0] = 1e-6
        yy = self.L * x
        y = yy
        e = y ** self.gamma
        elog = torch.log(y)
        denom = self.b + torch.mm(self.H, e)
        all_1 = torch.ones(self.d, self.d)
        K = self.scale * (self.b + (self.beta/self.d) *
                          torch.mm(all_1, e) + torch.ones(self.d, self.Npatch))
        x = K * e / denom

        xim = x

        return yy, xim


class Layer2(nn.Module):
    ""
    ""

    def __init__(self):
        """
        """
        super(Layer2, self).__init__()
        self.fs = 64
        self.N = 16
        self.d = 16
        self.Npatch = 1
        self.contrast = 1
        self.Ls = 0.066
        self.Lc = 1

        # nonlinear
        self.Hs = 0.066
        self.Hc = 1
        self.b = torch.ones(self.N, self.Npatch) * 30
        self.g = 1

        (H1, dHds1) = make_2d_gauss_kernel(
            self.fs, int(np.sqrt(self.N)), self.Ls)
        (H2, dHds2) = make_2d_gauss_kernel(
            self.fs, int(np.sqrt(self.N)), self.Hs)

        self.H1 = self.Lc*H1
        self.H = self.Hc*H2
        self.L = (torch.eye(self.N) - H1)
        self.iL = torch.inverse(self.L)

    def forward(self, x):
        """
        """
        yy = torch.mm(self.L, x.unsqueeze(-1))
        y = yy

        denom = self.b + torch.mm(torch.mm(self.H, self.iL), y)
        x = y / denom

        xim = x
        return yy, xim


class Layer3(nn.Module):
    """
    """

    def __init__(self):
        """
        """
        super(Layer3, self).__init__()
        self.general = 1
        self.s_wavelet = 0
        self.brightness = 0
        self.contrast = 0
        self.fs = 64
        self.N = 4
        obliq = 1
        [Hcsf, h] = make_csf_kernel(self.fs, self.N, obliq)
        lam = 0.5e-4
        self.L = Hcsf+lam*torch.eye(self.N**2)
        self.iL = torch.inverse(self.L)
        self.g = 1.5
        Hc_init = 1
        Hs_init = 0.02
        self.Hs = Hs_init
        self.Hc = Hc_init
        [H, dHds] = make_2d_gauss_kernel(self.fs, self.N, self.Hs)
        self.H = self.Hc * H
        self.d = self.N**2
        self.b = 0.04

    def forward(self, x):
        """
        """
        N = x.size(0)
        Npatches = x.size(1)
        d = self.d
        H = self.H
        b = torch.ones(d, 1) * self.b
        c_h = torch.ones(d, 1) * self.Hc
        c_h = c_h.repeat(1, Npatches)
        b = b.repeat(1, Npatches)
        gamma = self.g
        beta = []

        yy = torch.mm(self.L, x)

        # dn
        y = yy
        sgn = torch.sign(y)
        y = torch.abs(y)

        e = y ** gamma
        denom = self.b + torch.mm(H, e)
        elog = torch.log(y)
        x = sgn * e / denom
        xim = x
        return yy, xim


class Layer4(nn.Module):
    """
    """

    def __init__(self):
        """
        """
        super(Layer4, self).__init__()
        self.fs = 64
        self.ns = 2
        self.no = 4
        self.tw = 1
        self.N = 16
        (W, ind) = make_wavelet_kernel_2(
            self.N, self.ns, self.no, self.tw, 0, 1)
        self.L = W
        self.ind = ind
        self.d = W.size(0)

        self.Hs_ini = 0.24
        self.b_ini = 500
        self.Hc_ini = 0.9
        self.Hs = self.Hs_ini
        self.Hc = self.Hc_ini
        H_spatial = kernel_s_wavelet_spatial(ind, self.fs, [self.Hs])

    def foward(self, x):
        """
        """


def kernel_s_wavelet_spatial(ind, fs, sigmas_x):
    """
    """
    n_sub = len(ind)
    dim = ind[0][0]
    no = 0
    while dim == ind[no+1][0]:
        no += 1
    ns = (n_sub-2)/no
    d = sum([i[0]*i[1] for i in ind])
    dim_s = len(sigmas_x)
    domainS = torch.zeros(d, 6)

    for i in range(0, d):
        coords, bands = coor_s_pyr(i, ind)
        e = coords[0]
        o = coords[1]
        nb = bands

        if e == 0:
            f = fs/2 + (fs/4)*(math.sqrt(2) - 1)
            o = 45
            domainS[i, :] = torch.Tensor(
                [coords[3]/fs, coords[2]/fs, f*math.cos(o*math.pi/180),
                 f*math.sin(o*math.pi/180), e, 0])
        elif (e > 0) and (e < ns+1):
            f = fs/2**(e+1) + 0.5*(fs/2**e - fs/2**(e+1))
            o = 180*(coords[1]-1)/no
            domainS[i, :] = torch.Tensor(
                [(2**(e-1))*coords[3]/fs, (2**(e-1))*coords[2]/fs,
                 f*math.cos(o*math.pi/180), f*math.sin(o*math.pi/180), e,
                 180*(coords[1]-1)/no])
        else:
            f = 0
            o = 0
            domainS[i, :] = torch.Tensor(
                [(2**(e-1))*coords[3]/fs, (2**(e-1))*coords[2]/fs,
                 f*math.cos(o*math.pi/180), f*math.sin(o*math.pi/180), e, 0])
    Hx = torch.eye(d)
    for nbandf in range(0, n_sub):
        indi_f = pyrBandIndices(ind, nbandf)
        indi_c = indi_f
        nband_c = nbandf
        x = domainS[indi_f, 0].repeat(len(indi_c), 1)
        y = domainS[indi_f, 1].repeat(len(indi_c), 1)

        xp = domainS[indi_c, 0].repeat(len(indi_f), 1).T
        yp = domainS[indi_c, 1].repeat(len(indi_f), 1).T

        xpp = domainS[indi_c, 0].reshape(ind[nband_c]).T
        ypp = domainS[indi_c, 1].reshape(ind[nband_c]).T

        dxp = xpp[0, 1] - xpp[0, 0]
        dyp = ypp[1, 0] - ypp[0, 0]

        sigma_x2 = torch.zeros(len(indi_c), len(indi_c)) + sigmas_x[0]**2
        sigma_x3 = torch.zeros(len(indi_c), len(indi_c)) + sigmas_x[0]**3

        import numpy as np
        np.set_printoptions(precision=6)

        delta = ((x-xp)**2) + ((y-yp)**2)
        dxyp = dxp*dyp
        term2 = 1/(2*math.pi*sigma_x2)
        term3 = torch.exp(-delta/(2*sigma_x2))
        res = dxyp*term2*term3
        for i in range(len(indi_c)):
            Hx[indi_f, indi_c[i]] = res[:, i]
    return Hx


def coor_s_pyr(p, ind):
    """
    """
    s = np.array(ind).shape
    num_bands = len([s for s in ind if s == ind[0]]) - 1
    num_scales = int((len(ind) - 2) / num_bands)
    di = [ind[i+1][0] - ind[i][0] for i in range(0, len(ind)-1)]
    n = [i for i in range(len(di)) if di[i] != 0]
    num_or = n[0]
    band = 0
    for i in range(0, s[0]):
        indices = pyrBandIndices(ind, i)
        pp = np.where((indices-p) == 0)[0]
        if pp.size == 0:
            band += 1
        else:
            break
    scale = math.floor((band)/num_or) + \
        (abs(((band)/num_or) - math.floor((band)/num_or)) > 0.)

    orientation = 0
    if scale > 0 and scale <= num_scales:
        orientation = num_or - (num_or*scale - band)

    co = coord(pp[0]+1, ind[band])

    c = [scale, orientation, co[1], co[0]]
    return c, band


def coord(pos, N):
    """
    """
    x = []
    NN = N[0] * N[1]
    if ((pos >= 0 and pos < NN+1)):
        D = len(N)
        p = N[0]
        x.append(math.ceil(pos/p))

        for j in range(2, D-1):
            pe = p[0:j-1]

        pe = p
        x.append(math.ceil((pos-(x[0]-1)*p)))
    return x


def pyrBandIndices(pind, band):
    """
    """
    ind = 0
    for i in range(band):
        ind += pind[i][0] * pind[i][1]
    indices = np.arange(ind, ind+pind[band][0]*pind[band][1])

    return indices


def make_wavelet_kernel_2(N, ns, no, tw, ns_not_used, resid=1):
    """
    """
    pyr = expert_models.SteerableWavelet(ns, no-1, tw)
    rand_im = torch.randn(N, N)
    pyramid, high_pass = pyr(rand_im.unsqueeze(0))
    num_values = sum([stage.numel() for stage in pyramid]) + high_pass.numel()
    W = torch.zeros(num_values, N*N)

    for i in range(N*N):
        delta = torch.zeros(N*N, 1)
        delta[i] = 1
        wd, hpd = pyr(delta.view(1, N, N))
        wd_vec = torch.cat([high_pass.flatten()] +
                           [stage.flatten() for stage in wd])
        W[:, i] = wd_vec
    ind = [[high_pass.size(1), high_pass.size(2)]]
    for i, stage in enumerate(pyramid[:-1]):
        indices = [[stage.size(2), stage.size(3)]] * stage.size(1)
        ind += indices
    ind += [[pyramid[-1].size(1), pyramid[-1].size(2)]]
    return W, ind


def make_2d_gauss_kernel(fs, N, sigma):
    """
    """
    d = N ** 2
    (x, y, t, fx, fy, ft) = spatio_temp_freq_domain(N, N, 1, fs, fs, 1)
    dx = x[0, 1] - x[0, 0, ]
    X = x.T.repeat(1, d).flatten().reshape(d, d)
    Y = y.repeat(1, N).repeat(N, 1)
    Xotros = X.T
    Yotros = Y.T

    delta = (X-Xotros)**2 + (Y-Yotros)**2
    H = (dx**2)*(1/(2*math.pi*sigma**2))*torch.exp(-delta/(2*sigma**2))
    dHds = (dx**2)*(1/(math.pi*sigma**2))*((delta-2*sigma**2) /
                                           (2*sigma**3))*torch.exp(-delta/(2*sigma**2))

    return H, dHds


def spatio_temp_freq_domain(Ny, Nx, Nt, fsx, fsy, fst):
    """
    """
    int_x = Nx/fsx
    int_y = Ny/fsy
    int_t = Nt/fst

    x = torch.zeros(Ny, Nx*Nt)
    y = torch.zeros(Ny, Nx*Nt)
    t = torch.zeros(Ny, Nx*Nt)

    fot_x = torch.linspace(0, int_x, Nx+1)
    fot_x = fot_x[:-1]
    fot_x = fot_x.repeat(Ny, 1)

    fot_y = torch.linspace(0, int_y, Ny+1)
    fot_y = fot_y[:-1]
    fot_y = fot_y.repeat(Nx, 1).T

    fot_t = torch.ones(Ny, Nx)

    val_t = torch.linspace(0, int_t, Nt+1)
    val_t = val_t[:-1]

    for i in range(Nt):
        x = metefot(x, fot_x, i+1, 1)
        y = metefot(y, fot_y, i+1, 1)
        t = metefot(t, val_t[i]*fot_t, i+1, 1)

    [fx, fy] = freqspace([Ny, Nx])

    fx = fx*fsx/2
    fy = fy*fsy/2

    ffx = torch.zeros(Ny, Nx*Nt)
    ffy = torch.zeros(Ny, Nx*Nt)
    ff_t = torch.zeros(Ny, Nx*Nt)

    fot_fx = fx
    fot_fy = fy
    fot_t = torch.ones(Ny, Nx)

    [ft, ft2] = freqspace([Nt, Nt])
    val_t = ft*fst/2

    for i in range(Nt):
        ffx = metefot(ffx, fot_fx, i+1, 1)
        ffy = metefot(ffy, fot_fy, i+1, 1)
        ff_t = metefot(ff_t, val_t[i]*fot_t, i+1, 1)

    return x, y, t, ffx, ffy, ff_t


def metefot(sec, foto, N, ma):
    """
    """
    ss = foto.size()
    fil = ss[0]
    col = ss[1]
    s = sec.size()
    Nfot = s[1] / col

    if N > Nfot:
        sec = [sec, foto]
    else:
        if ma == 1:
            sec[:, (N-1)*col:N*col] = foto
    # if incorrect results finish this function.
    return sec


def freqspace(N):
    """
    returns 2-d frequency range vectors for N[0] x N[1] matrix
    """
    f1 = (torch.arange(0, N[0], 1)-math.floor(N[0]/2))*(2/N[0])
    f2 = (torch.arange(0, N[1], 1)-math.floor(N[1]/2))*(2/N[1])
    F2, F1 = torch.meshgrid([f1, f2])
    return F1, F2


def make_csf_kernel(fs, N, obliq=1, channel=1):
    """
    """
    (gN, CSSFO, CSFT, OE) = csfsso(fs, N, 330.74, 7.28, 0.837, 1.809, 1, 6.664)
    if obliq == 1:
        norma1 = torch.sqrt(torch.sum(gN**2))
        gN = gN / torch.sum(gN)
    # TODO: if channel>1
    H, _, _ = convmtx2(gN, N, N)
    marco = torch.zeros(2*N-1, 2*N-1)
    dims = [math.ceil(N/2)-1, math.ceil(N/2)+N-1]
    marco[dims[0]:dims[1], dims[0]:dims[1]] = 1
    filas = marco.flatten()
    hh = H[filas > 0]
    hh = torch.mm(torch.diag(1 / hh.sum(axis=1)), hh)

    return hh, gN


def csfsso(fs, N, g, fm, l, s, w, os):
    """
    """
    (fx, fy) = freqspace([N, N])
    fx = fx*fs/2
    fy = fy*fs/2

    f = torch.sqrt(fx**2 + fy**2)
    f[f == 0] = 0.0001

    a = torch.exp(-(f/fm))
    b = torch.exp(-(f**2/s**2))

    csft = g*(torch.exp(-(f/fm))-l*torch.exp(-(f**2/s**2)))
    oe = 1-w*(4*(1-torch.exp(-(f/os)))*fx**2*fy**2)/(f**4)
    csfsso = csft * oe
    h = fsamp2(csfsso)

    return h, csfsso, csft, oe


def fsamp2(f1, f2=None, hd=None, siz=None):
    """
    """
    # Test if f1 has imaginary component, if not add it
    if f1.size(-1) != 2:
        f1_imag = torch.zeros(f1.size())
        f1 = torch.stack([f1, f1_imag], axis=-1)
    hd = fourier.ifftshift(f1, dim=(-3, -2))
    h = fourier.fftshift(torch.ifft(hd, signal_ndim=2), dim=(-3, -2))
    real, imag = torch.unbind(h, -1)
    h = torch.rot90(real, 2, dims=(-2, -1))  # rotate for filter2 in matlab
    return h


def convmtx2(H, M, N):
    """
    """
    rows, cols = H.size()
    (blockMat, convVec) = make_block_mat(H, M)
    blockNonZeros = rows * M
    height, width = blockMat.size()

    totalNonZeros = cols * N * blockNonZeros
    convMatRows = (N+cols-1)*(M+rows-1)
    convMatCols = N*width

    convMat = torch.zeros(convMatRows, convMatCols)
    rowInd = torch.arange(0, blockMat.size(0))
    colInd = torch.arange(0, blockMat.size(1))
    rowInd = [0, blockMat.size(0)]
    colInd = [0, blockMat.size(1)]

    for k in range(N):
        convMat[rowInd[0]:rowInd[1], colInd[0]:colInd[1]] = blockMat
        rowInd = [r + rows+M-1 for r in rowInd]
        colInd = [c + width for c in colInd]

    return convMat, blockMat, convVec


def make_block_mat(convKern, inRows):
    """
    """
    convVec = make_conv_Vec(convKern, inRows)
    blockMat = torch.zeros(convVec.size(0), inRows)
    for k in range(inRows):
        blockMat[:, k] = convVec
        convVec = fourier.roll(convVec, shift=1, dim=0)

    return blockMat, convVec


def make_conv_Vec(convKern, inRows):
    """
    """
    convVec = torch.cat([convKern, torch.zeros(inRows-1, convKern.size(1))])
    convVec_flat = convVec.T.flatten()
    return convVec_flat
