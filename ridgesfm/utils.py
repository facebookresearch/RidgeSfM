# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pykeops.torch import LazyTensor
import torch
import cv2
import torchvision
import random
import torch.nn.functional as F
import pykeops.torch
import faiss
import numpy as np
import math
import glob
import PIL.Image
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections
import eval_scannet
import torchvision.transforms.functional as TTF
import time
import superpoint
from mobilenetv3 import *


def visdom_scatter(vis, xyz, rgb, win='3d', markersize=3, title=''):
    """
    Plot a point cloud using visdom handle vis
    """
    rgb = rgb.detach()
    rgb -= rgb.min()
    rgb /= rgb.max() / 255 + 1e-10
    rgb = rgb.floor().cpu().numpy()
    vis.scatter(
        xyz.detach().cpu().numpy(),
        opts={
            'markersize': markersize,
            'markercolor': rgb,
            'title': title,
            'markerborderwidth': 0},
        win=win)


def KMeans(x, K=1000, Niter=1, verbose=True):
    """
    KMeans using PyKeOps
    https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
    """
    N, D = x.shape  # Number of samples, dimension of the ambient space
    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[torch.randperm(N)[:K], :].clone()  # Random subset of points
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).float()  # Class weights
        for d in range(D):
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print(
            "K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
            Niter, end - start, Niter, (end - start) / Niter))

    return cl, c


def match_kcrosscheck(a, b, k):
    """
    Match rows of a and b.
    Find the k nearest neighbors in the other tensor.
    Return the pairs that satisify cross-check, i.e. (i,j) if
    A[i,:] is one of the k-nearest neighbors of B[j,:] and
    B[j,:] is one of the k-nearest neighbors of A[i,:].
    """
    a=a.numpy()
    b=b.numpy()
    index = faiss.IndexFlatL2(b.shape[1])
    index.add(b)
    D, I = index.search(a, k)
    index = faiss.IndexFlatL2(b.shape[1])
    index.add(a)
    DD, II = index.search(b, k)
    u, c = np.unique(np.vstack([np.hstack([np.arange(a.shape[0] * k)[:, None] // k, I.reshape(-1, 1)]), np.hstack(
        [II.reshape(-1, 1), np.arange(b.shape[0] * k)[:, None] // k])]), return_counts=True, axis=0)
    return torch.from_numpy(u[c == 2])


def match_kcrosscheck_binary(a, b, k):
    """
    Match rows of a and b for binary descriptors, i.e. ORB.
    Find the k nearest neighbors in the other tensor.
    Return the pairs that satisify cross-check, i.e. (i,j) if
    A[i,:] is one of the k-nearest neighbors of B[j,:] and
    B[j,:] is one of the k-nearest neighbors of A[i,:].
    """
    a=a.numpy()
    b=b.numpy()
    index = faiss.IndexBinaryFlat(b.shape[1] * 8)
    index.add(b)
    _, I = index.search(a, k)
    index = faiss.IndexBinaryFlat(b.shape[1] * 8)
    index.add(a)
    _, II = index.search(b, k)
    u, c = np.unique(np.vstack([np.hstack([np.arange(a.shape[0] * k)[:, None] // k, I.reshape(-1, 1)]), np.hstack(
        [II.reshape(-1, 1), np.arange(b.shape[0] * k)[:, None] // k])]), return_counts=True, axis=0)
    return torch.from_numpy(u[c == 2])


def matches_ratiotest(a, b, q=0.7):
    """
    Match rows of a and b using the ratio test.
    """
    a=a.numpy()
    b=b.numpy()
    index = faiss.IndexFlatL2(b.shape[1])
    index.add(b)
    Da, Ia = index.search(a, 2)
    filtera = (Da[:, 0] < q * Da[:, 1]).nonzero()[0]
    Ia = Ia[filtera, 0]
    Da = Da[filtera, 0]
    index = faiss.IndexFlatL2(b.shape[1])
    index.add(a)
    Db, Ib = index.search(b, 2)
    Ib = Ib[:, 0]
    Ib[Db[:, 0] > q * Db[:, 1]] = -1
    cross_check = (Ib[Ia] == filtera)
    matches = np.vstack([filtera, Ia, Da.astype(np.int64)]).T[cross_check]
    return torch.from_numpy(matches)


def matches_ratiotest_binary(a, b, q=0.7):
    """
    Match rows of a and b using the ratio test.
    """
    a=a.numpy()
    b=b.numpy()
    index = faiss.IndexBinaryFlat(b.shape[1] * 8)
    index.add(b)
    Da, Ia = index.search(a, 2)
    filtera = (Da[:, 0] < q * Da[:, 1]).nonzero()[0]
    Ia = Ia[filtera, 0]
    Da = Da[filtera, 0]
    index = faiss.IndexBinaryFlat(b.shape[1] * 8)
    index.add(a)
    Db, Ib = index.search(b, 2)
    Ib = Ib[:, 0]
    Ib[Db[:, 0] > q * Db[:, 1]] = -1
    cross_check = (Ib[Ia] == filtera)
    matches = np.vstack([filtera, Ia, Da.astype(np.int64)]).T[cross_check]
    return torch.from_numpy(matches)


class Sequential(torch.nn.Sequential):
    def __add__(self, x):
        r = Sequential()
        for m in self:
            r.append(m)
        for m in x:
            r.append(m)
        return r

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self

class UNet(torch.nn.Module):
    """
    A U-Net type network c.f. https://arxiv.org/abs/1505.04597
    built using MobileNet inverted residual blocks.
    join can be 'add', 'concat', or 'learnt'
    """

    def __init__(
            self,
            dimension,
            n_planes,
            size,
            stride,
            pad,
            reps,
            join='concat',
            hs=True,
            t=4,
            noise=0):
        super(UNet, self).__init__()
        self.dimension = dimension
        self.n_planes = n_planes
        self.join = join
        self.noise = noise

        self.a = torch.nn.ModuleList()
        self.b = torch.nn.ModuleList()
        self.c = torch.nn.ModuleList()

        m = Sequential()
        for i in range(reps):
            m.add(
                InvertedResidual(n_planes[0], t * n_planes[0], n_planes[0], 3, 1, 1, hs))
        self.a.append(m)

        for a, b, sz, st, p in zip(
                n_planes[:-1], n_planes[1:], size, stride, pad):
            m = Sequential()
            m.add(torch.nn.Conv2d(a, b, sz, st, p, bias=False))
            for i in range(reps):
                m.add(InvertedResidual(b, t * b, b, 3, 1, 1, hs))
            self.a.append(m)

            self.b.insert(
                0, torch.nn.ConvTranspose2d(
                    b, a, sz, st, p, bias=False))

            m = Sequential()
            if join == 'learnt':
                assert False
            for i in range(reps):
                m.add(
                    InvertedResidual(
                        (2 if i == 0 and join == 'concat' else 1) *a, t * a, a, 3, 1, 1, hs))
            self.c.insert(0, m)
        for i in range(reps):
            self.a[-1].add(InvertedResidual(b, t * b, b, 3, 1, 1, hs))

    def forward(self, x):
        saved = []
        output = []
        for a in self.a:
            x = a(x)
            if self.training and self.noise > 0:
                saved.append(x * (1 + self.noise * torch.randn_like(x)))
            else:
                saved.append(x)
        output.append(saved.pop())
        for b, c in zip(self.b, self.c):
            x = b(x)
            p = saved.pop()
            if x.numel() < p.numel():
                if self.dimension == 1:
                    q = [p.size(2) - x.size(2)]
                    x = torch.nn.functional.pad(
                        x, (0, q[0] - 0), mode='replicate')
                elif self.dimension == 2:
                    q = [p.size(2) - x.size(2), p.size(3) - x.size(3)]
                    x = torch.nn.functional.pad(
                        x, (0, q[1], 0, q[0]), mode='replicate')
                elif self.dimension == 3:
                    q = [
                        p.size(2) - x.size(2),
                        p.size(3) - x.size(3),
                        p.size(4) - x.size(4)]
                    x = torch.nn.functional.pad(
                        x, (0, q[2], 0, q[1], 0, q[0]), mode='replicate')
            if self.join == 'add':
                x = x + p
            elif self.join in ['concat', 'learnt']:
                x = torch.cat([x, p], 1)
            x = c(x)
            if self.training and self.noise > 0:
                x = x * (1 + self.noise * torch.randn_like(x))
            output.append(x)
        output.reverse()
        return output[0]


def multipredict_loss(
        depth_gt,
        prediction_mu,
        prediction_sigma,
        lmbda):
    """
    Ridge-regression loss for training a depth prediction network with factors of variation.

    depth_gt: Batch x           1 x H x W
    prediction_sigma: Batch x n_multipred x H x W
    Express depth_gt as prediction_sigma.view(n_batch,n_multipred,-1).transpose(-2,-1)@beta
    lmbda: Penalty on size of regression coefficients
    Pixels with depth_gt==0 are excluded from the calculations.
    """
    rg = (prediction_sigma.var(1) - 1).abs()
    n_batch = depth_gt.size(0)
    n_multipred = prediction_sigma.size(1)
    n_pixels = depth_gt.numel()
    vis = (depth_gt > 0).float()  # (B,1,H,W)
    res = (depth_gt - prediction_mu) * vis
    X = (prediction_sigma * vis).view(n_batch, n_multipred, -1).transpose(-2, -1)
    l = torch.eye(n_multipred, device=depth_gt.device) * \
        lmbda * n_pixels / (n_multipred * n_batch)

    #Ridge regression
    # beta^hat ~ (X^t X)^{-1] X^t Y
    # we calculate beta transpose.
    # "beta_hat^t = Y^t X (X^t X)^t^{-1} = Y^t X (X^t X)^{-1}"
    betaT = res.view(
        n_batch, 1, -1) @ X @ torch.inverse(X.transpose(-2, -1) @ X + l)
    p2 = (betaT @ prediction_sigma.view(n_batch, n_multipred, -1)).view_as(depth_gt)

    s = {
        'mu': prediction_mu,
        'mu+sigma*beta': prediction_mu + p2,
        'mu loss': res.pow(2).sum() / n_pixels,
        'adjusted depth loss': (res - p2).mul(vis).pow(2).sum() / n_pixels,
        'sigma variance loss': rg.sum() / n_pixels,
        'beta loss': betaT.pow(2).sum() / n_pixels
    }
    s['loss'] = s['mu loss'] + s['adjusted depth loss'] + \
        s['sigma variance loss'] + lmbda * s['beta loss']
    return s


def align_point_clouds(A, B):
    """
    Umeyama without scale factor
    A and B two dimensional - align point clouds
            three dimensional - align batch of point clouds
            four dimensional - ...
    """
    with torch.no_grad():
        centroids = A.mean(-2, keepdim=True), B.mean(-2, keepdim=True)
        A_, B_ = A - centroids[0], B - centroids[1]
        C = A_.transpose(-2, -1) @ B_ / A.size(-2)
        V, _, W = torch.svd(C)
        d = torch.sign(torch.det(V) * torch.det(W))
        if V.dim() == 2:
            V[:, -1] *= d[None]
        if V.dim() == 3:
            V[:, :, -1] *= d[:, None]
        if V.dim() == 4:
            V[:, :, :, -1] *= d[:, :, None]
        R = V @ W.transpose(-2, -1)
        T = centroids[1] - centroids[0] @ R
    return R, T


def taitbryan_to_rot(x):
    """
    Tait-Bryan angles to rotation matrices.
    Input has shape (*,3)
    [ca -sa  0]          [1    0    0 ]           [cc   0   -sc]
    [sa  ca  0]    x     [0    cb  -sb]     x     [ 0   1   0  ]
    [0   0   1]          [0    sb   cb]           [sc   0   cc ]
    i.e. xy-rotation, yz-rotation, xz-rotation
    """
    a, b, c = x.split(1, -1)
    ca, cb, cc = a.cos(), b.cos(), c.cos()
    sa, sb, sc = a.sin(), b.sin(), c.sin()
    return torch.cat([
        ca * cc + sa * sb * sc, -sa * cb, -ca * sc + sa * sb * cc,
        sa * cc - ca * sb * sc, ca * cb, -sa * sc - ca * sb * cc,
        cb * sc, sb, cb * cc
    ], -1).view(*x.shape[:-1], 3, 3)


def rot_to_taitbryan(x):
    m = x.contiguous().view(-1, 9)
    a = torch.atan2(-m[:, 1], m[:, 4])
    b = torch.asin(m[:, 7])
    c = torch.atan2(m[:, 6], m[:, 8])
    return torch.stack([a, b, c], 1).view(*x.shape[:-1])


def dc_cumsum(a):
    """
    divide and conquour cumulative sum
    i.e. dc_cumprod(torch.arange(8)) or dc_cumprod(torch.randn(8,3))
    """
    b = a[0]
    s = b.shape
    A = [a[1::2]]
    l, a = [a[0]], a[1:-1:2] + a[2:-1:2]
    while len(a) % 2:
        A.append(a[0::2])
        a = a[:-1:2] + a[1::2]
    for a_ in a:
        b = b + a_
        l.append(b)
    l = torch.stack(l, 0)
    while len(A):
        l = torch.stack([l, l + A.pop()], 1).view(-1, *s)
    return l


def dc_cumprod(a):
    """
    divide and conquour cumulative matrix product
    i.e. dc_cumprod(torch.arange(1,9)[:,None,None]) or dc_cumprod(torch.randn(8,3,3))
    """
    b = a[0]
    s = b.shape
    A = [a[1::2]]
    l, a = [a[0]], a[1:-1:2] @ a[2:-1:2]
    while len(a) % 2:
        A.append(a[0::2])
        a = a[:-1:2] @ a[1::2]
    for a_ in a:
        b = b @ a_
        l.append(b)
    l = torch.stack(l, 0)
    while len(A):
        l = torch.stack([l, l @ A.pop()], 1).view(-1, *s)
    return l


def matplotlib_fig2npy(fig):
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def write_gif(name, frames, duration=0.1):
    import imageio
    imageio.mimwrite(name + '.gif', frames, 'GIF', duration=duration)


def write_mp4(name, frames, fps=10):
    import imageio
    imageio.mimwrite(name + '.mp4', frames, 'mp4', fps=fps)


class PairwiseRidgeSfM:
    @torch.no_grad()
    def __init__(self, s,
                 k=(0, 1),
                 reps=64,
                 weight_decay=1,
                 wdc=0.5,
                 precision=0.05,
                 verbose=False,
                 vis=None,
                 initial_K=3,
                 min_matches=16,
                 iters=3,
                 gr=0.2):
        torch.set_num_threads(1)
        self.s = s
        self.vis = vis
        self.gr = gr
        self.plot_num = int(time.time())
        self.N = s['p_depth_i'][0].size(0)  # dimension of factors of variation
        self.Q = int(reps)                  # RANSAC reps
        self.k = k                          # pair of frames (k[0],k[1])
        self.K = initial_K                  # number of matched keypoints
        self.active = torch.arange(reps)
        self.backup = {'k': self.k,
                       'maxK': 0,
                       'n_homog': 0,
                       }

        if len(s['matches'][k]) <= min_matches:
            return
        self.matches = s['matches'][k]

        self.backup['n_candidate_matches'] = len(self.matches)
        if verbose:
            print(len(self.matches), 'candidate matches')

        self.M = self.matches.size(0)  # total matches
        self.p_depth = torch.stack([
            s['p_depth_i'][k[0]][:, self.matches[:, 0]],
            s['p_depth_i'][k[1]][:, self.matches[:, 1]]])
        self.p_depth0 = torch.stack([
            s['p_depth0_i'][k[0]][0, self.matches[:, 0]],
            s['p_depth0_i'][k[1]][0, self.matches[:, 1]]])
        self.xyz = torch.stack([
            s['feature_xyz'][self.k[0]][self.matches[:, 0]],
            s['feature_xyz'][self.k[1]][self.matches[:, 1]]
        ])
        self.A0, self.B0 = self.p_depth0[:, :, None] * self.xyz

        self.backup['K'] = torch.zeros(reps, dtype=torch.int64)
        self.backup['screen coverage'] = torch.zeros(reps, dtype=torch.int64)
        self.backup['cs'] = torch.zeros(reps)
        self.backup['svd'] = torch.zeros(reps, 3)
        self.backup['idxs'] = [
            torch.zeros(
                0, dtype=torch.int64) for _ in range(reps)]
        self.backup['beta'] = torch.zeros(2, reps, self.N)
        self.backup['R'] = torch.zeros(reps, 3, 3)
        self.backup['T'] = torch.zeros(reps, 1, 3)
        self.backup['RT'] = torch.zeros(reps, 6)
        self.bkup = {}

        self.precision = precision
        self.weight_decay = weight_decay
        self.wdc = wdc
        self.verbose = verbose
        self.iters = iters
        self.init_idxs()
        self.cc = [(self.Q, self.K)]
        self.beta = torch.zeros(2, self.Q, self.N)
        self.plod()
        if self.verbose:
            self.plot()
        while self.grow():
            self.plod()
            if not self.prune():
                break
            if self.verbose:
                self.plot()
        self.backup['idx'] = self.backup['screen coverage'].argmax()
        ok = torch.nonzero(self.backup['screen coverage'] >=
                           self.backup['screen coverage'][self.backup['idx']] * 2 // 3)
        self.backup['ok idx'] = ok
        self.backup['ok idxs'] = torch.unique(
            torch.cat([self.backup['idxs'][i] for i in ok], 0))

    def plod(self):
        for _ in range(self.iters):
            self.calc_beta()
            self.calc_AB()
            self.calc_RT()
        self.bkup1()

    @torch.no_grad()
    def idx_foo(self):
        self.Ai, self.Bi = self.A0[self.idxs], self.B0[self.idxs]
        self.R, self.T = align_point_clouds(self.Ai, self.Bi)
        self.xyzi = self.xyz[:, self.idxs.flatten(), :].view(
            2, self.Q, self.K, 3)
        self.p_depthi = self.p_depth.view(2 * self.N, self.M)[ :, self.idxs.flatten()].view(
            2,self.N, self.Q, self.K).permute( 0, 2, 1, 3)
        self.p_depth0i = self.p_depth0[:, self.idxs.flatten()].view(
            2, self.Q, self.K)

    @torch.no_grad()
    def init_idxs(self):
        M = len(self.matches)
        idxs = []
        k = 0
        while k < self.Q:
            k_ = min(self.Q - k, M // self.K)
            k += k_
            i = torch.randperm(M)[:k_ * self.K].view(-1, self.K)
            idxs.append(i)
        self.idxs = torch.cat(idxs, 0)
        self.idxs_ = self.idxs
        if self.verbose:
            print('init idxs', self.idxs.shape)
        self.idx_foo()

    @torch.no_grad()
    def calc_beta(self):
        Q = self.Q
        N = self.N
        K = self.K
        xyz = [
            self.xyzi[0] @ self.R,  # Q,K,3
            self.xyzi[1]    # Q,K,3
        ]
        multidepth = [
            self.p_depthi[0].view(Q, N, 1, K).permute(0, 3, 2, 1),  # Q,K,1,N
            self.p_depthi[1].view(Q, N, 1, K).permute(0, 3, 2, 1)]  # Q,K,1,N

        #    Q,1,3      Q,K,1               Q,K,3
        Y = self.T + self.p_depth0i[0, :, :, None] * xyz[0] - \
            self.p_depth0i[1, :, :, None] * xyz[1]  # Q K 3
        Y = Y.view(Q, K * 3, 1)  # (Q,K3,1)
        X = torch.cat([
            # Q,K,1,N        Q,K,3,1
            -(multidepth[0] * xyz[0][:, :, :, None]
              ).reshape(Q, K * 3, N),  # (Q,K3,N)
            (multidepth[1] * xyz[1][:, :, :, None]).reshape(Q, K * 3, N)],  # (Q,K3,N)
            2)  # (Q,K3,2N)
        wd = self.weight_decay * (self.K**self.wdc) * torch.eye(2 * N)
        Xt = X.transpose(-2, -1)
        XtX = Xt @ X
        beta = torch.inverse(XtX + wd) @ Xt @ Y  # (Q,2N,1)
        self.beta = beta.view(Q, 2, N).permute(1, 0, 2)

    @torch.no_grad()
    def calc_AB(self):
        q = (self.beta[:, :, None, :] @ self.p_depthi).view(2,
                                                            self.Q, self.K, 1) + self.p_depth0i[:, :, :, None]
        q.clamp_(min=0)
        self.Ai, self.Bi = q * self.xyzi

    def calc_RT(self):
        self.R, self.T = align_point_clouds(self.Ai, self.Bi)
        self.delta = self.Ai @ self.R + self.T - self.Bi
        self.delta_norm_max = self.delta.norm(2, 2).max(1)[0]
        if self.verbose:
            print('*RT*',
                  self.delta_norm_max.sort()[0][:8],
                  'potential survivors',
                  self.delta_norm_max.le(self.precision).sum().item())

    @torch.no_grad()
    def bkup1(self):
        self.bkup['beta'] = self.beta
        self.bkup['idxs'] = self.idxs
        self.bkup['R'] = self.R
        self.bkup['T'] = self.T
        self.bkup['RT'] = torch.cat(
            [rot_to_taitbryan(self.R), self.T[:, 0]], 1)
        self.bkup['K'] = self.K

    def bkup2(self, ok):
        bad = ~ok
        h = self.active[bad]
        self.backup['K'][h] = self.bkup['K']
        self.backup['R'][h] = self.bkup['R'][bad]
        self.backup['T'][h] = self.bkup['T'][bad]
        self.backup['RT'][h] = self.bkup['RT'][bad]
        self.backup['beta'][:, h] = self.beta[:, bad]
        AA, BB = self.A0[self.bkup['idxs'][bad]
                         ], self.B0[self.bkup['idxs'][bad]]
        RR, TT = align_point_clouds(AA, BB)
        for i, j in zip(h, self.bkup['idxs'][bad]):
            self.backup['idxs'][i] = j
            A, B = self.A0[j], self.B0[j]
            R, T = align_point_clouds(A, B)
            A = A @ R  # +T #don't add T as we are subtracting the mean anyway
            A -= A.mean(0)
            B -= B.mean(0)
            self.backup['svd'][i] = torch.svd(A).S
            A = A.view(-1)
            B = B.view(-1)
            self.backup['cs'][i] = torch.nn.functional.cosine_similarity(
                A, B, 0).item()
            idxs = [self.s['feature_idxs'][self.k[l]]
                    [self.s['matches'][self.k][:, l][j]] for l in range(2)]
            self.backup['screen coverage'][i] = sum([len(torch.unique(
                l // (self.s['W'] * 10) * (self.s['W'] // 10) + l % self.s['W'] // 10)) for l in idxs])

    @torch.no_grad()
    def grow(self):
        q = (self.beta[:,:, None, :] @ self.p_depth[:, None, :, :]
              ).view(2, self.Q, self.M, 1) + self.p_depth0.view(2, 1, self.M, 1)
        q.clamp_(min=0)
        A, B = q * self.xyz.view(2, 1, self.M, 3)
        score = (A @ self.R +self.T - B).norm( 2, 2)

        self.K += max(1, int(self.K * self.gr))
        if self.K > self.matches.size(0):
            self.K = self.bkup['K']
            self.bkup1()
            self.bkup2(torch.zeros(self.Q, dtype=torch.bool))
            return False

        # Pick best matches, regardless of whether currently used or not
        _, self.idxs = score.topk(self.K, dim=1, largest=False)
        self.idx_foo()
        if self.verbose:
            print(
                'grow particles',
                self.Q,
                '#matched',
                self.K,
                'uniq',
                torch.unique(
                    self.idxs).numel())
        return True

    @torch.no_grad()
    def prune(self):
        ok = (self.delta_norm_max <= self.precision)
        self.bkup2(ok)
        self.active = self.active[ok]
        if ok.sum().item() < 1:
            if self.verbose:
                print('!!')
            return False
        self.Q = ok.sum().item()
        self.idxs = self.idxs[ok]
        self.R = self.R[ok]
        self.T = self.T[ok]
        self.Ai = self.Ai[ok]
        self.Bi = self.Bi[ok]
        self.xyzi = self.xyzi[:, ok]
        self.p_depthi = self.p_depthi[:, ok]
        self.p_depth0i = self.p_depth0i[:, ok]
        self.beta = self.beta[:, ok]
        self.cc.append((self.Q, self.K))
        self.delta_norm_max = self.delta_norm_max[ok]
        RT = torch.cat([rot_to_taitbryan(self.R), self.T[:, 0]], 1)
        RTrange = (RT.max(0)[0] - RT.min(0)[0]).norm(2)
        if self.verbose:
            print(
                'keep',
                ok.sum().item(),
                '/',
                ok.numel(),
                self.K,
                'uniq',
                torch.unique(
                    self.idxs).numel(),
                'RT range',
                RTrange.item())
        return True

    @torch.no_grad()
    def plot(self):
        s = self.s
        k = self.k
        idx = self.delta_norm_max.argmin()
        idxs_ = torch.stack([
            s['feature_idxs'][k[0]][self.matches[:, 0]],
            s['feature_idxs'][k[1]][self.matches[:, 1]]
        ], 1)
        p = torch.cat([s['pics'][k[0]], s['pics'][k[1]]], 1)
        p = torch.cat([p, p], 2)
        if False:
            p[:, :self.s['H'], self.s['W']:] *= s['color_fltr'][k[0]
                                                                ].view(1, self.s['H'], self.s['W'])
            p[:, self.s['H']:, self.s['W']:] *= s['color_fltr'][k[1]
                                                                ].view(1, self.s['H'], self.s['W'])
        p = p.view(3, -1)
        I1 = torch.unique(self.idxs)
        I2 = self.idxs[idx]
        for c, I in enumerate([torch.arange(len(idxs_)), I1, I2]):
            rgb = torch.eye(3)[:, 2 - c:3 - c]
            for i, (a, b) in enumerate(idxs_[I]):
                x1 = a.item() // self.s['W']
                y1 = a.item() % self.s['W'] + self.s['W']
                x2 = b.item() // self.s['W'] + self.s['H']
                y2 = b.item() % self.s['W'] + self.s['W']
                r = int(((x1 - x2)**2 + (y1 - y2)**2)**0.5)
                p[:, (2 * self.s['W'] * torch.linspace(x1, x2, r).round() +
                      torch.linspace(y1, y2, r).round()).long()] = rgb
        p = p.view(3, self.s['H'] * 2, self.s['W'] * 2)
        self.vis.image(p, win='2')
        torchvision.utils.save_image(
            p[None, :, :, :], f'pair_plot_{self.plot_num}.png')

        i = torch.rand(self.s['H'] * self.s['W']) < ( 0.01 if self.K < 30 else 0.1)
        A = s['xyz'][0] * (
            s['p_depth'][k[0]].view(
                self.N,self.s['H'] * self.s['W']).T
            @ self.beta[0, idx, :, None] + s['p_depth0'][k[0]].view(self.s['H'] *  self.s['W'],1))
        B = s['xyz'][0] * (
            s['p_depth'][k[1]].view(
                self.N,self.s['H'] * self.s['W']).T
            @ self.beta[1, idx, :, None] + s['p_depth0'][k[1]].view(self.s['H'] *  self.s['W'],1))
        CA, CB = s['pics'][k[0]].view(3, -1).T, s['pics'][k[1]].view(3, -1).T
        R, T = self.R[idx], self.T[idx]
        print(R, T)
        q = torch.linspace(0, 1, 30)[:, None]
        cam = torch.cat([
            torch.cat([q, 0.75 * q, 1.8 * q], 1),
            torch.cat([q, -0.75 * q, 1.8 * q], 1),
            torch.cat([-q, 0.75 * q, 1.8 * q], 1),
            torch.cat([-q, -0.75 * q, 1.8 * q], 1),
            torch.cat([2 * q - 1, q * 0 + 0.75, q * 0 + 1.8], 1),
            torch.cat([2 * q - 1, q * 0 - 0.75, q * 0 + 1.8], 1),
            torch.cat([q * 0 + 1, 0.75 * (2 * q - 1), q * 0 + 1.8], 1),
            torch.cat([q * 0 - 1, 0.75 * (2 * q - 1), q * 0 + 1.8], 1)], 0) * 0.1
        cam_ = torch.ones_like(cam)
        xyz = torch.cat([A[i] @ R + T, B[i]], 0)
        rgb = torch.cat([CA[i], CB[i]], 0)
        filtr = rgb.mean(1) > 0.05
        xyz, rgb = xyz[filtr], rgb[filtr]
        xyz = torch.cat([xyz, cam, cam @ R + T, torch.zeros(2, 3)], 0)
        rgb = torch.cat([rgb, cam_ *
                         torch.Tensor([[1, 0, 0]]), cam_ *
                         torch.Tensor([[0, 0.5, 0.5]]), torch.ones(2, 3)], 0)
        m = xyz[:-2].min(0)[0]
        M = xyz[:-2].max(0)[0]
        rng = (M - m).max()
        xyz[-2] = (M + m - rng) / 2
        xyz[-1] = (M + m + rng) / 2
        xyz[:, 1:] *= -1
        visdom_scatter(self.vis, xyz, rgb, markersize=2)


@torch.no_grad()
def pix_splat(
        pcl_centroids,
        c33,
        c_3,
        intrinsic,
        device='cpu',
        percolate=False,
        H=480,
        W=640,
        scale=2):
    pic = torch.zeros(H // scale * W // scale, 3, device=device)
    if device == 'cpu':
        q = pcl_centroids.clone()
    else:
        q = pcl_centroids.to(device)
    assert pcl_centroids._cdata != q._cdata
    q[:, :3] = (q[:, :3] - c_3.to(device)) @ c33.to(device).T
    q = q[q[:, 2] > 0.05]
    z = q[:, 2:3].clone()
    if len(z):
        q[:, :3] /= z
        z /= z.max() + 0.1
        xy, rgb = (q[:, :3] @ intrinsic[:2, :3].to(device).T /
                   scale).round(), q[:, 3:]
        f = (xy[:, 0] >= 0) * (xy[:, 0] < W // scale) * \
            (xy[:, 1] >= 0) * (xy[:, 1] < H // scale)
        xy, z, rgb = xy[f], z[f], rgb[f]
        if len(z):
            r = xy[:, 0] + xy[:, 1] * W + z.view(-1)
            r = r.argsort()
            xy, rgb = xy[r], rgb[r]
            r = np.unique(
                xy.detach().cpu().numpy(),
                return_index=True,
                axis=0)[1]
            r = torch.from_numpy(r).to(device)
            xy, rgb = xy[r], rgb[r]
            pic[(xy @ torch.FloatTensor([[1], [W // scale]]
                                        ).to(device)).view(-1).long()] = rgb

    pic = pic.view(H // scale, W // scale, 3).permute(2, 0, 1)
    if percolate:
        while True:
            active = (pic.mean(0, keepdim=True) > 0).float()
            if active.mean() in [0, 1]:
                break
            n = F.avg_pool2d(active[None].float(), 3, 1, 1)
            nrgb = F.avg_pool2d(pic[None], 3, 1, 1) / \
                (n + (n == 0).float()) * (1 - active)
            pic += nrgb[0]
    return pic.cpu()


class GlobalBundleAdjustment(torch.nn.Module):
    def __init__(self, s, vis=None, min_screen_coverage=30):
        torch.nn.Module.__init__(self)
        self.n_multipred2 = s['p_depth_i'][0].size(0)
        self.s = s
        self.vis = vis
        self.b = s['ransac matches']
        q = 2**math.ceil(math.log(s['n_views'], 2))
        self.c_3 = torch.nn.Parameter(torch.zeros(q, 1, 3))
        self.c33_euler = torch.nn.Parameter(torch.zeros(q, 3))
        self.beta = torch.zeros(s['n_views'], 1, 1, self.n_multipred2)
        self.beta = torch.nn.Parameter(self.beta)
        self.vertices = []
        self.edges = []

        self.max_sift = max([len(q) for q in s['feature_xyz']])
        self.register_buffer('pdepth_xyz', torch.stack([  # padded to shape (n_view,3,n_multipred2,max_sift)
            torch.cat([p.view(self.n_multipred2, -1, 1) * xyz[None, :, :], torch.zeros(
                self.n_multipred2, self.max_sift - p.size(1), 3)], 1).permute(2, 0, 1)
            for p, xyz in zip(
                s['p_depth_i'],
                s['feature_xyz'])
        ]))
        self.register_buffer('pdepth0_xyz', torch.stack([  # pad (n_view,3,1,max_sift)
            torch.cat([p.view(1, -1, 1) * xyz[None, :, :], torch.zeros(1,
                                                                       self.max_sift - p.size(1), 3)], 1).permute(2, 0, 1)
            for p, xyz in zip(
                s['p_depth0_i'],
                s['feature_xyz'])
        ]))

        self.good_matches = []
        for k in sorted(
                list(
                    self.b),
                key=lambda x: x[1] *
                s['n_views'] -
                x[0]):
            m = self.b[k]
            i = m['idx']
            if k[1] == k[0] + 1 or (
                m['screen coverage'][i] >= min_screen_coverage
                and
                m['RT'][i][:3].norm(2).item() < 1
                and
                m['cs'][i].item() > 0.9
            ):
                self.good_matches.append(k)
        self.register_buffer('matches',
                             torch.cat([s['matches'][k][self.b[k]['idxs'][self.b[k]['idx']],
                                                        :2] + self.max_sift * torch.LongTensor([k]) for k in self.good_matches]))
        self.register_buffer('matches_weight', torch.cat([torch.ones(len(self.b[k]['idxs'][self.b[k]['idx']])) / len(
            self.b[k]['idxs'][self.b[k]['idx']]) / len(self.good_matches) for k in self.good_matches]))
        self.match_offsets = torch.cumsum(torch.LongTensor(
            [0] + [len(self.b[k]['idxs'][self.b[k]['idx']]) for k in self.good_matches]), 0).tolist()

        self.n_matches = self.match_offsets[-1]
        self.match_offsets = dict(zip(
            self.good_matches,
            self.match_offsets[:-1]))
        particle_sc = []
        # particles=[[],[],[]]
        bundle = [[], [], []]
        self.n_particles = len(self.good_matches)
        for i, k in enumerate(self.good_matches):
            m = self.b[k]
            bundle[0].append(k)
            bundle[1].append(m['R'][m['idx']])
            bundle[2].append(m['T'][m['idx']])
            particle_sc.append(m['screen coverage'][m['idx']])

        self.register_buffer('bundle_k', torch.LongTensor(bundle[0]))
        self.register_buffer('bundle_R', torch.stack(bundle[1]))
        self.register_buffer('bundle_T', torch.stack(bundle[2]))
        self.active_bundle = torch.nn.Parameter(torch.zeros(len(bundle[0])))
        self.active_matches = torch.nn.Parameter(torch.zeros(self.n_matches))
        self.register_buffer('particle_sc', torch.LongTensor(particle_sc))
        self.particle_sc_sum = self.particle_sc.sum()
        if 'p_depth0' in s:
            self.set_plot_fraction(1e-4)
        else:
            self.set_plot_num_features(100)
        self.camera_cumulative = True

    def camera(self, device=None):
        s = self.s
        if self.camera_cumulative:
            s['c33'] = dc_cumprod(
                taitbryan_to_rot(
                    self.c33_euler))[
                :s['n_views']]
            s['c_3'] = dc_cumsum(self.c_3)[:s['n_views']]
        else:
            s['c33'] = taitbryan_to_rot(self.c33_euler)[:s['n_views']]
            s['c_3'] = self.c_3[:s['n_views']]
        if device is not None:
            s['c33'] = s['c33'].to(device)
            s['c_3'] = s['c_3'].to(device)

    @torch.no_grad()
    def check_depth(self):
        q = self.s['p_depth0'] + (self.beta[:, 0].cpu() @ self.s['p_depth'].view(
            -1, self.n_multipred2, self.s['H'] * self.s['W'])).view(-1, 1, self.s['H'], self.s['W'])
        res = eval_scannet.eval_depth(q, self.s['depth'])
        print(res['mse_depth']**0.5, res['abs_depth'])

    def plot_depth(self):
        q = self.s['p_depth0'] + (self.beta[:, 0].cpu() @ self.s['p_depth'].view(
            -1, self.n_multipred2, self.s['H'] * self.s['W'])).view(-1, 1, self.s['H'], self.s['W'])
        write_mp4('depth_', (255 * q[:, 0] /
                             q.max()).byte(), 30 // s['frameskip'])

    def make_optim(self):
        self.optim = [
            torch.optim.Adam([self.beta], lr=1e-4, weight_decay=1e-3),
            torch.optim.Adam([self.active_bundle, self.active_matches], lr=1e-3),
            torch.optim.Adam([self.c_3, self.c33_euler], lr=1e-3)
        ]

    def forward(self):
        s = self.s
        self.camera()
        s['p_feature_xyz'] = (
            self.beta @ self.pdepth_xyz +
            self.pdepth0_xyz)[
            :,
            :,
            0].permute(
            0,
            2,
            1).contiguous()
        s['p_feature_xyz'] = s['p_feature_xyz'] @ s['c33'] + s['c_3']
        s['p_match_delta'] = (
            s['p_feature_xyz'].view(-1, 3)[self.matches[:, 0]]
            - s['p_feature_xyz'].view(-1, 3)[self.matches[:, 1]]).norm(2, dim=1)

    def gravity_loss(self):
        s = self.s
        return [['gravity', 1e-3 * (1 - s['c33'][:, 1, 1]).mean()]]

    def bundle_loss(self, i=100000000000):
        s = self.s
        c33 = s['c33'][self.bundle_k[:i]]
        c_3 = s['c_3'][self.bundle_k[:i]]
        return [['bundle',
                 ((c33[:, 0] - self.bundle_R[:i] @ c33[:, 1]).view(-1, 9).abs().sum(1) + (
                     c_3[:, 0] - self.bundle_T[:i] @ c33[:, 1] - c_3[:,1]).abs().view(
                         -1, 3).sum(1)).mul(self.particle_sc[:i]).sum() / self.particle_sc[:i].sum()]]

    def match_err_loss(self):
        sap = torch.sigmoid(self.active_matches)
        return [
            ['match_err', (self.s['p_match_delta'] * sap).sum() / self.n_particles],
            ['sap', 0.3 * (1 - sap).mean()]
        ]

    def camera_movement_loss(self):
        if self.camera_cumulative:
            return [['movement', 1e1 *
                     self.c33_euler.pow(2).mean() +
                     1e1 *
                     self.c_3.pow(2).mean()]]
        else:
            return [['movement',
                     1e1 * (self.c33_euler[:-1] - self.c33_euler[1:]).pow(2).mean()
                     +
                     1e1 * (self.c_3[:-1] - self.c_3[1:]).pow(2).mean()]]

    def scene_loss(self, verbose=False):
        l = (
            self.gravity_loss()
            +
            self.bundle_loss()
            +
            self.match_err_loss()
            +
            self.camera_movement_loss()
        )
        # beta is regulated by weight decay, not an autograd loss
        loss = sum([x[1] for x in l])
        assert loss.item() == loss.item()
        if verbose:
            l.sort(key=lambda x: -x[1])
            print(' '.join(['{0}:{1:+.1e}'.format(k, v)
                            for k, v in l]), '*', '{0:+.1e}'.format(loss))
            print('active bundle',
                  self.active_bundle.min().item(),
                  self.active_bundle.mean().item(),
                  self.active_bundle.max().item())
            print('active matches',
                  self.active_matches.min().item(),
                  self.active_matches.mean().item(),
                  self.active_matches.max().item())
        return loss

    def bundle_init(self, i, verbose=False):
        for x in self.optim:
            x.zero_grad()
        self.forward()
        l = (
            self.gravity_loss()
            +
            self.bundle_loss(i)
            +
            self.camera_movement_loss()
        )
        loss = sum([x[1] for x in l])
        assert loss.item() == loss.item()
        if verbose:
            l.sort(key=lambda x: -x[1])
            print(' '.join(['{0}:{1:+.1e}'.format(k, v)
                            for k, v in l]), '*', '{0:+.1e}'.format(loss))
            print('active bundle',
                  self.active_bundle.min().item(),
                  self.active_bundle.mean().item(),
                  self.active_bundle.max().item())
            print('active matches',
                  self.active_matches.min().item(),
                  self.active_matches.mean().item(),
                  self.active_matches.max().item())
        loss.backward()
        for x in self.optim:
            x.step()
        return loss.item()

    def flub(self, n=4):
        for i in range(1, 2**n + 1):
            for x in self.optim:
                x.zero_grad()
            self.forward()
            loss = self.scene_loss(verbose=(i == 2**n))
            loss.backward()
            for x in self.optim:
                x.step()
        return loss.item()

    def set_plot_num_features(self, f=10):
        s = self.s
        if 'pics' not in s:
            s['pics'] = torch.stack([TTF.to_tensor(PIL.Image.open(x))
                                     for x in torch.load(s['file_name'])['color']])
        self.p_depth_i = []
        self.p_depth0_i = []
        self.xyz_i = []
        self.pics_i = []
        for pd, pd0, idx, p in zip(
                s['p_depth_i'], s['p_depth0_i'], s['feature_idxs'], s['pics']):
            q = torch.cat([torch.randperm(pd.size(1))
                           for _ in range(f // pd.size(1) + 1)])[:f]
            self.p_depth_i.append(pd[:, q])
            self.p_depth0_i.append(pd0[:, q])
            q = idx[q]
            self.xyz_i.append(s['xyz'][0, q])
            self.pics_i.append(p.view(3, self.s['H'] * self.s['W'])[:, q].T)
        for k in ['p_depth_i', 'p_depth0_i', 'xyz_i', 'pics_i']:
            setattr(self, k, torch.stack(getattr(self, k)))

    def set_plot_fraction(self, f=1e-4, crop=5):
        s = self.s
        f = int(f * self.s['H'] * self.s['W'])
        self.p_depth_i = []
        self.p_depth0_i = []
        self.xyz_i = []
        self.pics_i = []
        for pd, pd0, p, x in zip(
                s['p_depth'], s['p_depth0'], s['pics'], s['color_fltr']):
            x = torch.nonzero(x).view(-1)
            x = x[torch.randperm(len(x))[:f]]
            self.pics_i.append(p.view(3, self.s['H'] * self.s['W'])[:, x].T)
            self.xyz_i.append(s['xyz'][0, x])
            self.p_depth_i.append(pd.view(-1, self.s['H'] * self.s['W'])[:, x])
            self.p_depth0_i.append(
                pd0.view(
                    1,
                    self.s['H'] *
                    self.s['W'])[
                    :,
                    x])
            if len(x) < f:
                assert False  # pad with nan?
        for k in ['p_depth_i', 'p_depth0_i', 'xyz_i', 'pics_i']:
            setattr(self, k, torch.stack(getattr(self, k)))

    @torch.no_grad()
    def plot_cam(self, i=-1):
        s = self.s
        if i == -1:
            i = s['n_niews']
        self.camera('cpu')
        a, b = [], []
        q = torch.linspace(0, 1, 10)[:, None]
        cam = torch.cat([
            torch.cat([q, 0.75 * q, 1.8 * q], 1),
            torch.cat([q, -0.75 * q, 1.8 * q], 1),
            torch.cat([-q, 0.75 * q, 1.8 * q], 1),
            torch.cat([-q, -0.75 * q, 1.8 * q], 1),
            torch.cat([2 * q - 1, q * 0 + 0.75, q * 0 + 1.8], 1),
            torch.cat([2 * q - 1, q * 0 - 0.75, q * 0 + 1.8], 1),
            torch.cat([q * 0 + 1, 0.75 * (2 * q - 1), q * 0 + 1.8], 1),
            torch.cat([q * 0 - 1, 0.75 * (2 * q - 1), q * 0 + 1.8], 1)], 0) * 0.4
        cam_ = torch.ones_like(cam)
        for j in range(i):
            q = j / (s['n_views'] - 1)
            a.append(cam @ self.s['c33'][j] + self.s['c_3'][j])
            b.append(
                cam_ * torch.Tensor([[max(1 - 2 * q, 0), min(2 * q, 2 - 2 * q), max(2 * q - 1, 0)]]))
        a.append(torch.ones(2, 3))
        b.append(torch.ones(2, 3))
        a = torch.cat(a, 0)
        b = torch.cat(b, 0)
        m = a[:-2].min(0)[0]
        M = a[:-2].max(0)[0]
        rng = (M - m).max()
        a[-2] = (M + m - rng) / 2
        a[-1] = (M + m + rng) / 2
        visdom_scatter(self.vis, a, b, win='3d recon')

    @torch.no_grad()
    def plot(self):
        s = self.s
        self.camera('cpu')
        q = s['p_match_delta'].sort()[0][::100]
        self.vis.line(q, win='p_match_delta_sorted')

        v = torch.arange(s['n_views'])
        a = (self.beta[:, 0].cpu() @ self.p_depth_i +
             self.p_depth0_i).transpose(-2, -1) * self.xyz_i
        a = a @ s['c33'] + s['c_3']
        a = [a[v].view(-1, 3)]
        b = [self.pics_i[v].view(-1, 3)]

        q = torch.linspace(0, 1, 30)[:, None]
        cam = torch.cat([
            torch.cat([q, 0.75 * q, 1.8 * q], 1),
            torch.cat([q, -0.75 * q, 1.8 * q], 1),
            torch.cat([-q, 0.75 * q, 1.8 * q], 1),
            torch.cat([-q, -0.75 * q, 1.8 * q], 1),
            torch.cat([2 * q - 1, q * 0 + 0.75, q * 0 + 1.8], 1),
            torch.cat([2 * q - 1, q * 0 - 0.75, q * 0 + 1.8], 1),
            torch.cat([q * 0 + 1, 0.75 * (2 * q - 1), q * 0 + 1.8], 1),
            torch.cat([q * 0 - 1, 0.75 * (2 * q - 1), q * 0 + 1.8], 1)], 0) * 0.4
        cam_ = torch.ones_like(cam)

        for i in range(0, s['n_views'], 20):
            q = i / (s['n_views'] - 1)
            a.append(cam @ self.s['c33'][i] + self.s['c_3'][i])
            b.append(
                cam_ * torch.Tensor([[max(1 - 2 * q, 0), min(2 * q, 2 - 2 * q), max(2 * q - 1, 0)]]))
        a.append(torch.ones(2, 3))
        b.append(torch.ones(2, 3))
        a = torch.cat(a, 0)
        b = torch.cat(b, 0)
        m = a[:-2].min(0)[0]
        M = a[:-2].max(0)[0]
        rng = (M - m).max()
        a[-2] = (M + m - rng) / 2
        a[-1] = (M + m + rng) / 2
        visdom_scatter(self.vis, a, b, win='3d recon', markersize=2)

    def cam_pose(self, fn=None):
        self.camera('cpu')
        pose = torch.cat([
            torch.cat([self.s['c33'].transpose(-2, -1), self.s['c_3'].transpose(-2, -1)], 2),
            torch.eye(4)[None, 3:].expand(self.s['c33'].size(0), 1, 4)], 1)
        res = eval_scannet.eval_pose(pose, self.s['pose'])
        if fn is not None:
            with open(fn, 'w') as f:
                json.dump(res, f)
        for k, v in res.items():
            if 'err_interp' in k:
                print(k, v)

    def pose(self, fn=None):
        self.camera('cpu')
        res = eval_scannet.eval_pth(
            self.s['file_name'],
            {
                'depth': self.s['p_depth0'] + (self.beta[:, 0].cpu() @ self.s['p_depth'].view(
                    -1, self.n_multipred2, self.s['H'] * self.s['W'])).view(-1, 1, self.s['H'], self.s['W']),
                'pose': torch.cat([
                    torch.cat([self.s['c33'].transpose(-2, -1), self.s['c_3'].transpose(-2, -1)], 2),
                    torch.eye(4)[None, 3:].expand(self.s['c33'].size(0), 1, 4)], 1),
                'K': torch.stack([self.s['intrinsic'][:3, :3] for _ in range(self.s['n_views'])])
            })

        if fn is not None:
            with open(fn, 'w') as f:
                json.dump(res, f)
        try:
            for k in [
                'pcl_point_cloud_rmse_loss_scaled',
                'pcl_rmse_depth_scaled',
                'pcl_cam_center_err_interp_scaled',
                    'pcl_cam_angle_err_interp_scaled']:
                print(k, res[k])
        except BaseException:
            print(res)

    @torch.no_grad()
    def vid_plot(
            self,
            filename,
            n_particles=int(1e5),
            frame_rate=24,
            trim=5,
            cam_skip=3,
            device='cpu',
            scale=2):
        scene = self
        s = self.s
        scene.camera('cpu')
        s['p_depth_adjusted'] = (scene.beta[:, 0].cpu() @ s['p_depth'].view(
            s['n_views'], -1, s['H'] * s['W'])).view(s['n_views'], 1, s['H'], s['W'])
        trim = 5
        pcl = (s['p_depth0'] + s['p_depth_adjusted'])
        pcl = (F.avg_pool2d(pcl,scale)[:, :, trim:-trim, trim:-trim].reshape(
            -1, (s['H'] // scale - 2 * trim) * (s['W'] // scale - 2 * trim),  1
        ) * F.avg_pool2d(s['xyz'][0].T.view(3,s['H'], s['W']), 2)[:, trim:-trim, trim:-trim].reshape(
            1, 3, (s['H'] // scale - 2 * trim) * (s['W'] // scale - 2 * trim)).permute(0, 2, 1)) @ s['c33'] + s['c_3']
        pcl = torch.cat([pcl.reshape(-1,3), F.avg_pool2d(s['pics'], scale)[:, :, trim:-trim, trim:-trim].reshape(
            -1, 3, (s['H'] // scale - 2 * trim) * (s['W'] // scale - 2 * trim)).permute(0, 2, 1).reshape(-1, 3)], 1)

        q = torch.linspace(0, 1, 100)[:, None]
        cam = torch.cat([
            torch.cat([2 * q - 1, q * 0 + 0.75, q * 0 + 1.8], 1),
            torch.cat([2 * q - 1, q * 0 - 0.75, q * 0 + 1.8], 1),
            torch.cat([q * 0 + 1, 0.75 * (2 * q - 1), q * 0 + 1.8], 1),
            torch.cat([q * 0 - 1, 0.75 * (2 * q - 1), q * 0 + 1.8], 1)], 0) * 0.2
        cam_ = torch.ones_like(cam)

        cam_gt = [[], []]
        cam_pr = [[], []]
        for i in range(0, s['n_views'], cam_skip):
            q = i / (s['n_views'] - 1)
            cam_pr[0].append(cam @ s['c33'][i] + s['c_3'][i])
            cam_gt[0].append(cam @ s['pose'][i, :3, :3].T +
                             s['pose'][i, None, :3, 3])
            cam_pr[1].append(
                cam_ * torch.Tensor([[max(1 - 2 * q, 0), min(2 * q, 2 - 2 * q), max(2 * q - 1, 0)]]))
            cam_gt[1].append(
                cam_ * torch.Tensor([[max(1 - 2 * q, 0), min(2 * q, 2 - 2 * q), max(2 * q - 1, 0)]]))
        cam_gt = torch.cat([torch.cat(x, 0) for x in cam_gt], 1)
        cam_pr = torch.cat([torch.cat(x, 0) for x in cam_pr], 1)

        pcl_centroids = KMeans(
            pcl.to(device),
            int(n_particles),
            20,
            True)[1].to('cpu')
        print(pcl.shape, pcl_centroids.shape)
        pics = []
        for idx in range(s['n_views']):
            pic = torch.cat([
                torch.cat([
                    pix_splat(
                        pcl_centroids,
                        s['c33'][idx],
                        s['c_3'][idx],
                        s['intrinsic'],
                        percolate=True,
                        device=device,
                        H=s['H'],
                        W=s['W'],
                        scale=scale
                    ),
                    F.avg_pool2d(s['pics'][idx], scale)], 1),
                torch.cat([
                    pix_splat(
                        cam_pr,
                        s['c33'][idx],
                        s['c_3'][idx],
                        s['intrinsic'],
                        device=device,
                        H=s['H'],
                        W=s['W'],
                        scale=scale
                    ),
                    pix_splat(
                        cam_gt,
                        s['pose'][idx, :3, :3].T,
                        s['pose'][idx, None, :3, 3],
                        s['intrinsic'],
                        device=device,
                        H=s['H'],
                        W=s['W'],
                        scale=scale
                    ),
                ], 1)], 2)
            pics.append(pic.mul(255.9).byte().permute(1, 2, 0))
        write_mp4(filename, pics, frame_rate)

        xyz = pcl_centroids[:, 0:3] * 40
        rgb = pcl_centroids[:, 3:6].mul(255.999).floor().byte()
        with open(f'{filename}.ply', 'w') as ply_file:
            print(f"""ply
format ascii 1.0
comment RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty
comment 3DV 2020
comment Benjamin Graham, David Novotny
comment Facebook AI Research
element vertex {len(pcl_centroids)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face 0
property list uchar int vertex_indices
end_header""", file=ply_file)
            for a, b in zip(xyz, rgb):
                print(
                    f"{a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]}",
                    file=ply_file)


def plot_ransac_edge(s, k):
    b = s['ransac matches'][k]
    idx = b['idx']
    print(k,
          'K',
          b['K'][idx].item(),
          '/',
          len(b['ok idxs']),
          b['svd'][idx].mul(10).long().tolist(),
          'cs',
          b['cs'][idx].item(),
          'sc',
          b['screen coverage'][idx].item())
    p = torch.cat([s['pics'][k[0]], s['pics'][k[1]]], 2).view(3, -1)
    for i, m in enumerate(s['matches'][k]):
        x1 = s['feature_idxs'][k[0]][m[0]].item() // s['W']
        y1 = s['feature_idxs'][k[0]][m[0]].item() % s['W']
        x2 = s['feature_idxs'][k[1]][m[1]].item() // s['W']
        y2 = s['feature_idxs'][k[1]][m[1]].item() % s['W'] + s['W']
        r = int(((x1 - x2)**2 + (y1 - y2)**2)**0.5)
        if i in b['ok idxs']:
            p[:, (2 * s['W'] * torch.linspace(x1, x2, r).round() +
                  torch.linspace(y1, y2, r).round()).long()] = torch.rand(3, 1)
    p = p.view(3, s['H'], s['W'] * 2)
    plt.figure(figsize=(20, 8))
    plt.imshow(p.permute(1, 2, 0))
    plt.show()


def plot_flow_depth_normals(s):
    frames = []
    for i in tqdm(range(s['n_views'] - 1), desc='draw sift match frames'):
        cf = s['color_fltr'][i].view(s['H'], s['W'], 1)
        fig = plt.figure(figsize=(s['W'] / 72, s['H'] / 72))
        ax = fig.add_subplot(111)
        plt.subplots_adjust(
            left=0,
            bottom=0,
            right=1,
            top=1,
            wspace=0,
            hspace=0)
        ax.set_xlim(0, 639)
        ax.set_ylim(479, 0)
        ax.set_axis_off()
        ax.imshow(s['pics'][i].permute(1, 2, 0) * cf)
        ax.scatter(
            s['feature_idxs'][i] %
            s['W'],
            s['feature_idxs'][i] //
            s['W'],
            c='r',
            s=5)
        k = (i, i + 1)
        if k in s['matches']:
            m = s['matches'][k]

            x = np.stack([
                s['feature_idxs'][i][m[:, 0]].numpy() % s['W'],
                s['feature_idxs'][i + 1][m[:, 1]].numpy() % s['W'],
                s['feature_idxs'][i + 1][m[:, 1]].numpy() + np.nan], 1).reshape(-1)
            y = np.stack([
                s['feature_idxs'][i][m[:, 0]].numpy() // s['W'],
                s['feature_idxs'][i + 1][m[:, 1]].numpy() // s['W'],
                s['feature_idxs'][i + 1][m[:, 1]].numpy() + np.nan], 1).reshape(-1)
            ax.plot(x, y, linewidth=1, c='w')
            ax.scatter(x[::3], y[::3], c='g', s=40)
        frames.append(matplotlib_fig2npy(fig))
    write_mp4('flow', frames, fps=30 // s['frameskip'])

    print('plot depth')
    write_mp4('depth', (255 *
                        s['p_depth0'][:, 0] /
                        s['p_depth0'].max()).byte(), 30 //
              s['frameskip'])

    print('plot normals')
    q = s['normal'][:, 3:-3, 3:-3]
    q = q - q.min()
    q *= 255 / q.max()
    write_mp4('normal', q.byte(), 30 // s['frameskip'])


@torch.no_grad()
def load_scene(cfg, NET):
    file_name = sorted(glob.glob(
        f'{cfg.scenes}/*/*frameskip={cfg.scene.frameskip}-*.pth'))[cfg.scene.n]

    print(file_name)
    s = torch.load(file_name)
    s['frameskip'] = cfg.scene.frameskip
    s['file_name'] = file_name
    s['n_views'] = len(s['color'])
    if 'intrinsic' not in s:
        print('default intrinsic matrix')
        s['intrinsic'] = np.array(
            [[578, 0, 319.5, 0], [0, 578, 239.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    if 'depth' in s:
        s['depth'] = torch.stack([torch.from_numpy(np.array(PIL.Image.open(
            img), copy=True)).float()[None] / s['depth scale factor'] for img in s['depth']])
    x = (np.linspace(0, s['H'] - 1, s['H'], dtype=np.float32) -
         s['intrinsic'][1, 2].item()) / s['intrinsic'][1, 1].item()
    y = (np.linspace(0, s['W'] - 1, s['W'], dtype=np.float32) -
         s['intrinsic'][0, 2].item()) / s['intrinsic'][0, 0].item()
    s['xyz'] = torch.from_numpy(np.ascontiguousarray(np.vstack([
        np.tile(y[None, None, :], (1, s['H'], 1)),
        np.tile(x[None, :, None], (1, 1, s['W'])),
        np.ones((1, s['H'], s['W']))]))).float().view(3, -1).t()[None]
    s['scene file'] = file_name

    s['pil_imgs'] = [
        PIL.Image.open(f) for f in tqdm(
            s['color'],
            desc='Load images')]
    if 'crop' in s:
        s['pil_imgs'] = [
            img.crop(
                (s['crop'][0],
                 s['crop'][1],
                    s['crop'][0] +
                    s['W'],
                    s['crop'][1] +
                    s['H'])) for img in s['pil_imgs']]
    s['imgs'] = [np.array(img, copy=True) for img in s['pil_imgs']]
    s['pics'] = torch.stack([torch.from_numpy(img).permute(
        2, 0, 1).float() / 255 for img in s['imgs']])
    if cfg.unet.resize > 0:
        s['pics_shrunk'] = [
            np.array(
                TTF.resize(
                    img,
                    cfg.unet.resize),
                copy=True) for img in s['pil_imgs']]
        s['pics_shrunk'] = torch.stack([torch.from_numpy(img).permute(
            2, 0, 1).float() / 255 for img in s['pics_shrunk']])
    else:
        s['pics_shrunk'] = s['pics']
    del s['pil_imgs']

    torch.cuda.empty_cache()
    s['p_depth'] = []
    s['p_depth0'] = []
    for x in tqdm(
            s['pics_shrunk'].split(
                4 if cfg.unet.m == 32 else 2),
            desc='predicting depth planes'):
        x = x.to(cfg.device) * 2 - 1
        x = NET(x)
        a = x[:, 1:]
        b = x[:, :1].clamp(min=0.3)
        s['p_depth0'].append(b.cpu())
        if cfg.n_multipred2<cfg.n_multipred:
            a = a.view(a.size(0), -1, s['H'] * s['W']).permute(0, 2, 1)
            U, S, V = torch.svd(a)
            a = (U * S[:, None, :])
            a = a.view(a.size(0), s['H'], s['W'], -1).permute(0, 3, 1, 2)
            a = a[:, :cfg.n_multipred2]
        s['p_depth'].append(a.cpu())
    s['p_depth'] = torch.cat(s['p_depth'], 0)
    s['p_depth0'] = torch.cat(s['p_depth0'], 0)
    torch.cuda.empty_cache()

    sxyz = s['xyz'].to(cfg.device)

    def f(pd0):
        xyz = sxyz.view(1, s['H'], s['W'], 3) * \
            pd0.view(-1, s['H'], s['W'], 1).to(cfg.device)
        z = 0
        for k in [1, 2]:
            q = []
            for i, j in [[k, k], [-k, k], [-k, -k],
                         [k, -k], [k, 0], [0, k], [-k, 0], [0, -k]]:
                q.append(xyz[:, 8 + i:s['H'] - 8 + i, 8 +
                             j:s['W'] - 8 + j] - xyz[:, 8:-8, 8:-8])
            for i, j in [
                [
                    0, 1], [
                    1, 2], [
                    2, 3], [
                    3, 0], [
                        4, 5], [
                            5, 6], [
                                6, 7], [
                                    7, 4]]:
                z += torch.cross(q[i], q[j])
        z = z / z.norm(2, -1, keepdim=True)
        return z.cpu()
    s['normal'] = torch.cat([f(pd0) for pd0 in tqdm(
        s['p_depth0'].split(4), desc='calculating normals')], 0)
    s['features_tensors'] = collections.defaultdict(list)
    s['features_idxs'] = collections.defaultdict(list)
    s['features_xyz'] = collections.defaultdict(list)
    s['features_grid_sample_xy'] = collections.defaultdict(list)
    s['features_screen_xy'] = collections.defaultdict(list)
    s['features'] = []

    print('calculate color_fltr')
    s['color_fltr'] = torch.ones(s['n_views'], s['H'], s['W'])
    s['color_fltr'][:, :15, :] = 0
    s['color_fltr'][:, -15:, :] = 0
    s['color_fltr'][:, :, :15] = 0
    s['color_fltr'][:, :, -15:] = 0
    if cfg.max_distance_filter_features > 0:
        s['color_fltr'] *= s['p_depth0'].view(-1,
                                              s['H'],
                                              s['W']) < cfg.max_distance_filter_features
    s['color_fltr'] = s['color_fltr'].view(-1, s['H'] * s['W'])
    if cfg.sift_features:
        s['features'].append('sift')
        cv2_sift = cv2.SIFT_create(
            contrastThreshold=0.02, edgeThreshold=20)
        for img, cf in tqdm(zip(s['imgs'], s['color_fltr']),
                            desc='calculating sift features'):
            kp, des = cv2_sift.detectAndCompute(img, None)
            des = torch.from_numpy(des)
            idxs = torch.LongTensor(
                [int(p.pt[1]) * s['W'] + int(p.pt[0]) for p in kp])
            keep = cf[idxs]
            s['features_idxs']['sift'].append(idxs[keep])
            s['features_tensors']['sift'].append(des[keep])
            a = torch.FloatTensor([p.pt + (1,) for p in kp])
            if a.size:
                a[:, 0] = (a[:, 0] - s['intrinsic'][0, 2]) / \
                    s['intrinsic'][0, 0]
                a[:, 1] = (a[:, 1] - s['intrinsic'][1, 2]) / \
                    s['intrinsic'][1, 1]
            s['features_xyz']['sift'].append(a[keep])
            a = torch.FloatTensor(
                [[p.pt[0] * 2 / s['W'] - 1, p.pt[1] * 2 / s['H'] - 1] for p in kp])
            s['features_grid_sample_xy']['sift'].append(a[keep])
            a = torch.FloatTensor([p.pt for p in kp])
            s['features_screen_xy']['sift'].append(a[keep])
    if cfg.orb_features:
        s['features'].append('orb')
        cv2_orb = cv2.ORB_create(nfeatures=1000)
        for img, cf in tqdm(zip(s['imgs'], s['color_fltr']),
                            desc='calculating orb features'):
            kp = cv2_orb.detect(img, None)
            kp, des = cv2_orb.compute(img, kp)
            if des is None:
                des = torch.zeros(0, 32, dtype=torch.uint8)
                kp = []
                idxs = torch.LongTensor([])
                keep = torch.LongTensor([])
            else:
                des = torch.from_numpy(des)
                idxs = torch.LongTensor(
                    [int(p.pt[1]) * s['W'] + int(p.pt[0]) for p in kp])
                keep = cf[idxs]
                idxs = idxs[keep]
                des = des[keep]
            s['features_idxs']['orb'].append(idxs)
            s['features_tensors']['orb'].append(des)
            a = torch.FloatTensor([p.pt + (1,) for p in kp])
            if a.size(0):
                a[:, 0] = (a[:, 0] - s['intrinsic'][0, 2]) / \
                    s['intrinsic'][0, 0]
                a[:, 1] = (a[:, 1] - s['intrinsic'][1, 2]) / \
                    s['intrinsic'][1, 1]
            s['features_xyz']['orb'].append(a[keep])
            a = torch.FloatTensor(
                [[p.pt[0] * 2 / s['W'] - 1, p.pt[1] * 2 / s['H'] - 1] for p in kp])
            s['features_grid_sample_xy']['orb'].append(a[keep])
            a = torch.FloatTensor([p.pt for p in kp])
            s['features_screen_xy']['orb'].append(a[keep])
    if cfg.superpoint_features:
        sp = superpoint.SuperPoint({}).to(cfg.device)
        s['features'].append('superpoint')
        for img, cf in tqdm(zip(s['pics'], s['color_fltr']),
                            desc='calculating superpoint features'):
            r = sp({'image': img.mean(0)[None, None].to(cfg.device)})
            kp = r['keypoints'][0].cpu()
            des = r['descriptors'][0].T.cpu(
                memory_format=torch.contiguous_format)
            idxs = kp[:, 1].long() * s['W'] + kp[:, 0].long()
            s['features_idxs']['superpoint'].append(idxs)
            s['features_tensors']['superpoint'].append(des)
            a = torch.cat([kp, torch.ones_like(kp[:, :1])], 1)
            if a.size(0):
                a[:, 0] = (a[:, 0] - s['intrinsic'][0, 2]) / \
                    s['intrinsic'][0, 0]
                a[:, 1] = (a[:, 1] - s['intrinsic'][1, 2]) / \
                    s['intrinsic'][1, 1]
            s['features_xyz']['superpoint'].append(a)
            a = kp * 2 / torch.FloatTensor([s['W'], s['H']]) - 1
            s['features_grid_sample_xy']['superpoint'].append(a)
            a = kp
            s['features_screen_xy']['superpoint'].append(a)

    k = 5
    delta_threshold = 0.2
    if len(cfg.depth_change_filter_features) > 1:
        print('filter features based on depth change')
        delta = (torch.nn.functional.max_pool2d(s['p_depth0'],
                                                1 + 2 * k,
                                                1,
                                                k) + torch.nn.functional.max_pool2d(-s['p_depth0'],
                                                                                    1 + 2 * k,
                                                                                    1,
                                                                                    k)).view(-1,
                                                                                             s['H'] * s['W'])
        for feat in s['features']:
            if feat in cfg.depth_change_filter_features:
                for i, f in enumerate(delta):
                    keep = f[s['features_idxs'][feat][i]] < delta_threshold
                    if len(keep):
                        s['features_tensors'][feat][i] = s['features_tensors'][feat][i][keep]
                        s['features_idxs'][feat][i] = s['features_idxs'][feat][i][keep]
                        s['features_xyz'][feat][i] = s['features_xyz'][feat][i][keep]
                        s['features_grid_sample_xy'][feat][i] = s['features_grid_sample_xy'][feat][i][keep]
                        s['features_screen_xy'][feat][i] = s['features_screen_xy'][feat][i][keep]
    if len(cfg.background_filter_features) > 1:
        print('filter background features')
        delta = (torch.nn.functional.max_pool2d(
            s['p_depth0'], 1 + 2 * k, 1, k) - s['p_depth0']).view(-1, s['H'] * s['W'])
        for feat in s['features']:
            if feat in cfg.background_filter_features:
                for i, f in enumerate(delta):
                    keep = f[s['features_idxs'][feat][i]] < delta_threshold
                    if True:
                        s['features_tensors'][feat][i] = s['features_tensors'][feat][i][keep]
                        s['features_idxs'][feat][i] = s['features_idxs'][feat][i][keep]
                        s['features_xyz'][feat][i] = s['features_xyz'][feat][i][keep]
                        s['features_grid_sample_xy'][feat][i] = s['features_grid_sample_xy'][feat][i][keep]
                        s['features_screen_xy'][feat][i] = s['features_screen_xy'][feat][i][keep]

    s['feature_idxs'] = [torch.cat(
        [s['features_idxs'][j][i] for j in s['features']], 0) for i in range(s['n_views'])]
    s['feature_xyz'] = [torch.cat(
        [s['features_xyz'][j][i] for j in s['features']], 0) for i in range(s['n_views'])]
    s['feature_grid_sample_xy'] = [
        torch.cat(
            [
                s['features_grid_sample_xy'][j][i] for j in s['features']],
            0) for i in range(
            s['n_views'])]
    s['feature_screen_xy'] = [torch.cat(
        [s['features_screen_xy'][j][i] for j in s['features']], 0) for i in range(s['n_views'])]
    s['p_depth_i'] = [p.view(-1, s['H'] * s['W'])[:, i]
                      for p, i in zip(s['p_depth'], s['feature_idxs'])]
    s['p_depth0_i'] = [p.view(-1, s['H'] * s['W'])[:, i]
                       for p, i in zip(s['p_depth0'], s['feature_idxs'])]
    return s


def make_edge(a, b):
    return tuple(sorted([a, b]))


def prematch_scene(cfg, s,thread_pool):
    random.seed(0)
    edges = set()
    q = min(cfg.prematch_multiplier *
            s['n_views'], s['n_views'] * (s['n_views'] - 1) // 2)
    if cfg.prematch_multiplier <= 2:
        step_range = [1]
    else:
        step_range = [1, 2, 3, 5, 8, 13, 21][:cfg.prematch_multiplier * 3 // 4]
    for step in step_range:
        for i in range(0, s['n_views'] - step):
            edges.add((i, i + step))
    while len(edges) < q:
        i = random.randint(0, s['n_views'] - 1)
        j = random.randint(0, s['n_views'] - 1)
        if i != j:
            edges.add(make_edge(i, j))
    s['matches'] = {}
    def match(e):
        offsets = [0, 0]
        mm=[]
        for feat in s['features']:
            if e[0] + 1 == e[1]:
                if 'kcrosscheck' in cfg.contiguous_matching and feat in [
                        'sift', 'r2d2', 'superpoint']:
                    def match_fn(x, y):
                        return match_kcrosscheck(x, y, int(cfg.contiguous_matching.split('_')[-1]))
                elif 'kcrosscheck' in cfg.contiguous_matching and feat in ['orb']:
                    def match_fn(x, y):
                        return match_kcrosscheck_binary(x, y, int(cfg.contiguous_matching.split('_')[-1]))
                elif 'ratiotest' in cfg.contiguous_matching and feat in ['sift', 'r2d2', 'superpoint']:
                    def match_fn(x, y):
                        return matches_ratiotest(x, y, float(cfg.contiguous_matching.split('_')[-1]))
                elif 'ratiotest' in cfg.contiguous_matching and feat in ['orb']:
                    def match_fn(x, y):
                        return matches_ratiotest_binary(x, y, float(cfg.contiguous_matching.split('_')[-1]))
            else:
                if 'kcrosscheck' in cfg.matching and feat in [
                        'sift', 'r2d2', 'superpoint']:
                    def match_fn(x, y):
                        return match_kcrosscheck(x, y, int(cfg.matching.split('_')[-1]))
                elif 'kcrosscheck' in cfg.matching and feat in ['orb']:
                    def match_fn(x, y):
                        return match_kcrosscheck_binary(x, y, int(cfg.matching.split('_')[-1]))
                elif 'ratiotest' in cfg.matching and feat in ['sift', 'r2d2', 'superpoint']:
                    def match_fn(x, y):
                        return matches_ratiotest(x, y, float(cfg.matching.split('_')[-1]))
                elif 'ratiotest' in cfg.matching and feat in ['orb']:
                    def match_fn(x, y):
                        return matches_ratiotest_binary(x, y, float(cfg.matching.split('_')[-1]))
            if s['features_tensors'][feat][e[0]].numel(
            ) and s['features_tensors'][feat][e[1]].numel():
                m = match_fn(s['features_tensors'][feat][e[0]], s['features_tensors'][feat][e[1]])
                # print(m.shape,'matches')
                if m.numel():
                    m[:, 0] += offsets[0]
                    m[:, 1] += offsets[1]
                    mm.append(m[:, :2])
                    offsets[0] += s['features_tensors'][feat][e[0]].shape[0]
                    offsets[1] += s['features_tensors'][feat][e[1]].shape[0]
        mm = torch.cat(mm, 0)
        fltr = torch.from_numpy(np.nonzero(cv2.findFundamentalMat(
            (s['feature_screen_xy'][e[0]][mm[:, 0]]).numpy(),
            (s['feature_screen_xy'][e[1]][mm[:, 1]]).numpy(),
            cv2.LMEDS, 5)[1])[0])
        s['matches'][e]=mm[fltr]
    #list(tqdm(thread_pool.imap_unordered(match, edges),desc='FAISS matching'))
    for e in tqdm(edges,desc='FAISS matching'):
        match(e)
