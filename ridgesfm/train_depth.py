# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.checkpoint
import math
import numpy as np
import glob
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TTF
import random
import torchvision
import torch.utils.data
import torch.nn.functional as F
import os
import PIL.Image
import torch
import utils
from mpl_toolkits.mplot3d import Axes3D
import logging
import hydra
log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="depth")
def my_app(cfg):
    log.info(f'{cfg}')
    log.info(f'Working directory : {os.getcwd()}')
    n_features = [
        1 * cfg.unet.m,
        2 * cfg.unet.m,
        3 * cfg.unet.m,
        4 * cfg.unet.m,
        5 * cfg.unet.m]
    if cfg.unet.ssp == 'tailored':
        size_stride_pad = [
            [[4, 4], [4, 4], [3, 4], [3, 4]],
            [2, 2, 2, 2],
            [[1, 1], [1, 1], [0, 1], [0, 1]]]
    else:
        size_stride_pad = [[4] * 6, [2] * 6, [1] * 6]
    data = {}
    for m in ['train', 'test']:
        def load(folder):
            tf = torch.load(folder)
            x = (np.linspace(0.5, 480 - 0.5, 480, dtype=np.float32) -
                 tf['intrinsic'][1, 2]) / tf['intrinsic'][1, 1]
            y = (np.linspace(0.5, 640 - 0.5, 640, dtype=np.float32) -
                 tf['intrinsic'][0, 2]) / tf['intrinsic'][0, 0]
            tf['xyz'] = [torch.from_numpy(np.ascontiguousarray(np.vstack([
                np.tile(y[None, None, :], (1, 480, 1)),
                np.tile(x[None, :, None], (1, 1, 640)),
                np.ones((1, 480, 640))]))).float()]
            tf['xyz'] = [a.view(3, -1).t()[None] for a in tf['xyz']]
            tf['folder'] = folder
            tf['n_views'] = len(tf['depth'])
            return tf
        data[m] = [load(folder) for folder in sorted(
            glob.glob(f'../data/scannet/{m}/*/files.pth'))]
    data['train'] = data['train'][:1413]

    class dataset(torch.utils.data.Dataset):
        def __init__(self, data, reps):
            torch.utils.data.Dataset.__init__(self)
            self.scenes = data
            self.reps = reps
            self.cj = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)

        def __len__(self):
            return self.reps

        def rgbCHW(self, k, i):
            img = self.scenes[k]['color'][i]
            img = PIL.Image.open(img)
            img = TTF.resize(img, 240)
            if len(self.scenes) > 100:
                img = self.cj(img)
            return TTF.to_tensor(img) * 2 - 1

        def depth(self, k, i):
            img = self.scenes[k]['depth'][i]
            img = torch.from_numpy(
                np.array(
                    PIL.Image.open(img),
                    copy=True)).float()[None] / 1000
            return img

        def cm(self, k, i):
            return self.scenes[k]['camera matrices'][i]

        def co(self, k, i):
            return self.scenes[k]['camera offsets'][i]

        def random_scene(self):
            return random.randint(0, len(self.scenes) - 1)

        def random_frame(self, k):
            return random.randint(0, self.scenes[k]['n_views'] - 1)

        def random_frames(self, k, n):
            return torch.randint(0, self.scenes[k]['n_views'] - 1, (n,))

        def __getitem__(self, k):
            kk = self.random_scene()
            v = self.random_frames(kk, cfg.n_views)
            pics = torch.stack([self.rgbCHW(kk, vv) for vv in v])
            depth = torch.stack([self.depth(kk, vv) for vv in v])
            if k % 2 == 1:
                pics = torch.flip(pics, (3,))
                depth = torch.flip(depth, (3,))
            return {
                'pics': pics,
                'depth': depth,
            }
    train_dataloader = torch.utils.data.DataLoader(
        dataset(data['train'], 100 * cfg.n_batch),
        batch_size=cfg.n_batch,
        shuffle=False,
        num_workers=18, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        dataset(data['test'], 10 * cfg.n_batch),
        batch_size=cfg.n_batch,
        shuffle=False,
        num_workers=10, pin_memory=True)

    net = torch.nn.Sequential(
        utils.conv_4x4_bn(
            3,
            cfg.unet.m // 2,
            2),
        utils.conv_4x4_bn(
            cfg.unet.m // 2,
            cfg.unet.m,
            2),
        utils.UNet(
            2,
            n_features,
            *size_stride_pad,
            cfg.unet.depth,
            checkpoint=True,
            join=cfg.unet.join,
            hs=cfg.unet.hs,
            t=cfg.unet.t,
            noise=cfg.unet.noise),
        torch.nn.Conv2d(
            cfg.unet.m,
            1 + cfg.n_multipred,
            1,
            1,
            0),
        torch.nn.Upsample(
            scale_factor=8))

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, torch.nn.ConvTranspose2d):
            n = m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    if os.path.exists('net.pth'):
        epoch, state = torch.load('net.pth', map_location=cfg.device)
        log.info(f'restart training {epoch}')
        net.load_state_dict(state)
    else:
        log.info('start training')
        epoch = 0

    class NET(torch.nn.Module):
        def __init__(self, net):
            torch.nn.Module.__init__(self)
            self.net = net

        def forward(self, depth_gt, x):
            p = self.net(x)
            prediction_mu = p[:, :1]
            prediction_sigma = p[:, 1:]
            sigma_variance_loss = (prediction_sigma.var(1) - 1).abs()
            n_batch = depth_gt.size(0)
            n_multipred = prediction_sigma.size(1)
            n_pixels = depth_gt.numel()
            vis = (depth_gt > 0).float()  # (B,1,H,W)
            res = (depth_gt - prediction_mu) * vis
            X = (prediction_sigma * vis).view(n_batch,
                                      n_multipred, -1).transpose(-2, -1)
            l = torch.eye(n_multipred, device=depth_gt.device) * \
                cfg.lambda_multipred * n_pixels / (n_multipred * n_batch)
            betaT = res.view(
                n_batch, 1, -1) @ X @ torch.inverse(X.transpose(-2, -1) @ X + l)
            sigma_dot_beta = (betaT @ prediction_sigma.view(n_batch, n_multipred, -1)).view_as(depth_gt)
            return (
                res.pow(2).sum() / n_pixels,
                (res - sigma_dot_beta).mul(vis).pow(2).sum() / n_pixels,
                sigma_variance_loss.sum() / n_pixels,
                cfg.lambda_multipred * betaT.pow(2).sum() / n_pixels
            )

    net = torch.nn.DataParallel(NET(net)).to(cfg.device)

    if cfg.optim.o == 'adam':
        optim = torch.optim.Adam(net.parameters(), lr=cfg.optim.lr)
    else:
        optim = torch.optim.SGD(
            net.parameters(),
            lr=cfg.optim.lr,
            momentum=0.9,
            weight_decay=cfg.optim.wd)
    for epoch in range(epoch + 1, 1000000):
        total_loss = {
            'mu loss': 0,
            'fine-tuned loss': 0,
            'sigma_variance loss': 0,
            'beta loss': 0,
            'loss': 0}
        net.train()
        for _ in range(1):
            torch.cuda.empty_cache()
            for s in train_dataloader:
                optim.zero_grad()
                for k in ['pics', 'depth']:
                    s[k] = s[k].view(-1, *s[k].shape[2:]).to(cfg.device)
                losses = net(s['depth'], s['pics'])
                losses = [x.mean() for x in losses]
                losses.append(sum(losses))
                losses[-1].backward()
                optim.step()
                if epoch <= 3:
                    print(f"{losses[0].item():.1e}    {losses[1].item():.1e}")
                total_loss['mu loss'] += losses[0].item() / 100
                total_loss['fine-tuned loss'] += losses[1].item() / 100
                total_loss['sigma_variance loss'] += losses[2].item() / 100
                total_loss['beta loss'] += losses[3].item() / 100
                total_loss['loss'] += losses[4].item() / 100
            torch.save([epoch, net.module.net.state_dict()], 'net.pth')
        log.info(f"Train Epoch: {epoch} "
                 f"p1 {total_loss['mu loss']:.1e} "
                 f"fine-tuned {total_loss['fine-tuned loss']:.1e} "
                 f"sigma_variance_loss {total_loss['sigma_variance loss']:.1e} "
                 f"beta {total_loss['beta loss']:.1e} "
                 f"total {total_loss['loss']:.1e}")

        total_loss = {
            'mu loss': 0,
            'fine-tuned loss': 0,
            'sigma_variance loss': 0,
            'beta loss': 0,
            'loss': 0}
        net.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for s in test_dataloader:
                for k in ['pics', 'depth']:
                    s[k] = s[k].view(-1, *s[k].shape[2:]).to(cfg.device)
                losses = net(s['depth'], s['pics'])
                losses = [x.mean() for x in losses]
                losses.append(sum(losses))
                total_loss['mu loss'] += losses[0].item() / 10
                total_loss['fine-tuned loss'] += losses[1].item() / 10
                total_loss['sigma_variance loss'] += losses[2].item() / 10
                total_loss['beta loss'] += losses[3].item() / 10
                total_loss['loss'] += losses[4].item() / 10
        log.info(f"Test Epoch: {epoch} "
                 f"mu {total_loss['mu loss']:.1e} "
                 f"fine-tuned {total_loss['fine-tuned loss']:.1e} "
                 f"sigma_variance_loss {total_loss['sigma_variance loss']:.1e} "
                 f"beta {total_loss['beta loss']:.1e} "
                 f"total {total_loss['loss']:.1e}")


if __name__ == "__main__":
    my_app()
