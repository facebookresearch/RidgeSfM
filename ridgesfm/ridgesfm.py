# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import utils
import torch
import eval_scannet
import multiprocessing.pool
from tqdm import tqdm
import logging
import hydra
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['TORCH_USE_RTLD_GLOBAL'] = 'YES'
log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="ransac")
def my_app(cfg):
    thread_pool = multiprocessing.pool.ThreadPool()
    log.info(f'{cfg}')
    log.info(f'Working directory : {os.getcwd()}')

    if cfg.unet.ssp == 'tailored':
        size_stride_pad = [
            [[4, 4], [4, 4], [3, 4], [3, 4]],
            [2, 2, 2, 2],
            [[1, 1], [1, 1], [0, 1], [0, 1]]]
    else:
        size_stride_pad = [[4] * 6, [2] * 6, [1] * 6]

    n_features = [
        1 * cfg.unet.m,
        2 * cfg.unet.m,
        3 * cfg.unet.m,
        4 * cfg.unet.m,
        5 * cfg.unet.m]
    net = torch.nn.Sequential(
        utils.conv_4x4_bn(3, cfg.unet.m // 2, 2),
        utils.conv_4x4_bn(cfg.unet.m // 2, cfg.unet.m, 2),
        utils.UNet(2, n_features, *size_stride_pad,
            cfg.unet.depth,
            join=cfg.unet.join, hs=cfg.unet.hs,
            t=cfg.unet.t),
        torch.nn.Conv2d(cfg.unet.m, 1 + cfg.n_multipred, 1, 1, 0),
        torch.nn.Upsample(scale_factor=8))
    net.load_state_dict(torch.load(cfg.unet.name, map_location=cfg.device)[1])
    net.eval()
    net.to(cfg.device)
    s = utils.load_scene(cfg, net)
    utils.prematch_scene(cfg, s, thread_pool)

    s['ransac matches'] = {}
    def f(k):
        torch.set_num_threads(1)
        if k[0] + 1 == k[1]:
            b = utils.PairwiseRidgeSfM(
                s,
                k,
                reps=64,
                precision=cfg.precision,
                min_matches=16,
                iters=3,
                gr=0.2,
                weight_decay=cfg.lmbda).backup
        else:
            b = utils.PairwiseRidgeSfM(
                s,
                k,
                reps=32,
                precision=cfg.precision,
                min_matches=16,
                iters=3,
                gr=0.2,
                weight_decay=cfg.lmbda).backup
        return b
    betas = list(tqdm(thread_pool.imap_unordered(f, s['matches'].keys()),desc='Pairwise-RidgeSfM',total=len(s['matches'])))
    for b in betas:
        if len(b) > 3:
            s['ransac matches'][b['k']] = b
    scene = utils.GlobalBundleAdjustment(s).to(cfg.device)
    scene.make_optim()
    for i in range(scene.bundle_k.size(0) * 6):
        active_k = min((i // 5 + 1), scene.bundle_k.size(0))
        n_cams = scene.bundle_k[active_k - 1][1]
        scene.bundle_init(active_k, i % 1000 == 0)

    for i in range(12):
        scene.flub(i)
    scene.vid_plot(
        cfg.outdir +
        f'/scene{cfg.scene.n}_frameskip{cfg.scene.frameskip}',
        frame_rate={1: 24, 3: 8, 10: 3, 30: 1}[cfg.scene.frameskip],
        cam_skip={1: 10, 3: 3, 10: 1, 30: 1}[cfg.scene.frameskip],
        device=cfg.device)
    if 'depth' in s:
        out_file = eval_scannet.get_dumpfile(cfg.outdir, s['file_name'])
        scene.pose(out_file)


if __name__ == "__main__":
    my_app()
