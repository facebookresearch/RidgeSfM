# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import numpy as np
import torch


def make_test_pth(seq_dir, frame_sel_idx):

    rgb_paths, d_paths, poses = [
        [
            os.path.join(seq_dir, folder, str(f)  + '.' + postfix)
            for f in frame_sel_idx
        ]
        for folder, postfix in zip(
            ('color', 'depth', 'pose'), ('jpg','png','txt')
        )
    ]

    poses = torch.stack([torch.tensor(np.loadtxt(p)).float() for p in poses])

    data = {
        'depth scale factor': 1000,
        'color': rgb_paths,
        'depth': d_paths,
        'pose': poses,
        'intrinsic': torch.tensor(
            np.loadtxt(os.path.join(
                seq_dir, 'intrinsic', 'intrinsic_depth.txt'))).float(),
        'H': 480,
        'W': 640,
    }

    return data


def process_scannet_seq(outdir, scannet_dir, seq, max_frames=300):
    rgb_dir = os.path.join(scannet_dir, seq, 'color')
    frames = sorted(os.listdir(rgb_dir))
    n_tot = len(frames)

    for frame_skip in (1, 3, 10, 30):
    
        vlen = min(frame_skip * max_frames, n_tot)
        n_frames = vlen // frame_skip
        vlen = n_frames * frame_skip
        max_start = max(n_tot-vlen-1, 0)
        
        if True:
            if max_start > 0:
                starts = [np.random.randint(max_start)]
            else:
                starts = [0]
        else:
            starts = np.round(np.linspace(0, max_start, 5)).astype(int)
            starts = np.unique(starts)
            if len(starts) > 1 and starts[1]-starts[0] < 20:
                # take only one video if all possible starts are too close
                starts = starts[:1]

        for start_at in starts:
            frame_sel_idx = start_at + np.arange(0, n_frames) * frame_skip
            assert len(frame_sel_idx)==n_frames
            test_data = make_test_pth(
                os.path.join(scannet_dir, seq), frame_sel_idx)
            
            data_name = f'seq={seq}-maxframes={max_frames}-frameskip={frame_skip}-start={start_at}.pth'
            data_dir = os.path.join(outdir, seq)
            os.makedirs(data_dir, exist_ok=True)
            out_path = os.path.join(data_dir, data_name)
            print(out_path)
            torch.save(test_data, out_path)


def main():
    scannet_dir = os.getcwd()+'/scannet/train/'
    outdir = 'scannet/validation_scenes/'
    os.makedirs(outdir, exist_ok=True)
    seqs = sorted(os.listdir(scannet_dir))[1413:]
    np.random.seed(0)
    for seq in seqs:
        process_scannet_seq(outdir, scannet_dir, seq)

    scannet_dir = os.getcwd()+'/scannet/test/'
    outdir = 'scannet/test_scenes/'
    os.makedirs(outdir, exist_ok=True)
    seqs = sorted(os.listdir(scannet_dir))
    np.random.seed(0)
    for seq in seqs:
        process_scannet_seq(outdir, scannet_dir, seq)


if __name__ == "__main__":
    main()
