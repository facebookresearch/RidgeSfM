# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import glob
import torch
import json
import numpy as np
import PIL
import collections
from tqdm import tqdm
from tabulate import tabulate


def eval_depth(pred, gt, crop=5, scale=1.0):
    # gt   ... Nx1x480x640 ... ground truth depth tensor
    # pred ... Nx1x480x640 ... predicted depth tensor

    gt = gt[:, :, crop:-crop, crop:-crop]
    pred = pred[:, :, crop:-crop, crop:-crop] * scale

    df_depth = gt - pred

    dmask = (gt > 0.).float()
    dmask_mass = torch.clamp(dmask.sum((1, 2, 3)), 1e-4)

    mse_depth = (dmask * (df_depth**2)).sum((1, 2, 3)) / dmask_mass
    abs_depth = (dmask * df_depth.abs()).sum((1, 2, 3)) / dmask_mass

    res = collections.OrderedDict([
        ('mse_depth', float(mse_depth.cpu().mean())),
        ('rmse_depth', float(mse_depth.cpu().mean().pow(0.5))),
        ('abs_depth', float(abs_depth.cpu().mean())),
    ])

    return res


def interpolate_cameras(C, R):
    if torch.isfinite(C).all():
        return C.clone(), R.clone()

    from pytorch3d.transforms import rotation_conversions
    from scipy.interpolate import interp1d

    ok = torch.isfinite(C.mean(1))
    quats = rotation_conversions.matrix_to_quaternion(R)

    n_frames = C.shape[0]
    y = torch.cat((quats, C), dim=1).numpy()
    x = torch.arange(n_frames).float().numpy()
    ok = np.isfinite(y.mean(1))

    fi = interp1d(
        x[ok], y[ok], kind='linear',
        bounds_error=False, axis=0,
        fill_value=(y[ok][0], y[ok][-1])
    )

    y_interp = fi(x)

    i_quats = torch.tensor(y_interp[:, :4]).float()
    i_R = rotation_conversions.quaternion_to_matrix(i_quats)
    i_C = torch.tensor(y_interp[:, 4:]).float()

    return i_C, i_R


def eval_pose(pred, gt):
    # ScanNet convention for x_cam ... x_world
    # cam->world: right multiply by extrinsics[:3,:3].T and then add extrinsics[:3,3]
    # i.e.: x_world = x_cam @ pose[:3, :3].t() + pose[:3, 3]
    # i.e.: x_cam = (x_world - pose[:3, 3]) @ pose[:3, :3]

    from pytorch3d import ops as pt3ops
    from pytorch3d.transforms import so3

    ok_gt = torch.isfinite(gt.mean((1, 2)))  # some GT poses are NaN
    if not ok_gt.any():
        return 'NO_GT'
    orig_C_pred = pred[ok_gt, :3, 3].clone()
    orig_R_pred = pred[ok_gt, :3, :3].clone()
    orig_C_gt = gt[ok_gt, :3, 3].clone()
    orig_R_gt = gt[ok_gt, :3, :3].clone()
    n_frames = orig_C_pred.shape[0]

    result = {}

    for interpolate in (True, False):
        registered = torch.isfinite(orig_C_pred.mean(1))
        if not registered.any():
            return None
        if interpolate:  # interpolate NaN cameras
            C_pred, R_pred = interpolate_cameras(orig_C_pred, orig_R_pred)
            R_gt = orig_R_gt.clone()
            C_gt = orig_C_gt.clone()
        else:  # remove NaN cameras
            C_pred = orig_C_pred.clone()[registered > 0]
            R_pred = orig_R_pred.clone()[registered > 0]
            C_gt = orig_C_gt.clone()[registered > 0]
            R_gt = orig_R_gt.clone()[registered > 0]

        for align_cams in (True, False):
            for estimate_scale in ((True, False) if align_cams else (False,)):
                if align_cams:
                    # estimate the rigid alignment
                    align_result = pt3ops.corresponding_points_alignment(
                        C_pred[None], C_gt[None], estimate_scale=estimate_scale)
                    # align centers and rotations
                    C_pred_align = (
                        align_result.s *
                        C_pred @ align_result.R[0] +
                        align_result.T[0])
                    R_pred_align = torch.bmm(
                        align_result.R.permute(
                            0, 2, 1).expand_as(R_pred), R_pred)
                else:
                    C_pred_align = C_pred.clone()
                    R_pred_align = R_pred.clone()

                # compute the rotation errors and camera center errors
                cam_center_error = (C_pred_align - C_gt).norm(dim=1).mean()
                cam_angle_error = so3.so3_relative_angle(
                    R_pred_align, R_gt).median() * 180 / np.pi

                # store the errors
                postfix = ''
                if not align_cams:
                    postfix += '_noalign'
                if interpolate:
                    postfix += '_interp'
                if estimate_scale:
                    postfix += '_scale'
                result['cam_center_err' + postfix] = float(cam_center_error)
                result['cam_angle_err' + postfix] = float(cam_angle_error)

                if estimate_scale and not interpolate and align_cams:
                    result['best_scale'] = float(align_result.s)

    return result


def load_frames(pth_data, depth_is_raylen=False):

    rgbd = []

    n_frames = len(pth_data['color'])
    for c, d, p in zip(pth_data['color'], pth_data['depth'], pth_data['pose']):
        de = torch.from_numpy(
            np.array(
                PIL.Image.open(d),
                copy=True)).float() / pth_data['depth scale factor']
        #de = de * (de<3).float() ##########
        rgb = torch.from_numpy(
            np.array(
                PIL.Image.open(c),
                copy=True)).float() / 255
        rgbd.append(torch.cat((rgb.permute(2, 0, 1), de[None]), dim=0))

    rgbd = torch.stack(rgbd, dim=0)

    intrinsic = pth_data['intrinsic'][None].repeat(n_frames, 1, 1)
    assert intrinsic.shape[0] == n_frames
    assert intrinsic.shape[1] == 4
    assert intrinsic.shape[2] == 4

    return {
        'color': rgbd[:, :3],
        'depth': rgbd[:, 3:],
        'intrinsic': intrinsic,
        'pose': pth_data['pose']
    }


def align_pose(align_result, pose):
    C = pose.clone()[:, :3, 3]
    R = pose.clone()[:, :3, :3]

    # align centers and rotations
    C_align = align_result.s * (C @ align_result.R[0]) + align_result.T
    R_align = torch.bmm(align_result.R.permute(0, 2, 1).expand_as(R), R)

    pose_align = torch.cat((R_align, C_align[:, :, None]), dim=2)
    pose_align = torch.cat((pose_align, pose_align[:, 2:, :] * 0.), dim=1)
    pose_align[:, 3, 3] = 1.

    return pose_align




def eval_point_cloud(
    pr_depth, pr_pose,
    gt_depth, gt_pose,
    intrinsic,
    crop=5, stride=10, scale=1
):
    '''
    Use pose, depth and intrinsics to construct gt and predicted point clouds.
    Align/scale the predicted point cloud and depth predictions using ytorch3d.ops.corresponding_points_alignment.
    '''

    from pytorch3d import ops as pt3ops
    from pytorch3d.transforms import so3

    ok_gt = torch.isfinite(gt_pose.mean((1, 2)))  # some GT poses are NaN
    if not ok_gt.any():
        return 'NO_GT'

    results = {}

    gt_depth = gt_depth[ok_gt]
    gt_pose = gt_pose[ok_gt]
    pr_depth = pr_depth[ok_gt]
    pr_pose = pr_pose[ok_gt]

    n = gt_depth.size(0)

    if len(intrinsic.shape) == 3:
        intrinsic = intrinsic.numpy()
        x = (np.arange(crop, gt_depth.size(2) - crop, stride, dtype=np.float32)
             [None] - intrinsic[:, 1, 2][:, None]) / intrinsic[:, 1, 1][:, None]
        y = (np.arange(crop, gt_depth.size(3) - crop, stride, dtype=np.float32)
             [None] - intrinsic[:, 0, 2][:, None]) / intrinsic[:, 0, 0][:, None]
        xyz = torch.from_numpy(np.ascontiguousarray(
            np.concatenate([
                np.tile(y[:, None, :], (1, len(x[0]), 1))[:, None],
                np.tile(x[:, :, None], (1, 1, len(y[0])))[:, None],
                np.ones(
                    (n, 1, len(x[0]), len(y[0]))
                ),
            ], axis=1)
        ))
        xyz = xyz.float().permute(0, 2, 3, 1).reshape(n, -1, 3)

        xyz = xyz.cuda()
        gt_depth = gt_depth.cuda()
        gt_pose = gt_pose.cuda()
        pr_depth = pr_depth.cuda()
        pr_pose = pr_pose.cuda()

    else:
        x = (np.arange(crop, gt_depth.size(2) - crop, stride, dtype=np.float32) -
             intrinsic[:, 1, 2].item()) / intrinsic[:, 1, 1]
        y = (np.arange(crop, gt_depth.size(3) - crop, stride, dtype=np.float32) -
             intrinsic[:, 0, 2].item()) / intrinsic[:, 0, 0]

        xyz = torch.from_numpy(np.ascontiguousarray(np.vstack([
            np.tile(y[None, None, :], (1, len(x), 1)),
            np.tile(x[None, :, None], (1, 1, len(y))),
            np.ones((1, len(x), len(y)))]))).float().view(3, -1).t()[None]

    mask = (gt_depth[:, :, crop:-crop:stride, crop:-crop:stride].flatten() > 0)
    gt_xyz = gt_depth[:, :, crop:-crop:stride,
                      crop:-crop:stride].reshape(n, -1, 1) * xyz
    pr_xyz = pr_depth[:, :, crop:-crop:stride,
                      crop:-crop:stride].reshape(n, -1, 1) * xyz

    pr_xyz = pr_xyz @ pr_pose[:, :3,
                              :3].transpose(-2, -1) + pr_pose[:, None, :3, 3]
    gt_xyz = gt_xyz @ gt_pose[:, :3,
                              :3].transpose(-2, -1) + gt_pose[:, None, :3, 3]
    pr_xyz = pr_xyz.view(-1, 3)[mask]
    gt_xyz = gt_xyz.view(-1, 3)[mask]

    for estimate_scale in [True]:
        align_result = pt3ops.corresponding_points_alignment(
            pr_xyz[None], gt_xyz[None], estimate_scale=estimate_scale)
        q = pr_xyz @ align_result.R[0] * align_result.s[0] + align_result.T[0]
        results['point_cloud_mse_loss' +
                ('_scaled' if estimate_scale else '')] = (gt_xyz -
                                                          q).pow(2).sum(1).mean(0).item()
        results['point_cloud_rmse_loss' +
                ('_scaled' if estimate_scale else '')] = (gt_xyz -
                                                          q).pow(2).sum(1).mean(0).pow(0.5).item()
        results['point_cloud_abs_loss' +
                ('_scaled' if estimate_scale else '')] = (gt_xyz -
                                                          q).norm(dim=1).mean(0).item()

        for k, v in eval_depth(pr_depth * align_result.s[0], gt_depth).items():
            results[k + ('_scaled' if estimate_scale else '')] = v

        pr_pose_align = align_pose(align_result, pr_pose)
        pose_results = eval_pose(pr_pose_align, gt_pose)

        results['cam_center_err_interp' + ('_scaled' if estimate_scale else '')] =\
            pose_results['cam_center_err_noalign_interp']
        results['cam_angle_err_interp' + ('_scaled' if estimate_scale else '')] =\
            pose_results['cam_angle_err_noalign_interp']

    return results


def eval_pth(split_pth, results, dumpfile=None):
    # split_pth: .pth file with the eval files
    # results: dict{
    #   'depth': n_frames x 1 x he x wi,
    #   'pose': n_frames x 4 x 4,
    #   'K': n_frames x 3 x 3, # intrinsics for each frame
    # }
    # dumpfile: path to a file with the result dump

    if results is None:
        result = None

    else:
        # load the split data
        split_data = torch.load(split_pth)
        split_data = load_frames(split_data, depth_is_raylen=False)

        # eval pose
        pose_gt = split_data['pose']
        pose_pred = results['pose']
        result = eval_pose(pose_pred.clone(), pose_gt.clone())

        # eval depth
        if (result is not None) and (result != 'NO_GT'):
            ok = torch.isfinite(pose_gt.sum((1, 2)))  # eval only defined depth
            ok *= torch.isfinite(pose_pred.sum((1, 2)))
            ok *= torch.isfinite(results['depth'].sum((1, 2, 3)))  # same here
            depth_gt = split_data['depth'][ok]
            depth_pred = results['depth'][ok]
            depth_result = eval_depth(
                depth_pred.clone(),
                depth_gt.clone(),
                scale=result['best_scale'])
            result.update(depth_result)

            result_pcl = eval_point_cloud(
                depth_pred.clone(), pose_pred[ok].clone(),
                depth_gt.clone(), pose_gt[ok].clone(),
                results['K'][ok].clone(), crop=5, stride=10, scale=1
            )
            for k, v in result_pcl.items():
                result['pcl_' + k] = v

    if dumpfile is not None:
        # print(f'dumping to {dumpfile}')
        with open(dumpfile, 'w') as f:
            json.dump(result, f)

    return result


def get_pths(
    splitdir='splits/',
    split_regexp_patterns=[],
):
    pths = glob.glob(os.path.join(splitdir, '*', '*.pth'))
    if len(split_regexp_patterns) > 0:
        # filter the splits based on the regexp patterns
        pths = [
            p for p in pths if any(
                pat in p for pat in split_regexp_patterns)]

    return pths


def get_dumpfile(dumpdir, pth):
    pth_name = os.path.split(pth)[-1].split('.')[0]
    dumpfile = os.path.join(dumpdir, pth_name + '.json')
    return dumpfile


def average_json_file_list(jsons):
    # load all jsons into an array
    seqresults = []
    for dumpfile in jsons:
        with open(dumpfile, 'r') as f:
            seqresult = json.load(f)
        if seqresult != 'NO_GT':
            seqresults.append(seqresult)

    # parse the result keys
    reskeys = set()
    for r in seqresults:
        if r is not None:
            for k in r:
                reskeys.add(k)

    # aggregate the results
    results = {k: [] for k in reskeys}
    for r in seqresults:
        if r is not None:
            for k, v in r.items():
                if np.isfinite(v):
                    results[k].append(float(v))
                else:
                    print('non-finite number!!')
                    print(f'{k}:{v}')

    # get the averages
    results_out = {}
    for k, v in results.items():
        results_out[k] = float(np.array(results[k]).mean())
        results_out[k + '_med'] = float(np.median(np.array(results[k])))

    results_out['n_sucessful_recons'] = len(
        [r for r in seqresults if r is not None])
    return results_out


def average_results(
    splitdir='slits/',
    dumpdir=None,
    split_regexp_patterns=[],
    ok_metrics=[],
):

    pths = get_pths(
        splitdir=splitdir, split_regexp_patterns=split_regexp_patterns,
    )

    # split to frameskips
    pat = 'frameskip='
    def get_frame_skip(pth): return int(
        pth[pth.find(pat) + len(pat):].split('-')[0])
    pth_frameskip = [get_frame_skip(pth) for pth in pths]
    unq_frameskip = sorted(list(set(pth_frameskip)))
    if len(split_regexp_patterns) == 0:
        assert len(unq_frameskip) == 4

    results = {}
    for frameskip in unq_frameskip:
        okpths = [
            pth for pth,
            fs in zip(
                pths,
                pth_frameskip) if frameskip == fs]
        jsons = [get_dumpfile(dumpdir, pth) for pth in okpths]
        if len(split_regexp_patterns) == 0:
            assert len(jsons) == 100

        results_frameskip = average_json_file_list(jsons)
        results[frameskip] = results_frameskip

    if False:
        # parse metrics automatically
        metrics = sorted(list(results[frameskip].keys()))
    else:
        # predefine which metrics to print
        metrics = [
            'pcl_cam_angle_err_interp_scaled',
            'pcl_cam_center_err_interp_scaled',
            'pcl_abs_depth_scaled',
            'pcl_rmse_depth_scaled',
            'pcl_point_cloud_abs_loss_scaled',
            'pcl_point_cloud_rmse_loss_scaled',
            'n_sucessful_recons',
        ]

    if len(ok_metrics) > 0:
        metrics = [m for m in metrics if m in ok_metrics]

    print('')
    print('SCANNET RESULTS:')
    tab_rows = []
    if False:
        # frameskip results per row
        for frameskip, r in results.items():
            row = [r[m] for m in metrics]
            tab_rows.append([frameskip, *row])
        print(
            tabulate(
                tab_rows, headers=['frameskip', *metrics],
            )
        )
    else:
        # frameskip results in columns
        for metric in metrics:
            row = [results[frameskip][metric] for frameskip in unq_frameskip]
            tab_rows.append([metric, *row])
        print(
            tabulate(
                tab_rows, headers=['frameskip', *unq_frameskip],
            )
        )


def eval_scannet(
    slam_fun_handle=None,
    split_regexp_patterns=[],
    splitdir='splits/',
    dumpdir=None,
    refresh=False,
    ok_metrics=[],
):
    """
    Args:
        slam_fun_handle: a function that takes a single .pth file as input
                         and returns a dict:
            {'depth': torch.FloatTensor(n_frames, 1, 480, 640),
             'pose':  torch.FloatTensor(n_frames, 4, 4), }

    Returns:
        results: Results averaged over the set of splits
    """

    all_results = []

    pths = get_pths(
        splitdir=splitdir, split_regexp_patterns=split_regexp_patterns,
    )

    for pth in tqdm(pths):

        if dumpdir is not None:
            os.makedirs(dumpdir, exist_ok=True)
            dumpfile = get_dumpfile(dumpdir, pth)
        else:
            dumpfile = None

        if dumpfile is not None and not refresh and os.path.isfile(dumpfile):
            print(f'skipping {pth}')
            with open(dumpfile, 'r') as f:
                pth_results = json.load(f)
        else:
            pth_reconstruction = slam_fun_handle(pth)
            pth_results = eval_pth(pth, pth_reconstruction, dumpfile=dumpfile)

        all_results.append(pth_results)

    average_results(
        splitdir=splitdir, dumpdir=dumpdir,
        ok_metrics=ok_metrics, split_regexp_patterns=split_regexp_patterns
    )
