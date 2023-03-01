import os
import sys
import cv2
import ipdb
import torch
import shutil
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import net
from model.util import warp_image_from_H
from torchvision.utils import save_image


def triplet_loss(a, p, n, margin=1.0, exp=1, reduce=False, size_average=False):

    triplet_loss = nn.TripletMarginLoss(
        margin=margin, p=exp, reduce=reduce, size_average=size_average
    )
    return triplet_loss(a, p, n)


def photo_loss_function(diff, q, averge=True):
    diff = (torch.abs(diff) + 0.01).pow(q)
    if averge:
        loss_mean = diff.mean()
    else:
        loss_mean = diff.sum()
    return loss_mean


def geometricDistance(correspondence, flow):
    flow = flow.permute(1, 2, 0).cpu().detach().numpy()

    p1 = correspondence[0]  # 0
    p2 = correspondence[1]  # 1

    if isinstance(correspondence[1][0], float):
        result = p2 - (p1 - flow[int(p1[1]), int(p1[0])])
        error = np.linalg.norm(result)
    else:
        result = [p2 - (p1 - flow[p1[1], p1[0]]), p1 - (p2 - flow[p2[1], p2[0]])]
        error = min(np.linalg.norm(result[0]), np.linalg.norm(result[1]))

    return error


def geometricDistance_v2(inp, out, scale_x=1.0, scale_y=1.0):
    ones = torch.ones_like(inp['points_1_all'])
    pts_1 = torch.cat([inp['points_1_all'], ones[:, :, :1]], -1)
    pts_2 = inp['points_2_all']
    homo_21_inv = torch.inverse(out['H_flow'][0])
    homo_12_inv = torch.inverse(out['H_flow'][1])

    def calc_pts_err(homo, pts1, pts2, scale_x, scale_y):
        pts = pts1.permute(0, 2, 1)
        pts[:, 0] = pts[:, 0] / scale_x
        pts[:, 1] = pts[:, 1] / scale_y
        warp_pts = torch.einsum('bnc,bck->bnk', homo, pts)
        warp_pts = warp_pts.permute(0, 2, 1)
        warp_pts = warp_pts / warp_pts[:, :, 2:]  # /z, normalilzation
        warp_pts[:, :, 0] *= scale_x
        warp_pts[:, :, 1] *= scale_y

        diff = torch.linalg.norm(warp_pts[:, :, :2] - pts2, dim=2)
        return diff.mean(1).data.cpu().numpy()

    err_1 = calc_pts_err(homo_21_inv, pts_1, pts_2, scale_x, scale_y)

    return err_1


def front_first_loss(output, input):
    pass


def compute_losses(output, input, params):
    '''
        1. (f0,l0), (b1,r1), (l1,b0), (r0,f1) photo_loss 正反全部计算
        2. 以前摄像头为准: l0→f0, r0→f1, b0→l1, b1→r1
        3. 以左摄像头为准: f0→l0, b0→l1, r0→f1, r1→b1 
        4. 以右摄像头为准: f1→r0, b1→r1, l0→f0, l1→b0 
        5. 以后摄像头为准: l1→b0, r1→b1, f0→l0, f1→f0 
        6. 顺序的强调用权重系数表示
    '''
    losses = {}
    imgs_patch = input['imgs_gray_patch']
    
    total_loss = []
    if params.loss_type in ['basic', 'fblr_all']:
        for i, camera in enumerate(params.camera_list):
            img1_warp, img2_warp = output["img_warp"][i]
            im_diff_fw = imgs_patch[:, i*2:i*2+1] - img2_warp
            im_diff_bw = imgs_patch[:, i*2+1:i*2+2] - img1_warp
            photo_loss_f = photo_loss_function(diff=im_diff_fw, q=1, averge=True)
            photo_loss_b = photo_loss_function(diff=im_diff_bw, q=1, averge=True)
            losses[camera] = photo_loss_f + photo_loss_b
            total_loss.append(losses[camera].unsqueeze(0))
    elif params.loss_type in ['front_first']:
        pass
    elif params.loss_type in ['left_first']:
        pass
    elif params.loss_type in ['right_first']:
        pass
    elif params.loss_type in ['back_first']:
        pass
    else:
        raise NotImplementedError
    
    losses['total'] = torch.cat(total_loss).mean()

    return losses


def compute_eval_results(data_batch, output_batch, manager):

    imgs_full = data_batch["imgs_ori"]
    batch_size, _, grid_h, grid_w = imgs_full.shape

    bhw = (batch_size, grid_h, grid_w)
    homo_21, homo_12 = output_batch['H_flow']

    # img1 warp to img2, img2_pred = img1_warp
    img1_full_warp = warp_image_from_H(homo_21, imgs_full[:, :3, ...], *bhw)
    img2_full_warp = warp_image_from_H(homo_12, imgs_full[:, 3:, ...], *bhw)

    scale_x = grid_w / float(data_batch['imgs_gray_full'].shape[3])
    scale_y = grid_h / float(data_batch['imgs_gray_full'].shape[2])
    errs = geometricDistance_v2(data_batch, output_batch, scale_x, scale_y)

    eval_results = {}
    eval_results["img1_full_warp"] = img1_full_warp
    eval_results["errs"] = errs

    return eval_results
