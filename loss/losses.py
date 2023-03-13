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
from model.util import *
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


def geometricDistance_offset(i, inp, out, scale_x=1.0, scale_y=1.0):
    pts_1 = inp['points_1']
    ones = torch.ones_like(pts_1)
    pts_1 = torch.cat([pts_1, ones[:, :, :1]], -1)
    pts_2 = inp['points_2']
    homo_21_inv = torch.inverse(out['H_flow'][i][0])
    homo_12_inv = torch.inverse(out['H_flow'][i][1])
    # ipdb.set_trace()
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


def camera_loss_sequnce():
    cam_align = {
        'front': ['left', 'right'], # 0, 1 img1=front
        'back': ['left', 'right'],  # 2, 3 img1=back
        'left': ['front', 'back'],  # 4, 5 img1=left
        'right': ['front', 'back'], # 6, 7 img1=right
    }
    sequence = {
        'basic': [0,1, 2,3, 4,5, 6,7],
        'all_fblr': [0,1, 2,3, 4,5, 6,7],
        'front_first': [0,1,5,7],
        'back_first': [2,3,4,6],
        'left_first': [4,5,1,3],
        'right_first': [6,7,0,2],
    }
    return sequence


def compute_losses(output, input, params):
    losses = {}
    imgs_patch = input['imgs_gray_patch']
    total_loss = 0
    for i, camera in enumerate(params.camera_list):
        img1_warp, img2_warp = output["img_warp"][i]
        img1 = imgs_patch[:, i*2:i*2+1]
        img2 = imgs_patch[:, i*2+1:i*2+2]
        im_diff_fw = img1 - img2_warp
        im_diff_bw = img2 - img1_warp
        # im_diff_fw = img1 - img2
        # im_diff_bw = img2 - img1
        # im_diff = img2 - img1_warp
        photo_loss_f = photo_loss_function(diff=im_diff_fw, q=1, averge=True)
        photo_loss_b = photo_loss_function(diff=im_diff_bw, q=1, averge=True)
        # photo_loss = photo_loss_function(diff=im_diff, q=1, averge=True)
        total_loss += photo_loss_f + photo_loss_b
        # total_loss += photo_loss_f + photo_loss_b + photo_loss * 1e-8
        
    # gt_photo_error: b16 = 0.77   b07 = 0.25
    # train: gt 0.5862 vs pd 0.4665
    # test:  gt 0.6544 vs pd 0.5836
    losses['total'] = total_loss / len(params.camera_list)
    return losses


def compute_losses_v0(output, input, params):
    '''
        1. (f0,l0), (b1,r1), (l1,b0), (r0,f1) photo_loss 正反全部计算
        2. 以前摄像头为准: l0→f0, r0→f1, b0→l1, b1→r1
        3. 以后摄像头为准: l1→b0, r1→b1, f0→l0, f1→r0 
        4. 以左摄像头为准: f0→l0, b0→l1, r0→f1, r1→b1 
        5. 以右摄像头为准: f1→r0, b1→r1, l0→f0, l1→b0 
        6. 顺序的强调用权重系数表示
        7. l0->f0 corresponding img1=f0<-img2=l0: index of f0 = 0
        8. l0->f0(0), r0->f1(1), b0->l1(5), b1->r1(7)
    '''
    sequence = camera_loss_sequnce()
    assert params.pair_loss_type in sequence.keys()
    idx_seq = sequence[params.pair_loss_type]
    
    losses = {}
    imgs_patch = input['imgs_gray_patch']
    photo_losses = []
    for i, camera in enumerate(params.camera_list):
        img1_warp, img2_warp = output["img_warp"][i]
        im_diff_fw = imgs_patch[:, i*2:i*2+1] - img2_warp
        im_diff_bw = imgs_patch[:, i*2+1:i*2+2] - img1_warp
        photo_loss_f = photo_loss_function(diff=im_diff_fw, q=1, averge=True)
        photo_loss_b = photo_loss_function(diff=im_diff_bw, q=1, averge=True)
        photo_losses.append(photo_loss_f)
        photo_losses.append(photo_loss_b)
    
    
    if len(idx_seq) == 8:
        total_loss = 0
        for i, cam in enumerate(params.camera_list):
            losses[cam] = photo_losses[i*2] + photo_losses[i*2+1]
            total_loss += losses[cam]
        losses['total'] = total_loss / len(params.camera_list)
    else:
        total_loss = []
        for i in idx_seq:
            total_loss.append(photo_losses[i])
        
        # loss系数设置
        if len(idx_seq) == 4:
            coef = [0.35, 0.35, 0.15, 0.15]
        else:
            coef = [1/8 for _ in range(8)]
        
        total_loss = [a*x.unsqueeze(0) for a,x in zip(coef, total_loss)]
        # ipdb.set_trace()
        losses['total'] = 2 * torch.cat(total_loss).sum()

    return losses


def compute_eval_results(data_batch, output_batch, params):
    imgs_full = data_batch["imgs_ori"]
    batch_size, _, grid_h, grid_w = data_batch["imgs_ori"].shape
    bhw = (batch_size, grid_h, grid_w)
    errs = np.zeros(batch_size)
    img1_full_warp_list = []
    
    _h, _w = data_batch['imgs_gray_full'].shape[2:]
    scale_x = grid_w / _w
    scale_y = grid_h / _h
    
    for i, camera in enumerate(params.camera_list):
        img_1_full = imgs_full[:, i*6:i*6+3]
        img_2_full = imgs_full[:, i*6+3:i*6+6]

        if params.forward_version == 'offset':
            homo_21, homo_12 = output_batch['H_flow'][i]
            img1_full_warp = warp_image_from_H(homo_21, img_1_full, *bhw)
            # img2_full_warp = warp_image_from_H(homo_12, img_2_full, *bhw)
            err = geometricDistance_offset(i, data_batch, output_batch, scale_x, scale_y)
        elif params.forward_version == 'basis':
            H_flow_f, H_flow_b = output_batch['H_flow'][i]
            # ipdb.set_trace()
            # H_flow_f = upsample2d_flow_as(H_flow_f, imgs_full, mode="bilinear", if_rate=True)
            H_flow_b = upsample2d_flow_as(H_flow_b, imgs_full, mode="bilinear", if_rate=True)
            img1_full_warp = get_warp_flow(img_1_full, H_flow_b, 0)
            # img2_full_warp = get_warp_flow(img_2_full, H_flow_f, 0)
            err = np.float32(0)
        else:
            raise
        
        # ipdb.set_trace()
        img1_full_warp_list.append(img1_full_warp)
        errs += err

    eval_results = {}
    eval_results["img1_full_warp"] = torch.cat(img1_full_warp_list, 1)
    eval_results["errs"] = errs / len(params.camera_list)
    
    return eval_results
