import os
import sys
import shutil
import torch
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


def compute_losses(output, input, params):
    losses = {'total': 0}

    # compute losses
    if params.loss_type == "basic":
        imgs_patch = input['imgs_gray_patch']
        start = input['start']
        
        fea1_full, fea2_full = output["fea_full"]
        fea1_patch, fea2_patch = output["fea_patch"]
        img1_warp, img2_warp = output["img_warp"]
        fea1_patch_warp, fea2_patch_warp = output["fea_patch_warp"]
        
        if params.model_version == 'basis':
            H_flow_f, H_flow_b = output['H_flow']
            fea2_warp = get_warp_flow(fea2_full, H_flow_f, start=start)
            fea1_warp = get_warp_flow(fea1_full, H_flow_b, start=start)
        elif params.model_version == 'offset':
            batch_size, _, h_patch, w_patch = imgs_patch.shape
            bhw = (batch_size, h_patch, w_patch)
            homo_21, homo_12 = output['H_flow']
            fea1_warp = warp_image_from_H(homo_21, fea1_full, *bhw)
            fea2_warp = warp_image_from_H(homo_12, fea2_full, *bhw)
        else:
            raise
        
        im_diff_fw = imgs_patch[:, :1, ...] - img2_warp
        im_diff_bw = imgs_patch[:, 1:, ...] - img1_warp

        fea_diff_fw = fea1_warp - fea1_patch_warp
        fea_diff_bw = fea2_warp - fea2_patch_warp
        

        # loss
        losses["photo_loss_l1"] = photo_loss_function(
            diff=im_diff_fw, q=1, averge=True
        ) + photo_loss_function(diff=im_diff_bw, q=1, averge=True)

        losses["fea_loss_l1"] = photo_loss_function(
            diff=fea_diff_fw, q=1, averge=True
        ) + photo_loss_function(diff=fea_diff_bw, q=1, averge=True)

        losses["triplet_loss"] = (
            triplet_loss(fea1_patch, fea2_warp, fea2_patch).mean()
            + triplet_loss(fea2_patch, fea1_warp, fea1_patch).mean()
        )

        photo = losses["photo_loss_l1"]
        feature = params.weight_fil * losses["fea_loss_l1"]
        triplet = losses["triplet_loss"]
        
        if params.loss_func_type == 'feature':
            losses['total'] = feature
        elif params.loss_func_type == 'triplet':
            losses['total'] = triplet
        elif params.loss_func_type == 'photo':
            losses['total'] = photo
        elif params.loss_func_type == 'all':
            losses["total"] = feature + triplet + photo
        elif params.loss_func_type == 'origin':
            losses["total"] = feature + triplet
    else:
        raise NotImplementedError

    return losses


def compute_eval_results(data_batch, output_batch, params):
    start = data_batch['start']
    imgs_full = data_batch["imgs_ori"]
    H_flow_f, H_flow_b = output_batch['H_flow']
    batch_size, _, grid_h, grid_w = imgs_full.shape
    bhw = (batch_size, grid_h, grid_w)
    
    errs, errs_p = [], []

    if params.model_version == 'basis':
        H_flow_f = upsample2d_flow_as(H_flow_f, imgs_full, mode="bilinear", if_rate=True)
        H_flow_b = upsample2d_flow_as(H_flow_b, imgs_full, mode="bilinear", if_rate=True)
        img1_full_warp = get_warp_flow(imgs_full[:, :3, ...], H_flow_b, start=start)
        # calc err
        points = data_batch["points"]
        for i in range(len(points)):  # len(points)
            point = eval(points[i])
            err = 0
            tmp = []
            for j in range(6):  # len(point['matche_pts'])
                points_value = point['matche_pts'][j]
                err_p = geometricDistance(points_value, H_flow_f[i])
                err += err_p
                tmp.append(err_p)
            errs.append(err / (j + 1))
            errs_p.append(tmp)

    elif params.model_version == 'offset':
        img1_full_warp = warp_image_from_H(H_flow_b, imgs_full[:, :3, ...], *bhw)
        # calc err
        scale_x = grid_w / float(data_batch['imgs_gray_full'].shape[3])
        scale_y = grid_h / float(data_batch['imgs_gray_full'].shape[2])
        errs = geometricDistance_v2(data_batch, output_batch, scale_x, scale_y)
    else:
        raise
    

    eval_results = {}
    eval_results["img1_full_warp"] = img1_full_warp
    eval_results["errs"] = errs
    eval_results["errs_p"] = errs_p

    return eval_results
