import os
import sys
import cv2
import torch
import shutil
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import net


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
    homo_21_inv = torch.inverse(out['H_flow_21'])
    homo_12_inv = torch.inverse(out['H_flow_12'])

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
    # err_2 = calc_pts_err(homo_12_inv, pts_2, pts_1, scale_x, scale_y)

    return err_1


def compute_losses(params, input, output):
    losses = {}

    # compute losses
    if params.loss_type == "basic":

        fea1_patch = output["fea1_patch"]
        fea2_patch = output["fea2_patch"]
        fea1_warp = output["fea1_full_warp_p"]
        fea2_warp = output["fea2_full_warp_p"]

        im_diff_1 = input['img1_patch_gray'] - output["img2_patch_gray_warp_p"]
        im_diff_2 = input['img2_patch_gray'] - output["img1_patch_gray_warp_p"]

        fea_diff_1 = output["fea2_full_warp_p"] - output["fea2_patch_warp"]
        fea_diff_2 = output["fea1_full_warp_p"] - output["fea1_patch_warp"]

        # loss
        losses["photo_loss_l1"] = photo_loss_function(
            diff=im_diff_1, q=1, averge=True
        ) + photo_loss_function(diff=im_diff_2, q=1, averge=True)

        losses["fea_loss_l1"] = photo_loss_function(
            diff=fea_diff_1, q=1, averge=True
        ) + photo_loss_function(diff=fea_diff_2, q=1, averge=True)

        losses["triplet_loss"] = (
            triplet_loss(fea1_patch, fea2_warp, fea2_patch).mean()
            + triplet_loss(fea2_patch, fea1_warp, fea1_patch).mean()
        )

        feature_loss = (
            losses["triplet_loss"] + params.weight_fil * losses["fea_loss_l1"]
        )
        photo_loss = losses["photo_loss_l1"]
        if params.loss_func_type == 'feature':
            losses['total'] = feature_loss
        elif params.loss_func_type == 'photo':
            losses['total'] = photo_loss
        elif params.loss_func_type == 'all':
            losses["total"] = feature_loss + photo_loss
    else:
        raise NotImplementedError

    return losses


def compute_eval_results(data_batch, output_batch, manager):
    # scale (    )
    grid_h, grid_w = data_batch["img1_full_rgb"].shape[2:]
    full_h, full_w = data_batch['img1_full_gray'].shape[2:]
    scale_x = grid_w / float(full_w)
    scale_y = grid_h / float(full_h)
    errs = geometricDistance_v2(data_batch, output_batch, scale_x, scale_y)

    eval_results = {}
    eval_results["img1_full_rgb_warp"] = output_batch['img1_full_rgb_warp']
    eval_results["img2_full_rgb_warp"] = output_batch['img2_full_rgb_warp']
    eval_results["errs"] = errs

    return eval_results
