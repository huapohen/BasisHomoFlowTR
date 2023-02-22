import os
import sys
import shutil
import torch
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


def vis_save_image_and_exit(output, input):
    svd = 'experiments/tmp'
    os.makedirs(svd, exist_ok=True)
    shutil.rmtree(svd)
    os.makedirs(svd)
    imgs_patch = input['imgs_gray_patch']
    img1_warp, img2_warp = output["img_warp"]
    im_diff_fw = imgs_patch[:, :1, ...] - img2_warp
    im_diff_bw = imgs_patch[:, 1:, ...] - img1_warp
    img_full_diff = input['img1_full'] - input['img2_full']
    img_full_rgb_diff = input['img1_full_rgb'] - input['img2_full_rgb']
    save_image(output["img_warp"][0], f'{svd}/img1_warp.jpg')
    save_image(output["img_warp"][1], f'{svd}/img2_warp.jpg')
    save_image(imgs_patch[:, :1, ...], f'{svd}/img1_patch.jpg')
    save_image(imgs_patch[:, 1:, ...], f'{svd}/img2_patch.jpg')
    save_image(im_diff_fw, f'{svd}/im_diff_fw.jpg')
    save_image(im_diff_bw, f'{svd}/im_diff_bw.jpg')
    save_image(input['imgs_gray_full'], f'{svd}/imgs_gray_full.png')
    save_image(output['fea_full'][0], f'{svd}/fea_full1.jpg')
    save_image(output['fea_full'][1], f'{svd}/fea_full2.jpg')
    save_image(img_full_diff, f'{svd}/img_full_diff.jpg')
    save_image(img_full_rgb_diff, f'{svd}/img_full_rgb_diff.jpg')
    save_image(input['img1_full_rgb'] / 255.0, f'{svd}/img1_full_rgb.jpg')
    save_image(input['img2_full_rgb'] / 255.0, f'{svd}/img2_full_rgb.jpg')
    save_image(output['img1_warp_rgb'] / 255.0, f'{svd}/img1_warp_rgb.jpg')
    save_image(output['img2_warp_rgb'] / 255.0, f'{svd}/img2_warp_rgb.jpg')

    sys.exit()


def compute_losses(output, input, params):
    losses = {}

    # compute losses
    if params.loss_type == "basic":
        imgs_patch = input['imgs_gray_patch']

        fea1_patch, fea2_patch = output["fea_patch"]
        img1_warp, img2_warp = output["img_warp"]
        fea1_warp, fea2_warp = output['fea_warp']
        fea1_patch_warp, fea2_patch_warp = output["fea_patch_warp"]

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

        if 'is_vis_and_exit' in vars(params) and params.is_vis_and_exit:
            vis_save_image_and_exit(output, input)

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
