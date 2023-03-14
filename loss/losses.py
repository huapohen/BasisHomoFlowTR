import cv2
import sys
import ipdb
import copy
import torch
import imageio
import torch.nn as nn
import numpy as np
from model import net
from model.util import *
from torchvision.utils import save_image
from ipdb import set_trace as ip



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


def geometricDistance_origin(correspondence, flow, start=0):
    flow = flow.permute(1, 2, 0).cpu().detach().numpy()

    p1 = correspondence[0] # 0
    p2 = correspondence[1] # 1
    
    # xy = start.detach().cpu().numpy()
    # x = xy[0].item()
    # y = xy[1].item()
    # rx = 640 / 576
    # ry = 360 / 320
    # p1 = [p1[0] - x, p1[1] - y]
    # p2 = [p2[0] - x, p2[1] - y]
    # p1 = [p1[0] * rx, p1[1] * ry]
    # p2 = [p2[0] * rx, p2[1] * ry]
    # p1 = [(p1[0] - x) * rx, (p1[1] - y) * ry]
    # p2 = [(p2[0] - x) * rx, (p2[1] - y) * ry]
    # p1 = [(p1[0] - x) / rx, (p1[1] - y) / ry]
    # p2 = [(p2[0] - x) / rx, (p2[1] - y) / ry]
    # p1 = [p1[0] / rx, p1[1] / ry]
    # p2 = [p2[0] / rx, p2[1] / ry]
    try:
        if isinstance(correspondence[1][0], float):
            result = p2 - (p1 - flow[int(p1[1]), int(p1[0])])
            error = np.linalg.norm(result)
        else:
            result = [p2 - (p1 - flow[p1[1], p1[0]]), p1 - (p2 - flow[p2[1], p2[0]])]
            error = min(np.linalg.norm(result[0]), np.linalg.norm(result[1]))
    except:
        error = np.float32(0)
    return error


def geometricDistance_cascade(correspondence, i, H_flow_f_list, ratio=1.0):
    flow_list = []
    layers = len(H_flow_f_list)
    for k in range(layers):
        flow = H_flow_f_list[k][i].permute(1, 2, 0).cpu().detach().numpy()
        flow_list.append(flow)

    p1 = correspondence[0]  # 0
    p2 = correspondence[1]  # 1
    flow = cv2.resize(
        flow, (0, 0), fx=1 / ratio, fy=1 / ratio, interpolation=cv2.INTER_LINEAR
    )  # h, w, 2 -> y, x

    p1_xy = copy.deepcopy(p1)
    p2_xy = copy.deepcopy(p2)
    for k in range(layers):
        p1_xy = flow_list[k][int(p1_xy[1]), int(p1_xy[0])]
        p2_xy = flow_list[k][int(p2_xy[1]), int(p2_xy[0])]
    # ipdb.set_trace()
    if isinstance(correspondence[1][0], float):
        result = p2 - (p1 - p1_xy)
        error = np.linalg.norm(result)
    else:
        result = [p2 - (p1 - p1_xy), p1 - (p2 - p2_xy)]
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


def compute_losses_origin(params, train_batch, output):
    losses = {}
    imgs_patch = train_batch[f'imgs_gray_patch']
    img1_patch = imgs_patch[:, :1]
    img2_patch = imgs_patch[:, 1:]
    loss_list = []
    num = len(output['img_warp'])
    for i in range(num):
        if num == 1 or (num > 1 and i >= 1):
            img1_warp, img2_warp = output["img_warp"][i]
            im_diff_fw = img1_patch - img2_warp
            im_diff_bw = img2_patch - img1_warp
            loss_f = photo_loss_function(diff=im_diff_fw, q=1, averge=True)
            loss_b = photo_loss_function(diff=im_diff_bw, q=1, averge=True)
            loss_list.append(loss_f + loss_b)
        # ipdb.set_trace()
    if num == 1:
        losses["total"] = loss_list[0]
    else:
        num -= 1
        loss = 0
        for i in range(num):
            loss += loss_list[i]
        losses['total'] = loss
    return losses


def compute_eval_results_origin(data_batch, output_batch, params):
    imgs_full = data_batch["imgs_ori"]
    points = data_batch["points"]
    img1_full = imgs_full[:, :3]
    
    H_flow_f_patch, H_flow_b_patch = output_batch['H_flow'][-1]
    H_flow_f_full = net.upsample2d_flow_as(H_flow_f_patch, imgs_full, "bilinear", True)
    H_flow_b_full = net.upsample2d_flow_as(H_flow_b_patch, imgs_full, "bilinear", True)
    
    # basis 在每一个像素点上的 H 是一样的，此篇论文是整体用一个basis，而不是分grid用不同basis
    # 所以，直接插值即可；  插值和缩放是两码事，缩放要 乘以或者除以 一个缩放因子
    # 哪种情况缩放？  不管是resize还是Homo * scale对此数据集竟然不影响？ 
    # 只要resize到 points的分辨率大小就行了！？
    img1_full_warp = net.get_warp_flow(img1_full, H_flow_b_full, start=0)
    H_flow_f = H_flow_f_full
    # H_flow_f = H_flow_f_patch

    errs = []
    errs_p = []

    for i in range(len(points)):
        point = eval(points[i])
        err = 0
        tmp = []
        for j in range(6):
            pts = point['matche_pts'][j]
            err_p = geometricDistance_origin(pts, H_flow_f[i], data_batch['start'][i])
            err += err_p
            tmp.append(err_p)

        errs.append(err / (j + 1))
        errs_p.append(tmp)

    eval_results = {}
    eval_results["img1_full_warp"] = img1_full_warp
    eval_results["errs"] = errs
    eval_results["errs_p"] = errs_p

    return eval_results


def compute_eval_results_new(data_batch, output_batch, params):
    eval_results = {}

    imgs_full = data_batch["imgs_ori"]
    points = data_batch["points"]
    batch_size, _, grid_h, grid_w = imgs_full.shape
    bhw = (batch_size, grid_h, grid_w)

    errs, errs_p = [], []

    if params.model_version == 'v1':
        H_flow_f_list = []
        # for i in range(len(output_batch['H_flow'])):
        if 1:
            i = -1
            H_flow_f = output_batch['H_flow'][i][0]
            H_flow_f = upsample2d_flow_as(H_flow_f, imgs_full, "bilinear", if_rate=True)
            H_flow_f_list.append(H_flow_f)
            if params.is_save_gif:
                H_flow_b = output_batch['H_flow'][i][1]
                H_flow_b = upsample2d_flow_as(
                    H_flow_b, imgs_full, "bilinear", if_rate=True
                )
                img1_full_warp = get_warp_flow(imgs_full[:, :3], H_flow_b, 0)
        if params.is_save_gif:
            eval_results["img1_full_warp"] = img1_full_warp
        # calc err
        points = data_batch["points"]
        for i in range(len(points)):  # len(points)
            point = eval(points[i])
            err = 0
            tmp = []
            for j in range(6):  # len(point['matche_pts'])
                points_value = point['matche_pts'][j]
                err_p = geometricDistance_origin(points_value, H_flow_f_list[-1][i])
                # err_p = geometricDistance_cascade(points_value, i, H_flow_f_list, params.resize_ratio)
                err += err_p
                tmp.append(err_p)
            errs.append(err / (j + 1))
            errs_p.append(tmp)

    elif params.model_version == 'v2':
        H_flow_f, H_flow_b = output_batch['H_flow']
        if params.is_save_gif:
            eval_results["img1_full_warp"] = warp_image_from_H(
                H_flow_b, imgs_full[:, :3, ...], *bhw
            )
        # calc err
        scale_x = grid_w / float(data_batch['imgs_gray_full'].shape[3])
        scale_y = grid_h / float(data_batch['imgs_gray_full'].shape[2])
        errs = geometricDistance_v2(data_batch, output_batch, scale_x, scale_y)

    eval_results["errs"] = errs
    eval_results["errs_p"] = errs_p

    return eval_results
