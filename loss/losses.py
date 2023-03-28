import os
import sys
import cv2
import copy
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
from ipdb import set_trace as ip


def triplet_loss(a, p, n, margin=1.0, exp=1, reduce=False, size_average=False):

    triplet_loss = nn.TripletMarginLoss(
        margin=margin, p=exp, reduce=reduce, size_average=size_average
    )
    return triplet_loss(a, p, n)


def photo_loss_function(diff, q, averge=True, mask=False):
    diff = (torch.abs(diff) + 0.01).pow(q)
    if averge:
        loss_mean = diff.mean()
        if mask:
            _h, _w = diff.shape[2:]
            loss_mean_list = []
            for i in range(len(diff)):
                zero_sum = (diff[i] == 0.01).sum()
                if zero_sum == _h * _w:
                    ratio = zero_sum.new_ones(zero_sum.shape).float()
                else:
                    ratio = (_h * _w - zero_sum) / (_h * _w)
                val = diff[i].mean() / ratio
                loss_mean_list.append(val.unsqueeze(0))
            loss_mean = torch.cat(loss_mean_list, 0).mean()
            # ip()
    else:
        # loss_mean = diff.sum()
        loss_mean = diff
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
        'front': ['left', 'right'],  # 0, 1 img1=front
        'back': ['left', 'right'],  # 2, 3 img1=back
        'left': ['front', 'back'],  # 4, 5 img1=left
        'right': ['front', 'back'],  # 6, 7 img1=right
    }
    sequence = {
        'basic': [0, 1, 2, 3, 4, 5, 6, 7],
        'all_fblr': [0, 1, 2, 3, 4, 5, 6, 7],
        'front_first': [0, 1, 5, 7],
        'back_first': [2, 3, 4, 6],
        'left_first': [4, 5, 1, 3],
        'right_first': [6, 7, 0, 2],
    }
    return sequence


def save_inference_images(params, output, input, inps):
    img1_warp, img2_warp, x1_patch, x2_patch, k, loss_f, loss_b = inps
    vis = 'experiments/vis'
    if os.path.exists(vis) and k == 0:
        shutil.rmtree(vis)
    os.makedirs(vis, exist_ok=True)

    def unnormalize(im, multiply_std=True, add_mean=False):
        mean_I = np.array([118.93, 113.97, 102.60]).mean()
        std_I = np.array([69.85, 68.81, 72.45]).mean()
        im = im[0].permute(1, 2, 0).cpu().numpy()
        im = im * std_I
        # ip()
        if add_mean:
            im[im != 0] += mean_I
            # im[im != 0] = 50
        return im.astype(np.uint8)

    i1w = unnormalize(img1_warp)
    i2w = unnormalize(img2_warp)
    cv2.imwrite(f'{vis}/{k}_i1w_{loss_b:.4f}.jpg', i1w)
    cv2.imwrite(f'{vis}/{k}_i2w_{loss_f:.4f}.jpg', i2w)
    p1 = unnormalize(x1_patch)
    p2 = unnormalize(x2_patch)
    cv2.imwrite(f'{vis}/{k}_p1.jpg', p1)
    cv2.imwrite(f'{vis}/{k}_p2.jpg', p2)
    imageio.mimsave(f'{vis}/{k}_g1.gif', [p2, i1w], 'GIF', duration=0.5)
    imageio.mimsave(f'{vis}/{k}_g2.gif', [p1, i2w], 'GIF', duration=0.5)
    imageio.mimsave(f'{vis}/{k}_ori_gray.gif', [p1, p2], 'GIF', duration=0.5)
    i1f = input['imgs_ori'][0, :3].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    i2f = input['imgs_ori'][0, 3:].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    i1f = cv2.cvtColor(i1f, cv2.COLOR_BGR2RGB)
    i2f = cv2.cvtColor(i2f, cv2.COLOR_BGR2RGB)
    imageio.mimsave(f'{vis}/{k}_ori_rgb.gif', [i1f, i2f], 'GIF', duration=0.5)

    if params.is_add_ones_mask:
        mask1_warp = output["mask_img1_warp"][0][0].permute(1, 2, 0) * 255
        mask2_warp = output["mask_img2_warp"][0][0].permute(1, 2, 0) * 255
        mask1_warp = mask1_warp.cpu().numpy().astype(np.uint8)
        mask2_warp = mask2_warp.cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'{vis}/{k}_m1w.jpg', mask1_warp)
        cv2.imwrite(f'{vis}/{k}_m2w.jpg', mask2_warp)
        # --------------------------------------------------
        i1w_m = output["mask_img1_warp"][0]
        i2w_m = output["mask_img2_warp"][0]
        m_img1w = img1_warp * i1w_m
        m_img2w = img2_warp * i2w_m
        m_img1w_sub = img1_warp * (i1w_m > 0.001)
        m_img2w_sub = img2_warp * (i2w_m > 0.001)
        m_img1w_reverse = img1_warp * (~i1w_m.bool())
        m_img2w_reverse = img2_warp * (~i2w_m.bool())
        ip()  # i1w_m[0, 0, :50, 140:]
        ind = 0
        names = [
            'm_img1w',
            'm_img2w',
            'm_img1w_reverse',
            'm_img2w_reverse',
            'm_img1w_sub',
            'm_img2w_sub',
        ]
        for ix in [
            m_img1w,
            m_img2w,
            m_img1w_reverse,
            m_img2w_reverse,
            m_img1w_sub,
            m_img2w_sub,
        ]:
            ix = unnormalize(ix, multiply_std=True, add_mean=True)
            cv2.imwrite(f'{vis}/{k}_{names[ind]}.jpg', ix)
            ind += 1

    if 1:
        x1_full_warp, x2_full_warp = output['x_full_warp'][0]
        x1fw = unnormalize(x1_full_warp)
        x2fw = unnormalize(x2_full_warp)
        cv2.imwrite(f'{vis}/{k}_x1fw.jpg', x1fw)
        cv2.imwrite(f'{vis}/{k}_x2fw.jpg', x2fw)

    if 1:
        H_flow_f, H_flow_b = output['H_flow'][0]
        imgs_full = input['imgs_ori']
        img1_full = imgs_full[:, :3]
        img2_full = imgs_full[:, 3:]
        H_flow_f = upsample2d_flow_as(
            H_flow_f, imgs_full, mode="bilinear", if_rate=True
        )
        H_flow_b = upsample2d_flow_as(
            H_flow_b, imgs_full, mode="bilinear", if_rate=True
        )
        img1_full_warp = get_warp_flow(img1_full, H_flow_b, 0)
        img2_full_warp = get_warp_flow(img2_full, H_flow_f, 0)

        def to_rgb(img):
            img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        img1_full = to_rgb(img1_full)
        img2_full = to_rgb(img2_full)
        img1_full_warp = to_rgb(img1_full_warp)
        img2_full_warp = to_rgb(img2_full_warp)
        imageio.mimsave(
            f'{vis}/{k}_ori_f.gif', [img1_full, img2_full_warp], 'GIF', duration=0.5
        )
        imageio.mimsave(
            f'{vis}/{k}_ori_b.gif', [img2_full, img1_full_warp], 'GIF', duration=0.5
        )

        # if 1:
        if 0:

            def _unnorm(im):
                im = im[0].permute(1, 2, 0).cpu().numpy()
                im = im * 255
                return im.astype(np.uint8)

            mask1 = _unnorm(output['mask_img1_warp'][0])
            mask2 = _unnorm(output['mask_img2_warp'][0])
            cv2.imwrite(f'{vis}/{k}_mask_b.jpg', mask1)
            cv2.imwrite(f'{vis}/{k}_mask_f.jpg', mask2)

            mask1_full = _unnorm(output["mask1_full_warp"][0])
            mask2_full = _unnorm(output["mask2_full_warp"][0])
            cv2.imwrite(f'{vis}/{k}_mask_b_full.jpg', mask1_full)
            cv2.imwrite(f'{vis}/{k}_mask_f_full.jpg', mask2_full)

    sys.exit()


def compute_losses(output, input, params, k=0):
    losses = {}
    imgs_patch = input['imgs_gray_patch']
    ph, pw = imgs_patch.shape[2:]
    edge_loss = 0
    photo_loss = 0

    for i, camera in enumerate(params.camera_list):
        img1_warp, img2_warp = output["img_warp"][i]
        img1 = imgs_patch[:, i * 2 : i * 2 + 1]
        img2 = imgs_patch[:, i * 2 + 1 : i * 2 + 2]
        im_diff_fw = img1 - img2_warp
        im_diff_bw = img2 - img1_warp
        if params.calc_gt_photo_loss:
            im_diff_fw = img1 - img2
            im_diff_bw = img2 - img1
        if params.is_add_ones_mask:
            im_diff_fw *= output['mask_img2_warp'][i]
            im_diff_bw *= output['mask_img1_warp'][i]
        is_mask = True if params.is_add_ones_mask else False
        photo_loss_f = photo_loss_function(
            diff=im_diff_fw, q=1, averge=True, mask=is_mask
        )
        photo_loss_b = photo_loss_function(
            diff=im_diff_bw, q=1, averge=True, mask=is_mask
        )

        if params.is_save_intermediate_results:
            img1_mean = (torch.abs(img1) + 0.01).pow(1).mean()
            img2_mean = (torch.abs(img2) + 0.01).pow(1).mean()
            img1w_mean = (torch.abs(img1_warp) + 0.01).pow(1).mean()
            img2w_mean = (torch.abs(img2_warp) + 0.01).pow(1).mean()
            print(
                f'photo_loss_f: {photo_loss_f:.4f}, img1={img1_mean:.4f}, img2w={img2w_mean:.4f}'
            )
            print(
                f'photo_loss_b: {photo_loss_b:.4f}, img2={img2_mean:.4f}, img1w={img1w_mean:.4f}'
            )
            inps = img1_warp, img2_warp, img1, img2, k, photo_loss_f, photo_loss_b
            save_inference_images(params, output, input, inps)

        if params.loss_sequence == '21':
            photo_loss += photo_loss_f
        elif params.loss_sequence == '12':
            photo_loss += photo_loss_b
        else:
            photo_loss += photo_loss_f + photo_loss_b

        if params.is_add_edge_loss:
            edge_diff_f1 = (
                im_diff_fw[:, :, 1:-1, 1:-1] - im_diff_fw[:, :, 2:, 1:-1]
            ).abs()
            edge_diff_f2 = (
                im_diff_fw[:, :, 1:-1, 1:-1] - im_diff_fw[:, :, 1:-1, 2:]
            ).abs()
            edge_loss_f = edge_diff_f1 + edge_diff_f2
            edge_diff_b1 = (
                im_diff_bw[:, :, 1:-1, 1:-1] - im_diff_bw[:, :, 2:, 1:-1]
            ).abs()
            edge_diff_b2 = (
                im_diff_bw[:, :, 1:-1, 1:-1] - im_diff_bw[:, :, 1:-1, 2:]
            ).abs()
            edge_loss_b = edge_diff_b1 + edge_diff_b2
            if params.is_add_ones_mask:
                edge_loss_f *= output['mask_img2_warp'][i]
                edge_loss_b *= output['mask_img1_warp'][i]
            edge_loss_f = edge_loss_f.sum() / (edge_loss_f > 0.01).sum()
            edge_loss_b = edge_loss_b.sum() / (edge_loss_b > 0.01).sum()
            edge_loss += edge_loss_f + edge_loss_b

    num_cam = len(params.camera_list)
    if params.is_add_photo_loss:
        losses['photo'] = photo_loss / num_cam
    if params.is_add_edge_loss:
        losses['edge'] = edge_loss / num_cam
    losses['total'] = (photo_loss + edge_loss) / num_cam
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
        im_diff_fw = imgs_patch[:, i * 2 : i * 2 + 1] - img2_warp
        im_diff_bw = imgs_patch[:, i * 2 + 1 : i * 2 + 2] - img1_warp
        photo_loss_f = photo_loss_function(diff=im_diff_fw, q=1, averge=True)
        photo_loss_b = photo_loss_function(diff=im_diff_bw, q=1, averge=True)
        photo_losses.append(photo_loss_f)
        photo_losses.append(photo_loss_b)

    if len(idx_seq) == 8:
        total_loss = 0
        for i, cam in enumerate(params.camera_list):
            losses[cam] = photo_losses[i * 2] + photo_losses[i * 2 + 1]
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
            coef = [1 / 8 for _ in range(8)]

        total_loss = [a * x.unsqueeze(0) for a, x in zip(coef, total_loss)]
        # ipdb.set_trace()
        losses['total'] = 2 * torch.cat(total_loss).sum()

    return losses


def compute_eval_results(data_batch, output_batch, params):
    imgs_full = data_batch["imgs_ori"]
    batch_size, _, img_h, img_w = data_batch["imgs_ori"].shape
    bhwf = (batch_size, img_h, img_w)
    errs = np.zeros(batch_size)
    img1_full_warp_list = []
    img2_full_warp_list = []

    _h, _w = data_batch['imgs_gray_full'].shape[2:]
    scale_x = img_w / _w
    scale_y = img_h / _h

    for i, camera in enumerate(params.camera_list):
        img_1_full = imgs_full[:, i * 6 : i * 6 + 3]
        img_2_full = imgs_full[:, i * 6 + 3 : i * 6 + 6]

        if params.forward_version == 'offset':
            homo_21, homo_12 = output_batch['H_flow'][i]
            img1_full_warp = warp_image_from_H(homo_21, img_1_full, *bhwf)
            img2_full_warp = warp_image_from_H(homo_12, img_2_full, *bhwf)
            err = geometricDistance_offset(
                i, data_batch, output_batch, scale_x, scale_y
            )
        elif params.forward_version == 'basis':
            H_flow_f, H_flow_b = output_batch['H_flow'][i]
            # ipdb.set_trace()
            H_flow_f = upsample2d_flow_as(
                H_flow_f, imgs_full, mode="bilinear", if_rate=True
            )
            H_flow_b = upsample2d_flow_as(
                H_flow_b, imgs_full, mode="bilinear", if_rate=True
            )
            img1_full_warp = get_warp_flow(img_1_full, H_flow_b, 0)
            img2_full_warp = get_warp_flow(img_2_full, H_flow_f, 0)
            err = np.float32(0)
        else:
            raise

        # ipdb.set_trace()
        img1_full_warp_list.append(img1_full_warp)
        img2_full_warp_list.append(img2_full_warp)
        errs += err

    eval_results = {}
    eval_results["img1_full_warp"] = torch.cat(img1_full_warp_list, 1)
    eval_results["img2_full_warp"] = torch.cat(img2_full_warp_list, 1)
    eval_results["errs"] = errs / len(params.camera_list)

    return eval_results
