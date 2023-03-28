import os
import cv2
import sys
import ipdb
import copy
import imageio
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as ip

from model.util import *
from easydict import EasyDict

warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = False
        # torch.backends.cuda.matmul.allow_tf32 = True
        self.params = params
        if 'forward_mode' not in vars(params):
            params.forward_mode = 'eval'
        self.inplanes = 64
        self.layers = [3, 4, 6, 3]
        self.basis_vector_num = 16
        if 'b16' in params.set_name:
            self.crop_size = params.crop_size_outdoor
        elif params.set_name == 'b07':
            self.crop_size = params.crop_size_dybev
        else:
            self.crop_size = params.crop_size
        ch, cw = self.crop_size
        corners = np.array([[0, 0], [cw, 0], [0, ch], [cw, ch]], dtype=np.float32)
        # The buffer is the same as the Parameter except that the gradient is not update.
        self.register_buffer('corners', torch.from_numpy(corners.reshape(1, 4, 2)))
        self.basis = gen_basis(ch, cw).unsqueeze(0).reshape(1, 8, -1)
        self.share_feature = ShareFeature(1)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.block = BasicBlock
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)

        self.conv_last = nn.Conv2d(
            512, 8, kernel_size=1, stride=1, padding=0, groups=8, bias=False
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def nets_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_last(x)  # bs,8,h,w
        x = self.pool(x).reshape(-1, 4, 2)
        return x

    def forward(self, input):
        params = self.params
        b, c, ph, pw = input['imgs_gray_patch'].shape
        bhwp = [b, ph, pw]
        b2hwp = [b, 2, ph, pw]
        b_, c_, h_, w_ = input["imgs_gray_full"].shape
        bhwf = [b_, h_, w_]
        start = input['start']
        output = {
            'offset': [],
            'points_pred': [],
            'H_flow': [],
            "img_warp": [],
            'mask_img1_warp': [],
            'mask_img2_warp': [],
            'x_full_warp': [],
            'H_full': [],
            'mask1_full_warp': [],
            'mask2_full_warp': [],
        }

        num_cameras = int(c / 2)
        for i in range(num_cameras):

            x1_patch = input['imgs_gray_patch'][:, i * 2 : i * 2 + 1]
            x2_patch = input['imgs_gray_patch'][:, i * 2 + 1 : i * 2 + 2]
            x1_full = input["imgs_gray_full"][:, i * 2 : i * 2 + 1]
            x2_full = input["imgs_gray_full"][:, i * 2 + 1 : i * 2 + 2]
            ones_patch = x1_patch.new_ones(x1_patch.shape)
            ones_full = x1_full.new_ones(x1_full.shape)
            scale_x = x1_full.shape[3] / x1_patch.shape[3]
            scale_y = x1_full.shape[2] / x1_patch.shape[2]

            fea1_patch = self.share_feature(x1_patch)
            fea2_patch = self.share_feature(x2_patch)

            x1 = torch.cat([fea1_patch, fea2_patch], dim=1)
            x2 = torch.cat([fea2_patch, fea1_patch], dim=1)

            if params.forward_version == 'basis':
                weight_f = self.nets_forward(x1).reshape(-1, 8, 1)
                weight_b = self.nets_forward(x2).reshape(-1, 8, 1)
                H_flow_f = (self.basis * weight_f).sum(1).reshape(*b2hwp)  # * 0
                H_flow_b = (self.basis * weight_b).sum(1).reshape(*b2hwp)  # * 0

                if (
                    params.test_pipeline_mode == 'resize'
                    and params.forward_mode == 'eval'
                ):
                    H_patch_b = copy.deepcopy(H_flow_b)
                    H_patch_f = copy.deepcopy(H_flow_f)
                    H_patch_b[:, 0] *= scale_x
                    H_patch_b[:, 1] *= scale_y
                    H_patch_f[:, 0] *= scale_x
                    H_patch_f[:, 1] *= scale_y
                    img1_warp = get_warp_flow(x1_patch, H_patch_b, start)  # _b  2<-1
                    img2_warp = get_warp_flow(x2_patch, H_patch_f, start)  # _f  1<-2
                else:
                    img1_warp = get_warp_flow(x1_full, H_flow_b, start)  # _b  2<-1
                    img2_warp = get_warp_flow(x2_full, H_flow_f, start)  # _f  1<-2
                output["img_warp"].append([img1_warp, img2_warp])
                output["H_flow"].append([H_flow_f, H_flow_b])

                if params.is_add_ones_mask:
                    if params.test_pipeline_mode == 'crop':
                        H_patch_f = upsample2d_flow_as(
                            H_flow_f, x1_full, mode="bilinear", if_rate=True
                        )
                        H_patch_b = upsample2d_flow_as(
                            H_flow_b, x1_full, mode="bilinear", if_rate=True
                        )
                        ones = ones_full
                    else:  # resize
                        H_patch_b = copy.deepcopy(H_flow_b)
                        H_patch_f = copy.deepcopy(H_flow_f)
                        H_patch_b[:, 0] *= scale_x
                        H_patch_b[:, 1] *= scale_y
                        H_patch_f[:, 0] *= scale_x
                        H_patch_f[:, 1] *= scale_y
                        ones = (
                            ones_full if params.forward_mode == 'train' else ones_patch
                        )
                    output["mask_img1_warp"].append(
                        get_warp_flow(ones, H_patch_b, start)
                    )
                    output["mask_img2_warp"].append(
                        get_warp_flow(ones, H_patch_f, start)
                    )

                    if 0:
                        H_full_f = upsample2d_flow_as(
                            H_flow_f, x1_full, mode="bilinear", if_rate=True
                        )
                        H_full_b = upsample2d_flow_as(
                            H_flow_b, x1_full, mode="bilinear", if_rate=True
                        )
                        output["mask1_full_warp"].append(
                            get_warp_flow(ones_full, H_full_b, start)
                        )
                        output["mask2_full_warp"].append(
                            get_warp_flow(ones_full, H_full_f, start)
                        )

                if 1:
                    H_full_f = upsample2d_flow_as(
                        H_flow_f, x1_full, mode="bilinear", if_rate=True
                    )
                    H_full_b = upsample2d_flow_as(
                        H_flow_b, x1_full, mode="bilinear", if_rate=True
                    )
                    x1_full_warp = get_warp_flow(x1_full, H_full_b, start)
                    x2_full_warp = get_warp_flow(x2_full, H_full_f, start)
                    output["H_full"].append([H_full_f, H_full_b])
                    output["x_full_warp"].append([x1_full_warp, x2_full_warp])

            elif params.forward_version == 'offset':
                offset_1 = self.nets_forward(x1)
                offset_2 = self.nets_forward(x2)
                # offset_1, offset_2 = 0, 0
                # offset_1, offset_2 = 50, 50
                points_2_pred = self.corners + offset_1
                points_1_pred = self.corners + offset_2
                points_1 = input['points_1'][:, i * 2 : i * 2 + 4].contiguous()
                points_2 = input['points_2'][:, i * 2 : i * 2 + 4].contiguous()
                homo_21 = dlt_homo(points_2_pred, points_1)
                homo_12 = dlt_homo(points_1_pred, points_2)
                ch, cw = self.crop_size
                img_h, img_w = x1_full.shape[2:]
                if (
                    params.test_pipeline_mode == 'resize'
                    and params.forward_mode == 'eval'
                ):
                    bhw = bhwf
                    if 1:
                        homo_21[:, :, 0] *= cw / img_w  # 先缩小
                        homo_21[:, :, 1] *= ch / img_h
                        homo_21[:, 0, :] *= img_w / cw  # 后放大
                        homo_21[:, 1, :] *= img_h / ch
                        homo_12[:, :, 0] *= cw / img_w
                        homo_12[:, :, 1] *= ch / img_h
                        homo_12[:, 0, :] *= img_w / cw
                        homo_12[:, 1, :] *= img_h / ch
                else:
                    bhw = bhwp
                img1_warp = warp_image_from_H(homo_21, x1_full, *bhw)
                img2_warp = warp_image_from_H(homo_12, x2_full, *bhw)
                if (
                    params.test_pipeline_mode == 'resize'
                    and params.forward_mode == 'eval'
                ):
                    img1_warp = F.interpolate(img1_warp, size=(ph, pw), mode='bilinear')
                    img2_warp = F.interpolate(img2_warp, size=(ph, pw), mode='bilinear')
                output['offset'].append([offset_1, offset_2])
                output['points_pred'].append([points_1_pred, points_2_pred])
                output['H_flow'].append([homo_21, homo_12])
                output["img_warp"].append([img1_warp, img2_warp])

                if params.is_add_ones_mask:
                    m1w = warp_image_from_H(homo_21, ones_full, *bhw)
                    m2w = warp_image_from_H(homo_12, ones_full, *bhw)
                    if (
                        params.test_pipeline_mode == 'resize'
                        and params.forward_mode == 'eval'
                    ):
                        m1w = F.interpolate(m1w, size=(ph, pw), mode='nearest')
                        m2w = F.interpolate(m2w, size=(ph, pw), mode='nearest')
                    output["mask_img1_warp"].append(m1w)
                    output["mask_img2_warp"].append(m2w)

                    if 0:
                        inps = [homo_21, homo_12, ones_full, bhw]
                        inps += [m1w, img1_warp, x2_patch]
                        self.test_pipeline(inps)

        return output

    def test_pipeline(self, inps):
        homo_21, homo_12, ones_full, bhw, m1w, img1_warp, x2_patch = inps
        m1w_v2 = warp_image_from_H_start(homo_21, ones_full, *bhw)
        m2w_v2 = warp_image_from_H_start(homo_12, ones_full, *bhw)
        mask1_warp = m1w[0].detach().cpu().permute(1, 2, 0) * 255
        mask2_warp = m1w[0].detach().cpu().permute(1, 2, 0) * 255
        mask1_warp_v2 = m1w_v2[0].detach().cpu().permute(1, 2, 0) * 255
        mask2_warp_v2 = m2w_v2[0].detach().cpu().permute(1, 2, 0) * 255
        mask1_warp[mask1_warp == 255] = 0
        mask1_warp[mask1_warp != 255] = 255

        mask1_warp = mask1_warp.numpy().astype(np.uint8)
        mask2_warp = mask2_warp.numpy().astype(np.uint8)
        mask1_warp_v2 = mask1_warp_v2.numpy().astype(np.uint8)
        mask2_warp_v2 = mask2_warp_v2.numpy().astype(np.uint8)

        vis = 'experiments/vis'
        os.makedirs(vis, exist_ok=True)
        cv2.imwrite(f'{vis}/m1w.jpg', mask1_warp)
        cv2.imwrite(f'{vis}/m2w.jpg', mask2_warp)
        cv2.imwrite(f'{vis}/mask1_warp_v2.jpg', mask1_warp_v2)
        cv2.imwrite(f'{vis}/mask2_warp_v2.jpg', mask2_warp_v2)
        mask1_warp_v2[mask1_warp_v2 < 255] = 1
        mask1_warp_v2[mask1_warp_v2 == 255] = 0
        mask1_warp_v2 *= 255
        cv2.imwrite(f'{vis}/mask1_warp_v2_verify.jpg', mask1_warp_v2)

        i1w = img1_warp[0].detach().permute(1, 2, 0).cpu()
        i1w = i1w.numpy().astype(np.uint8)
        i2p = x2_patch[0].detach().permute(1, 2, 0).cpu()
        i2p = i2p.numpy().astype(np.uint8)
        imageio.mimsave(f'{vis}/i1w.gif', [i2p, i1w], 'GIF', duration=0.5)
        cv2.imwrite(f'{vis}/i1w.jpg', i1w)
        cv2.imwrite(f'{vis}/i2p.jpg', i2p)
        print(i1w.shape, i2p.shape)
        # ip()
        print()

    def compute_homo(self, input1, input2):
        fea1_patch = self.share_feature(input1)
        fea2_patch = self.share_feature(input2)

        x = torch.cat([fea1_patch, fea2_patch], dim=1)
        weight_f = self.nets_forward(x)

        offset = weight_f.reshape(-1, 4, 2)
        corners2_pred = self.corners + offset
        corners1 = self.corners.expand_as(corners2_pred)
        homo_12 = dlt_homo(corners1, corners2_pred)

        return homo_12


# ========================================================================================================================


def fetch_net(params):

    if params.net_type == "basic":
        net = Net(params)
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return net


def util_test_net_forward():
    '''add this line, because GTX30 serials' default torch.matmul() on cuda is uncorrected'''
    torch.backends.cuda.matmul.allow_tf32 = False
    from easydict import EasyDict
    import ipdb
    import json

    with open('experiments/baseshomo/exp_41/params.json') as f:
        params = EasyDict(json.load(f))
    # params.crop_size = [ph, pw]
    # params.test_pipeline_mode = 'crop'
    params.forward_version = 'offset'
    # params.forward_version = 'basis'
    params.is_add_ones_mask = True
    # params.set_name = 'nature'
    # params.set_name = 'b16'
    net = Net(params)
    net.cuda()

    # ipdb.set_trace()
    # px, py = 50, 50
    # pw, ph = 300, 300
    px, py = 0, 0
    pw, ph = 576, 320
    pts = [[px, py], [px + pw, py], [px, py + ph], [px + pw, py + ph]]
    pts_1 = torch.from_numpy(np.array(pts)[np.newaxis]).float()
    pts_2 = torch.from_numpy(np.array(pts)[np.newaxis]).float()

    img = cv2.imread('dataset/test/10467_A.jpg', 0)
    cv2.imwrite('experiments/vis/ori.jpg', img)

    if params.test_pipeline_mode == 'crop':
        img_patch = img[py : py + ph, px : px + pw]
    else:
        img_patch = cv2.resize(img, (576, 320))

    input = torch.from_numpy(img[np.newaxis, np.newaxis].astype(np.float32))
    input_patch = torch.from_numpy(img_patch[np.newaxis, np.newaxis].astype(np.float32))
    data_dict = {}
    data_dict["imgs_gray_full"] = torch.cat([input, input], dim=1).cuda()
    data_dict["imgs_gray_patch"] = torch.cat([input_patch, input_patch], dim=1).cuda()
    data_dict['points_1'] = pts_1.cuda()
    data_dict['points_2'] = pts_2.cuda()
    data_dict['start'] = torch.tensor([0, 0]).reshape(2, 1, 1).float().cuda()

    with torch.no_grad():
        out = net(data_dict)
    print(out.keys())
    res = out['img_warp'][0][0][0].cpu().detach().numpy().transpose((1, 2, 0))
    cv2.imwrite('dataset/test/test_res.jpg', res)


if __name__ == '__main__':

    util_test_net_forward()
