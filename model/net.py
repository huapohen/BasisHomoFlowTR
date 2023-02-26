from __future__ import absolute_import, division, print_function
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import warnings
from torch.nn.modules.utils import _pair, _quadruple
from model.util import *
from easydict import EasyDict


warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = False


class Net(nn.Module):
    def __init__(self, params):

        super(Net, self).__init__()
        self.params = params
        self.inplanes = 64
        self.layers = [3, 4, 6, 3]
        self.basis_vector_num = 16
        ch, cw = params.crop_size
        corner = np.array([[[0, 0], [cw, 0], [0, ch], [cw, ch]]], dtype=np.float32)
        ch, cw = params.crop_size_dybev
        corner_bev = np.array([[[0, 0], [cw, 0], [0, ch], [cw, ch]]], dtype=np.float32)
        # The buffer is the same as the Parameter except that the gradient is not updated.
        self.register_buffer('corner', torch.from_numpy(corner))
        self.register_buffer('corner_bev', torch.from_numpy(corner_bev))

        self.share_feature = ShareFeature(1)
        inc = 2 if params.is_gray else 6
        self.bias = False
        self.conv1 = nn.Conv2d(inc, 64, 7, 2, 3, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.block = BasicBlock
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        if self.params.is_add_lrr_module:
            self.sp_layer3 = Subspace(256)
            self.sp_layer4 = Subspace(512)

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
        if self.params.is_add_lrr_module:
            x = self.sp_layer3(self.layer3(x))
            x = self.sp_layer4(self.layer4(x))
        else:
            x = self.layer3(x)
            x = self.layer4(x)
        x = self.conv_last(x)  # bs,8,h,w
        y = self.pool(x).squeeze(3)  # bs,8,1,1

        return y

    def forward(self, input):
        x1_patch_gray = input['img1_patch_gray']
        x2_patch_gray = input['img2_patch_gray']
        output = {}

        fea1_patch = self.share_feature(x1_patch_gray)
        fea2_patch = self.share_feature(x2_patch_gray)

        x = torch.cat([fea1_patch, fea2_patch], dim=1)
        output['weight_1'] = self.nets_forward(x)

        x = torch.cat([fea2_patch, fea1_patch], dim=1)
        output['weight_2'] = self.nets_forward(x)

        output['fea1_patch'] = fea1_patch
        output['fea2_patch'] = fea2_patch

        return output


def warp_head(model, input, output):
    start = input['start']
    x1_patch_gray = input['img1_patch_gray']
    x2_patch_gray = input['img2_patch_gray']
    x1_patch_rgb = input['img1_patch_rgb']
    x2_patch_rgb = input['img2_patch_rgb']
    x1_full_gray = input['img1_full_gray']
    x2_full_gray = input['img2_full_gray']
    x1_full_rgb = input['img1_full_rgb']
    x2_full_rgb = input['img2_full_rgb']

    output['offset_1'] = output['weight_1'].reshape(-1, 4, 2)
    output['offset_2'] = output['weight_2'].reshape(-1, 4, 2)
    output['points_2_pred'] = input['points_1'] + output['offset_1']
    output['points_1_pred'] = input['points_2'] + output['offset_2']
    homo_21 = dlt_homo(output['points_2_pred'], input['points_1'], method="Axb")
    homo_12 = dlt_homo(output['points_1_pred'], input['points_2'], method="Axb")

    batch_size, _, h_patch, w_patch = x1_patch_gray.size()
    batch_size, _, h_full, w_full = x1_full_gray.size()
    tgt_hwp = (batch_size, h_patch, w_patch)
    tgt_hwf = (batch_size, h_full, w_full)

    # all features are gray

    x1_patch_gray_warp_p = warp_from_H(homo_21, x1_full_gray, *tgt_hwp, start)
    x2_patch_gray_warp_p = warp_from_H(homo_12, x2_full_gray, *tgt_hwp, start)
    fea1_patch_warp = model.share_feature(x1_patch_gray_warp_p)
    fea2_patch_warp = model.share_feature(x2_patch_gray_warp_p)

    # suffix `_p` means patch size and `_f` means full size, therefore
    #  after `warp_from_H` the `_patch_warp` and `_full_warp` means nothing!
    fea1_full = model.share_feature(x1_full_gray)
    fea2_full = model.share_feature(x2_full_gray)
    fea1_full_warp_p = warp_from_H(homo_21, fea1_full, *tgt_hwp, start)
    fea2_full_warp_p = warp_from_H(homo_12, fea2_full, *tgt_hwp, start)
    fea1_full_warp_f = warp_from_H(homo_21, fea1_full, *tgt_hwf, 0)
    fea2_full_warp_f = warp_from_H(homo_12, fea2_full, *tgt_hwf, 0)

    # img1 warp to img2, img2_pred = img1_warp
    x1_patch_rgb_warp_p = warp_from_H(homo_21, x1_patch_rgb, *tgt_hwp, start)
    x2_patch_rgb_warp_p = warp_from_H(homo_12, x2_patch_rgb, *tgt_hwp, start)
    x1_full_rgb_warp_f = warp_from_H(homo_21, x1_full_rgb, *tgt_hwf, 0)
    x2_full_rgb_warp_f = warp_from_H(homo_12, x2_full_rgb, *tgt_hwf, 0)

    output['H_flow_21'] = homo_21
    output['H_flow_12'] = homo_12

    output['fea1_full'] = fea1_full
    output['fea2_full'] = fea2_full
    output["fea1_full_warp_p"] = fea1_full_warp_p
    output["fea2_full_warp_p"] = fea2_full_warp_p
    output["fea1_full_warp_f"] = fea1_full_warp_f
    output["fea2_full_warp_f"] = fea2_full_warp_f
    output["fea1_patch_warp"] = fea1_patch_warp
    output["fea2_patch_warp"] = fea2_patch_warp

    output["img1_patch_gray_warp_p"] = x1_patch_gray_warp_p
    output["img2_patch_gray_warp_p"] = x2_patch_gray_warp_p
    output["img1_patch_rgb_warp_p"] = x1_patch_rgb_warp_p
    output["img2_patch_rgb_warp_p"] = x2_patch_rgb_warp_p
    output['img1_full_rgb_warp_f'] = x1_full_rgb_warp_f
    output['img2_full_rgb_warp_f'] = x2_full_rgb_warp_f

    return output


def fetch_net(params):

    if params.net_type == "basic":
        net = Net(params)
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return net


def second_stage(params, input, output):
    output = warp_head(input, output)

    # if params.is_unit_test_model:
    #     import sys
    #     import model.unit_test as ut

    #     sys.exit()

    return output
