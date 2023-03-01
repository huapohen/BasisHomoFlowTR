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
        corners = np.array([[0, 0], [cw, 0], [0, ch], [cw, ch]], dtype=np.float32)
        # The buffer is the same as the Parameter except that the gradient is not update.
        self.register_buffer('corners', torch.from_numpy(corners.reshape(1, 4, 2)))

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
        y = self.pool(x).squeeze(3)  # bs,8,1,1

        return y

    def forward(self, input):
        
        x1_patch, x2_patch = (
            input['imgs_gray_patch'][:, :1, ...],
            input['imgs_gray_patch'][:, 1:, ...],
        )
        x1_full, x2_full = (
            input["imgs_gray_full"][:, :1, ...],
            input["imgs_gray_full"][:, 1:, ...],
        )

        fea1_patch = self.share_feature(x1_patch)
        fea2_patch = self.share_feature(x2_patch)

        x = torch.cat([fea1_patch, fea2_patch], dim=1)
        x = torch.cat([fea2_patch, fea1_patch], dim=1)
        weight_f = self.nets_forward(x)
        weight_b = self.nets_forward(x)

        output = {}

        output['offset_1'] = weight_f.reshape(-1, 4, 2)
        output['offset_2'] = weight_b.reshape(-1, 4, 2)

        output['points_2_pred'] = self.corners + output['offset_1']
        output['points_1_pred'] = self.corners + output['offset_2']
        homo_21 = dlt_homo(output['points_2_pred'], input['points_1'])
        homo_12 = dlt_homo(output['points_1_pred'], input['points_2'])

        batch_size, _, h_patch, w_patch = x1_patch.size()
        bhw = (batch_size, h_patch, w_patch)
        
        img1_warp = warp_image_from_H(homo_21, x1_full, *bhw)
        img2_warp = warp_image_from_H(homo_12, x2_full, *bhw)

        output['H_flow'] = [homo_21, homo_12]
        output["img_warp"] = [img1_warp, img2_warp]
        
        return output
    
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

    # ipdb.set_trace()
    px, py = 63, 39
    pw, ph = 576, 320
    px, py = 0, 0
    pw, ph = 300, 300
    pts = [[px, py], [px + pw, py], [px, py + ph], [px + pw, py + ph]]
    pts_1 = torch.from_numpy(np.array(pts)[np.newaxis]).float()
    pts_2 = torch.from_numpy(np.array(pts)[np.newaxis]).float()

    img = cv2.imread('dataset/test/10467_A.jpg', 0)
    img_patch = img[py : py + ph, px : px + pw]

    input = torch.from_numpy(img[np.newaxis, np.newaxis].astype(np.float32))
    input_patch = torch.from_numpy(img_patch[np.newaxis, np.newaxis].astype(np.float32))
    data_dict = {}
    data_dict["imgs_gray_full"] = torch.cat([input, input], dim=1).cuda()
    data_dict["imgs_gray_patch"] = torch.cat([input_patch, input_patch], dim=1).cuda()
    data_dict['points_1'] = pts_1.cuda()
    data_dict['points_2'] = pts_2.cuda()

    params = EasyDict({'crop_size': (ph, pw)})
    net = Net(params)
    net.cuda()

    out = net(data_dict)
    print(out.keys())
    res = out['img_warp'][0][0].cpu().detach().numpy().transpose((1, 2, 0))
    cv2.imwrite('dataset/test/test_res.jpg', res)


if __name__ == '__main__':

    util_test_net_forward()
