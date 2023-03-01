import cv2
import sys
import ipdb
import logging
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = self.conv_last(x) # bs,8,h,w
        x = self.pool(x).reshape(-1, 4, 2)
        return x

    def forward(self, input):
        b, c, ph, pw = input['imgs_gray_patch'].shape
        
        output = {
            'offset': [],
            'points_pred': [],
            'H_flow': [],
            "img_warp": [],
        }
        
        num_cameras = int(c / 2)
        for i in range(num_cameras):
        
            x1_patch = input['imgs_gray_patch'][:, i*2:i*2+1]
            x2_patch = input['imgs_gray_patch'][:, i*2+1:i*2+2]
            x1_full = input["imgs_gray_full"][:, i*2:i*2+1]
            x2_full = input["imgs_gray_full"][:, i*2+1:i*2+2]
            
            fea1_patch = self.share_feature(x1_patch)
            fea2_patch = self.share_feature(x2_patch)

            x = torch.cat([fea1_patch, fea2_patch], dim=1)
            x = torch.cat([fea2_patch, fea1_patch], dim=1)
            offset_1 = self.nets_forward(x)
            offset_2 = self.nets_forward(x)

            points_2_pred = self.corners + offset_1
            points_1_pred = self.corners + offset_2
            
            points_1 = input['points_1'][:, i*2:i*2+4].contiguous()
            points_2 = input['points_2'][:, i*2:i*2+4].contiguous()
            homo_21 = dlt_homo(points_2_pred, points_1)
            homo_12 = dlt_homo(points_1_pred, points_2)

            img1_warp = warp_image_from_H(homo_21, x1_full, b, ph, pw)
            img2_warp = warp_image_from_H(homo_12, x2_full, b, ph, pw)
            
            output['offset'].append([offset_1, offset_2])
            output['points_pred'].append([points_1_pred, points_2_pred])
            output['H_flow'].append([homo_21, homo_12])
            output["img_warp"].append([img1_warp, img2_warp])
        
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
