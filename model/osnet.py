"""Defines the neural network, losss function and metrics"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .swin_multi import SwinTransformer
from timm.models.layers import trunc_normal_
from .module.aspp import ASPP
import torch.nn.functional as F
from .basenet import warp_image_from_H, dlt_homo
import numpy as np
import cv2

__all__ = ['OSNet', 'Discriminator']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class OSNet(nn.Module):
    # 224*224
    def __init__(self, params, backbone, init_mode="resnet", norm_layer=nn.LayerNorm):
        super(OSNet, self).__init__()

        self.init_mode = init_mode
        self.params = params
        self.crop_size = params.crop_size  # h,w 
        self.fea_extra = self.feature_extractor(self.params.in_channels, 1)
        self.h_net = backbone(params, norm_layer=norm_layer)

        ch, cw = self.crop_size
        corners = np.array([[0, 0], [cw, 0], [0, ch], [cw, ch]], dtype = np.float32)
        self.register_buffer('corners', torch.from_numpy(corners.reshape(1, 4, 2)))

        self.apply(self._init_weights)
        self.mask_pred = self.mask_predictor(32)

    def _init_weights(self, m):
        if "swin" in self.init_mode:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif "resnet" in self.init_mode:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def feature_extractor(input_channels, out_channles, kernel_size=3, padding=1):
        layers = []
        channels = [input_channels // 2, 4, 8, out_channles]
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def mask_predictor(input_channels, reduction=1):
        layers = []
        layers.append(nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3,
                                stride=1, padding=1, groups=2, bias=False))
        layers.append(ASPP(in_channels=input_channels * 2, out_channels=input_channels // 4, dilations=(1, 2, 5, 1)))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=input_channels // reduction, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=input_channels // reduction, out_channels=1, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, data_batch):
        img1_full, img2_full = data_batch["imgs_gray_full"][:, :1, :, :], data_batch["imgs_gray_full"][:, 1:, :, :]
        img1_patch, img2_patch = data_batch["imgs_gray_patch"][:, :1, :, :], data_batch["imgs_gray_patch"][:, 1:, :, :]
        batch_size, _, h_patch, w_patch = data_batch["imgs_gray_patch"].size()
        start = data_batch['start']

        # ==========================full features======================================
        img1_patch_fea, img2_patch_fea = list(map(self.fea_extra, [img1_patch, img2_patch]))
        img1_full_fea, img2_full_fea = list(map(self.fea_extra, [img1_full, img2_full]))

        # ========================forward ====================================

        forward_fea = torch.cat([img1_patch_fea, img2_patch_fea], dim=1)
        weight_f = self.h_net(forward_fea)

        # ========================backward===================================
        backward_fea = torch.cat([img2_patch_fea, img1_patch_fea], dim=1)
        weight_b = self.h_net(backward_fea)

        # ===output===
        bhw = (batch_size, h_patch, w_patch)

        output = {}

        output['offset_1'] = weight_f.reshape(batch_size, 4, 2)
        output['offset_2'] = weight_b.reshape(batch_size, 4, 2)

        # output['offset_1'] = torch.tensor([[10, 0], [-10, 0], [0, 0], [0, 0]]).reshape(1, 4, 2).float().cuda()  # for test

        output['points_2_pred'] = self.corners + output['offset_1']
        output['points_1_pred'] = self.corners + output['offset_2']
        homo_21 = dlt_homo(output['points_2_pred'], data_batch['points_1'])
        homo_12 = dlt_homo(output['points_1_pred'], data_batch['points_2'])

        # img1 warp to img2, img2_pred = img1_warp
        img1_warp = warp_image_from_H(homo_21, img1_full, *bhw)
        img2_warp = warp_image_from_H(homo_12, img2_full, *bhw)

        fea1_warp = warp_image_from_H(homo_21, img1_full_fea, *bhw)
        fea2_warp = warp_image_from_H(homo_12, img2_full_fea, *bhw)

        fea1_patch_warp, fea2_patch_warp = self.fea_extra(
            img1_warp
        ), self.fea_extra(img2_warp)

        output = {}
        output['H_flow'] = [homo_21, homo_12]
        # output['fea_full'] = [fea1_full, fea2_full]
        output["fea_warp"] = [fea1_warp, fea2_warp]
        output["fea_patch"] = [forward_fea, backward_fea]
        output["fea_patch_warp"] = [fea1_patch_warp, fea2_patch_warp]
        output["img_warp"] = [img1_warp, img2_warp]
        output['basis_weight'] = [weight_f, weight_b]
        return output


def Ms_Transformer(pretrained=False, **kwargs):
    """Constructs a Multi-scale Transformer model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OSNet(backbone=SwinTransformer, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

if __name__ == '__main__':
    from common import utils
    import ipdb
    ipdb.set_trace()
    px, py = 63, 10
    pw, ph = 512, 320
    pts = [[px, py], [px + pw, py], [px, py + ph], [px + pw, py + ph]]
    pts_1 = torch.from_numpy(np.array(pts)[np.newaxis]).float()
    pts_2 = torch.from_numpy(np.array(pts)[np.newaxis]).float()

    img = cv2.imread('model/test.jpg', 0)
    img_patch = img[py:py+ph, px:px+pw]

    input = torch.from_numpy(img[np.newaxis, np.newaxis].astype(np.float32))
    input_patch = torch.from_numpy(img_patch[np.newaxis, np.newaxis].astype(np.float32))
    data_dict = {}
    data_dict["imgs_gray_full"] = torch.cat([input, input], dim = 1).cuda()
    data_dict["imgs_gray_patch"] = torch.cat([input_patch, input_patch], dim = 1).cuda()
    data_dict["start"] = torch.tensor([px, py]).reshape(2, 1, 1).float()
    data_dict['points_1'] = pts_1.cuda()
    data_dict['points_2'] = pts_2.cuda()

    # net
    params = utils.Params('/data/xingchen/project/baseshomo/experiments/swin_model/params.json')
    net = OSNet(params, backbone=SwinTransformer, init_mode = 'swin')
    net.cuda()

    out = net(data_dict)
    print(out.keys())
