import cv2
import torch
import warnings
import numpy as np
import torch.nn as nn
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
        # The buffer is the same as the Parameter except that the gradient is not updated.
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
        if self.params.is_add_lrr_module:
            self.sp_layer3 = Subspace(256)
            self.sp_layer4 = Subspace(512)
        print(f'model_version: {params.model_version}')
        print(f'is_add_lrr_module: {params.is_add_lrr_module}')
        print(f'loss_func_type: {params.loss_func_type}')
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
        # parse input
        x1_patch, x2_patch = (
            input['imgs_gray_patch'][:, :1, ...],
            input['imgs_gray_patch'][:, 1:, ...],
        )
        x1_full, x2_full = (
            input["imgs_gray_full"][:, :1, ...],
            input["imgs_gray_full"][:, 1:, ...],
        )

        # net forward
        fea1_patch, fea2_patch = self.share_feature(x1_patch), self.share_feature(
            x2_patch
        )

        weight_f = self.nets_forward(torch.cat([fea1_patch, fea2_patch], dim=1))
        weight_b = self.nets_forward(torch.cat([fea2_patch, fea1_patch], dim=1))

        fea1_full, fea2_full = self.share_feature(x1_full), self.share_feature(x2_full)

        batch_size, _, h_patch, w_patch = x1_patch.size()
        bhw = (batch_size, h_patch, w_patch)

        output = {}

        if self.params.model_version == 'basis':
            bchw = [batch_size, 2, h_patch, w_patch]
            H_flow_f = (self.basis * weight_f).sum(1).reshape(*bchw)
            H_flow_b = (self.basis * weight_b).sum(1).reshape(*bchw)
            img1_warp = get_warp_flow(x1_full, H_flow_b, start=input['start'])
            img2_warp = get_warp_flow(x2_full, H_flow_f, start=input['start'])
            output['H_flow'] = [H_flow_f, H_flow_b]
        elif self.params.model_version == 'offset':
            output['offset_1'] = weight_f.reshape(batch_size, 4, 2)
            output['offset_2'] = weight_b.reshape(batch_size, 4, 2)
            output['points_2_pred'] = self.corners + output['offset_1']
            output['points_1_pred'] = self.corners + output['offset_2']
            homo_21 = dlt_homo(output['points_2_pred'], input['points_1'])
            homo_12 = dlt_homo(output['points_1_pred'], input['points_2'])
            img1_warp = warp_image_from_H(homo_21, x1_full, *bhw)
            img2_warp = warp_image_from_H(homo_12, x2_full, *bhw)
            output['H_flow'] = [homo_21, homo_12]
        else:
            raise

        fea1_patch_warp = self.share_feature(img1_warp)
        fea2_patch_warp = self.share_feature(img2_warp)

        output['fea_full'] = [fea1_full, fea2_full]
        output["fea_patch"] = [fea1_patch, fea2_patch]
        output["fea_patch_warp"] = [fea1_patch_warp, fea2_patch_warp]
        output["img_warp"] = [img1_warp, img2_warp]
        output['basis_weight'] = [weight_f, weight_b]
        return output


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
    # ipdb.set_trace()
    img = cv2.imread('dataset/test/10467_A.jpg', 0)
    img_patch = img[py : py + ph, px : px + pw]

    input = torch.from_numpy(img[np.newaxis, np.newaxis].astype(np.float32))
    input_patch = torch.from_numpy(img_patch[np.newaxis, np.newaxis].astype(np.float32))
    data_dict = {}
    data_dict["imgs_gray_full"] = torch.cat([input, input], dim=1).cuda()
    data_dict["imgs_gray_patch"] = torch.cat([input_patch, input_patch], dim=1).cuda()
    data_dict["start"] = torch.tensor([px, py]).reshape(2, 1, 1).float()
    data_dict['points_1'] = pts_1.cuda()
    data_dict['points_2'] = pts_2.cuda()

    # net
    crop_size = (ph, pw)
    with open('experiments/params.json') as f:
        params = json.load(f)
    params = EasyDict(params)
    params.crop_size = crop_size
    net = Net(params)
    net.cuda()

    out = net(data_dict)
    print(out.keys())
    res = out['img_warp'][0][0].cpu().detach().numpy().transpose((1, 2, 0))
    cv2.imwrite('dataset/test/test_res.jpg', res)


if __name__ == '__main__':

    util_test_net_forward()
