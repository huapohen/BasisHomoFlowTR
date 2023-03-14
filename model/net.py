import ipdb
import copy
import torch
import warnings
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .swin_transformer import swin
from .u2net import u2net, u2netp
from .lightres import lightres
from .util import *
from .convnext import *
from .origin_net import BasesNet
import dataset.data_loader as datamaker


warnings.filterwarnings("ignore")

''' add this line, because GTX30 serials' default torch.matmul() on cuda is uncorrected '''
torch.backends.cuda.matmul.allow_tf32 = False


class Net(nn.Module):
    def __init__(self, params):

        super(Net, self).__init__()
        self.params = params
        self.inplanes = 64
        self.layers = [3, 4, 6, 3]
        self.basis_vector_num = 16
        self.crop_size = params.crop_size
        h, w = self.crop_size
        # 8,2,h,w --> 1, 8, 2*h*w
        self.basis = gen_basis(h, w).unsqueeze(0).reshape(1, 8, -1)
        self.basis2 = gen_basis(360, 640).unsqueeze(0).reshape(1, 8, -1)
        if params.share_module == 'origin':
            sf_params = [params.proj_in_chans, params.proj_out_chans]
            share_feature = ShareFeature_ori(*sf_params) 
            if params.is_share_feature:
                self.share_feature = nn.ModuleList(
                    [share_feature for _ in range(params.encoder_layers)]
                )
            else:
                self.share_feature = nn.ModuleList(
                    [copy.deepcopy(share_feature) for _ in range(params.encoder_layers)]
                )
        elif params.share_module == 'new':
            proj_params = (
                params.proj_in_chans,
                params.proj_out_chans,
                params.proj_hidden_chans,
                params.proj_kernel,
            )
            self.share_feature = ShareFeature_new(*proj_params)
        elif params.share_module == 'unet':
            self.share_feature = u2netp(params.proj_in_chans, fea_out_ch=1)
        else:
            raise

        in_chans = params.proj_out_chans * 2

        if params.backbone_type == 'swin':
            self.backbone = swin(params.expand_ratio, in_chans)
            params.feature_dim = 1024
        elif params.backbone_type == 'origin':
            params.feature_dim = int(
                params.basesnet_dim * 2**3 * params.basesnet_ratio
            )
            basesnet = BasesNet(
                new=params.basesnet_new,
                in_chans=in_chans,
                expand_ratio=params.basesnet_ratio,
                dim=params.basesnet_dim,
                layers=params.basesnet_layers,
                is_attn=params.basesnet_is_attn,
            )
            if params.is_share_backbone:
                self.backbone = nn.ModuleList(
                    [basesnet for _ in range(params.encoder_layers)]
                )
            else:
                self.backbone = nn.ModuleList(
                    [copy.deepcopy(basesnet) for _ in range(params.encoder_layers)]
                )
        elif params.backbone_type == 'light':
            self.backbone = lightres(params.expand_ratio, in_chans)
            params.feature_dim = int(params.expand_ratio * 1024)
        elif params.backbone_type == 'next':
            self.backbone = convnext_tiny(params.expand_ratio, in_chans)

        if params.is_tr_head:
            hidden_dim = params.hidden_dim
            if self.params.model_version == 'v1':
                self.query_embed = nn.Embedding(8, hidden_dim)
            elif self.params.model_version == 'v2':
                self.query_embed = nn.Embedding(9, hidden_dim)
            else:
                raise

            self.input_proj = nn.Conv2d(
                params.feature_dim * 2, hidden_dim, kernel_size=1
            )
            self.position_embedding = build_position_encoding(params)
            self.weight_embed = MLP(hidden_dim, hidden_dim, 1, 3)
            self.transformer = build_transformer(params)
            # self.transformer = build_deforamble_transformer(params)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def tr_head(self, x1, x2):
        src = torch.cat([x1, x2], dim=1)
        src = self.input_proj(src)
        b, _, h, w = src.shape
        mask = src.new_zeros(b, h, w, dtype=torch.uint8)
        pos = self.position_embedding([src, mask])
        mask = src.new_zeros(b, 1, h, w, dtype=torch.uint8)
        # Cameras Video Kernel Network
        query = self.query_embed.weight
        hs, memory = self.transformer(src, mask, query, pos)
        weight = self.weight_embed(hs)[-1]
        return weight
    
    def forward(self, input):
        params = self.params
        # test stage is different from train stage, especially for multi-stage net
        x1_patch = input[f'imgs_gray_patch'][:, :1, ...]
        x2_patch = input[f'imgs_gray_patch'][:, 1:, ...]
        x1_full = input["imgs_gray_full"][:, :1, ...]
        x2_full = input["imgs_gray_full"][:, 1:, ...]
        start = input['start']
        if params.is_resize_mode and params.dataset_mode == 'test':
            start = input['start_00']
        
        batch_size, _, h_patch, w_patch = x1_patch.shape
        bhw = (batch_size, h_patch, w_patch)
        b2hw = (batch_size, 2, h_patch, w_patch)

        output = {}
        output["img_warp"] = []
        output["H_flow"] = []

        if params.model_version == 'v1':
            if self.basis.device != x1_full.device:
                self.basis = self.basis.to(x1_full)
                self.basis2 = self.basis2.to(x1_full)
            # if self.params.is_tr_head:
            #     weight_f = self.tr_head(x1, x2)
            #     weight_b = self.tr_head(x2, x1)

            fea1_patch, fea2_patch = [], []
            for i in range(params.encoder_layers):
                fea1_patch.append(self.share_feature[i](x1_patch if i == 0 else img1_warp))
                fea2_patch.append(self.share_feature[i](x2_patch if i == 0 else img2_warp))
                info1 = self.backbone[i](torch.cat([fea1_patch[0], fea2_patch[i]], 1))
                info2 = self.backbone[i](torch.cat([fea2_patch[0], fea1_patch[i]], 1))
                H_flow_f = (self.basis * info1).sum(1).reshape(*b2hw)
                H_flow_b = (self.basis * info2).sum(1).reshape(*b2hw)
                img1_warp = get_warp_flow(x1_full, H_flow_b, input['start'])
                img2_warp = get_warp_flow(x2_full, H_flow_f, input['start'])
                # output["img_warp"].append([img1_warp, img2_warp])
                # output["H_flow"].append([H_flow_f, H_flow_b])
            output["img_warp"].append([img1_warp, img2_warp])
            output["H_flow"].append([H_flow_f, H_flow_b])
            # 是从原图一步跳过去的，没有每阶段断续连跳
   

        elif params.model_version == 'v2':
            for i in range(params.encoder_layers):
                info1 = self.tr_head(fea1_patch[0], fea2_patch[i]).reshape(-1, 3, 3)
                info2 = self.tr_head(fea2_patch[0], fea1_patch[i]).reshape(-1, 3, 3)
                H_flow_f = (self.basis * info1).sum(1).reshape(*b2hw)
                H_flow_b = (self.basis * info2).sum(1).reshape(*b2hw)
                img1_warp = warp_image_from_H(H_flow_b, x1_full, *bhw)
                img2_warp = warp_image_from_H(H_flow_f, x2_full, *bhw)
        else:
            raise
        # ipdb.set_trace()
        # output["img_warp"] = [img1_warp, img2_warp]
        # output["H_flow"] = [H_flow_f, H_flow_b]

        return output


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def fetch_net(params):

    if params.net_type == "basic":
        net = Net(params)
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return net
