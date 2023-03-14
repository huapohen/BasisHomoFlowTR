import torch
import warnings
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class BasesNet(nn.Module):
    def __init__(
        self,
        new=True,
        in_chans=2,
        expand_ratio=1.0,
        dim=64,
        layers=[3, 4, 6, 3],
        is_attn=False,
    ):
        super().__init__()
        self.new = new
        self.is_attn = is_attn
        dim = int(expand_ratio * dim)
        self.inplanes = dim
        # [64, 128, 256, 512]
        ds = [dim * 2**i for i in range(4)]
        ss = [1, 2, 2, 2]
        self.conv1 = nn.Conv2d(
            in_chans, dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.block = BasicBlock
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList(
            [self._make_layer(self.block, ds[i], layers[i], ss[i]) for i in range(4)]
        )
        if self.new:
            self.attn = nn.ModuleList([SelfAttention(d) for d in ds])
            ks = [9, 5, 5, 1]
            ss = [8, 4, 2, 1]
            ps = [4, 2, 2, 0]
            ouc = ds[-1]
            self.connect = nn.ModuleList(
                [BaseConv(ds[i], ouc, ks[i], ss[i], ps[i]) for i in range(4)]
            )
            self.layer5 = nn.Sequential(BaseConv(4 * ouc, ouc, 1))

        self.proj_out = nn.Sequential(
            nn.Conv2d(ds[-1], 8, 1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c 1 1 -> b c 1'),
        )

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        xs = []
        for i in range(4):
            x = self.layers[i](x)
            if self.is_attn and i >= 2:
                x = self.attn[i](x)
            xs.append(x)
        if self.new:
            feature = []
            for i in range(4):
                feature.append(self.connect[i](xs[i]))
            y = self.layer5(torch.cat(feature, 1))
            return [y]
        # return [xs[-1]]
        return self.proj_out(xs[-1])


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        # self.groups = groups = inplanes // 2
        self.groups = groups = 1
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, 1, groups, False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, 1, groups, False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = rearrange(out, 'b (c g) h w -> b (g c) h w', g=self.groups)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = rearrange(out, 'b (g c) h w -> b (c g) h w', g=self.groups)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BaseConv(nn.Module):
    def __init__(self, inc, ouc, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, ouc, k, s, p, bias=False),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * C * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(
            m_batchsize, C, width, height
        )  # B * C * W * H

        out = self.gamma * self_attetion + x
        return out
