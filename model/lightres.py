import sys
import time
import thop
import torch
import torch.nn as nn

__all__ = ['lightnet']

ck = 9

NET_CONFIG = {  # k, inc, ouc, s, act, res
    "blocks2": [[ck, 32, 64, 1, 0, 0]],  # 112
    "blocks3": [[ck, 64, 128, 2, 1, 0], [ck, 128, 128, 1, 0, 1]],  # 56
    "blocks4": [[ck, 128, 256, 2, 1, 0], [ck, 256, 256, 1, 0, 1]],  # 28
    "blocks5": [
        [ck, 256, 512, 2, 1, 0],
        [ck, 512, 512, 1, 1, 1],
        [ck, 512, 512, 1, 1, 1],
        [ck, 512, 512, 1, 1, 1],
        [ck, 512, 512, 1, 1, 1],
        [ck, 512, 512, 1, 0, 1],
    ],  # 14
    "blocks6": [[ck, 512, 1024, 2, 1, 0], [ck, 1024, 1024, 1, 0, 0]],  # 7
}


Act = nn.ReLU


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNorm(nn.Module):
    def __init__(self, inc, ouc, kernel=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2d(inc, ouc, kernel, stride, padding, 1, groups, False)
        # self.norm = nn.BatchNorm2d(ouc)
        self.norm = LayerNorm(ouc)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self, inc, ouc, kernel=3, stride=1, need_act=1, residual=0):
        super().__init__()
        self.dwconv = ConvNorm(inc, inc, kernel, stride, inc)
        self.pwconv = ConvNorm(inc, ouc, 1)
        # self.act = Act(True)
        self.act = Act()
        self.residual = residual
        self.need_act = need_act

    def forward(self, x):
        if self.residual == 1:
            shortcut = x
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pwconv(x)
        if self.residual:
            x += shortcut
        if self.need_act == 1:
            x = self.act(x)
        return x


class LightNet(nn.Module):
    def __init__(
        self, r=1.0, with_act=True, in_chans=1, need_residual=True, m=make_divisible
    ):
        super().__init__()

        self.conv1 = ConvNorm(in_chans, m(32 * r), 3, 2)

        for blk in range(2, 7):
            blocks = nn.Sequential(
                *[
                    DepthwiseSeparable(
                        m(inc * r), m(ouc * r), k, s, a, p if need_residual else 0
                    )
                    for i, (k, inc, ouc, s, a, p) in enumerate(
                        NET_CONFIG[f"blocks{blk}"]
                    )
                ]
            )
            setattr(self, f"blocks{blk}", blocks)

        self.act = Act()
        self.with_act = with_act

    def forward(self, x):
        out = []
        x = self.conv1(x)
        for blk in range(2, 7):
            blocks = getattr(self, f"blocks{blk}")
            x = blocks(self.act(x))
            if blk - 1 >= 3:
                out.append(x)
        if self.with_act:
            out = [self.act(i) for i in out]
        return out


def lightres(width, in_chans=1, with_act=True, need_residual=True, **kwargs):
    backbone = LightNet(width, with_act, in_chans, need_residual, **kwargs)
    return backbone


if __name__ == '__main__':
    import thop
    import time
    import ipdb

    r = 0.5
    inc = 1
    net = lightres(r, in_chans=inc, need_residual=True)
    input_shape = (inc, 320, 576)
    x = torch.randn(1, *input_shape)
    flops, params = thop.profile(net, inputs=(x,), verbose=False)
    net.eval()
    output = net(x)
    # ipdb.set_trace()
    y = output[-1]
    split_line = '=' * 30
    print(
        f'''
            {net.__class__}
            {split_line}
            Input  shape: {tuple(x.shape[1:])}
            Output shape: {tuple(y.shape[1:])}
            Flops: {flops / 10 ** 6:.3f} M
            Params: {params / 10 ** 3:.3f} K
            {split_line}'''
    )
