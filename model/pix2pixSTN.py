import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Pix2PixSTN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9, norm='instance', 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Pix2PixSTN, self).__init__() 
        activation = nn.ReLU(True)
        norm_layer = get_norm_layer(norm)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                      STNSplitLayer(ngf * mult),
                      ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                       # nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       nn.Upsample(scale_factor=2, mode="bilinear"),
                       nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding = 1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class STNSplitLayer(nn.Module):
    def __init__(self, in_ch, out_ch = 80):
        super(STNSplitLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_ch, out_features=out_ch),
            nn.ReLU(),
            nn.Linear(in_features=out_ch, out_features=24),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))
        bias = bias.expand(4, -1).reshape(-1)

        nn.init.constant_(self.fc[2].weight, 0)
        self.fc[2].bias.data.copy_(bias)

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        x = F.adaptive_avg_pool2d(img, (1, 1))
        theta = self.fc(x.view(batch_size, -1)).view(batch_size, 4, 2, 3)
        thetas = theta.split(1, dim = 1)
        imgs = img.split(img.shape[1]//4, dim = 1)

        img_trans = []
        for theta_, img_ in zip(thetas, imgs):
            grid = F.affine_grid(theta_.squeeze(1), torch.Size((batch_size, img_.shape[1], img_.shape[2], img_.shape[3])))
            img_transform = F.grid_sample(img_, grid)
            img_trans.append(img_transform)
        img_trans = torch.cat(img_trans, dim = 1)

        return img_trans

class STNLayer(nn.Module):
    def __init__(self, in_ch, out_ch = 80):
        super(STNLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_ch, out_features=out_ch),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(in_features=out_ch, out_features=6),
            # nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        nn.init.constant_(self.fc[2].weight, 0)
        self.fc[2].bias.data.copy_(bias)

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        x = F.adaptive_avg_pool2d(img, (1, 1))
        theta = self.fc(x.view(batch_size, -1)).view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, img.shape[1], img.shape[2], img.shape[3])))
        img_transform = F.grid_sample(img, grid)

        return img_transform

if __name__ == "__main__":
    import ipdb
    ipdb.set_trace()

    # net = Pix2PixSTN(3, 3)
    # input = torch.randn(1, 3, 448, 304)
    # out = net(input)
    # print(out.size())

    input = torch.randn(1, 256, 32, 32)
    stn = STNLayer(256)
    out = stn(input)
    print(out.size())
