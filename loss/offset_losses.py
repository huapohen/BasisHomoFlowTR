import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model import net


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda().eval()
        # self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, output):
        y, x = output['img_a_pred'], output['img_a_m']
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            ratio = output['ones_mask_w_avm_sum_ratio'][i]
            loss += self.weights[i] * (ratio * F.l1_loss(y_vgg[i], x_vgg[i]))
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


vgg_loss_func = VGGLoss()


def triplet_loss(a, p, n, margin=1.0, exp=1, reduce=False, size_average=False):

    triplet_loss = nn.TripletMarginLoss(
        margin=margin, p=exp, reduce=reduce, size_average=size_average
    )
    return triplet_loss(a, p, n)


def photo_loss_function(output, q, averge=True, ones_mask=False):
    diff = output['img_a_pred'] - output['img_ga_m']
    mean_I = torch.tensor([118.93, 113.97, 102.60]).reshape(1, 1, 1, 3).to(diff)
    std_I = torch.tensor([69.85, 68.81, 72.45]).reshape(1, 1, 1, 3).to(diff)
    diff = (diff.permute(0, 2, 3, 1) - mean_I) / std_I
    diff = diff.mean(dim=3)
    diff = (torch.abs(diff) + 0.01).pow(q)
    if ones_mask:
        for i in range(diff.shape[0]):
            diff[i] = diff[i] / output['ones_mask_w_avm_sum_ratio'][i]
    if averge:
        loss_mean = diff.mean()
    else:
        loss_mean = diff
    return loss_mean


def compute_losses(output, train_batch, params):
    losses = {}

    if params.loss_type == "basic":
        pass
    elif params.loss_type == "L2":
        loss = photo_loss_function(output, 2, ones_mask=True)
        losses["total"] = loss
    elif params.loss_type == "L1":
        loss = photo_loss_function(output, 1, ones_mask=True)
        losses["total"] = loss
    elif params.loss_type == "VGG":
        losses["l1"] = photo_loss_function(output, 1, ones_mask=True)
        losses["vgg"] = vgg_loss_func(output)
        losses["total"] = 0.1 * losses["l1"] + losses["vgg"]
    else:
        raise NotImplementedError

    return losses


def compute_eval_results(data_batch, output_batch, manager):

    imgs_full = data_batch["imgs_ori"]
    batch_size, _, grid_h, grid_w = imgs_full.shape

    bhw = (batch_size, grid_h, grid_w)
    homo_21, homo_12 = output_batch['H_flow']

    # img1 warp to img2, img2_pred = img1_warp
    img1_full_warp = net.warp_image_from_H(homo_21, imgs_full[:, :3, ...], *bhw)
    img2_full_warp = net.warp_image_from_H(homo_12, imgs_full[:, 3:, ...], *bhw)

    scale_x = grid_w / float(data_batch['imgs_gray_full'].shape[3])
    scale_y = grid_h / float(data_batch['imgs_gray_full'].shape[2])

    eval_results = {}
    eval_results["img1_full_warp"] = img1_full_warp

    return eval_results


if __name__ == "__main__":
    import ipdb

    ipdb.set_trace()
    input = torch.randn(1, 3, 448, 304)
    gt = torch.randn(1, 3, 448, 304)
    loss = vgg_loss_func(input.cuda(), gt.cuda())
    print(loss)
