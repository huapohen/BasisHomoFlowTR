import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model import net

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids = None):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda().eval()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
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


def photo_loss_function(diff, q, averge=True):
    diff = (torch.abs(diff) + 0.01).pow(q)
    if averge:
        loss_mean = diff.mean()
    else:
        loss_mean = diff.sum()
    return loss_mean


def geometricDistance(correspondence, flow):
    flow = flow.permute(1, 2, 0).cpu().detach().numpy()

    p1 = correspondence[0]  # 0
    p2 = correspondence[1]  # 1

    if isinstance(correspondence[1][0], float):
        result = p2 - (p1 - flow[int(p1[1]), int(p1[0])])
        error = np.linalg.norm(result)
    else:
        result = [p2 - (p1 - flow[p1[1], p1[0]]), p1 - (p2 - flow[p2[1], p2[0]])]
        error = min(np.linalg.norm(result[0]), np.linalg.norm(result[1]))

    return error

def geometricDistance_v2(inp, out, scale_x = 1.0, scale_y = 1.0):
    ones = torch.ones_like(inp['points_1_all'])
    pts_1 = torch.cat([inp['points_1_all'], ones[:, :, :1]], -1)
    pts_2 = inp['points_2_all']
    homo_21_inv = torch.inverse(out['H_flow'][0])
    homo_12_inv = torch.inverse(out['H_flow'][1])

    def calc_pts_err(homo, pts1, pts2, scale_x, scale_y):
        pts = pts1.permute(0, 2, 1)
        pts[:, 0] = pts[:, 0] / scale_x
        pts[:, 1] = pts[:, 1] / scale_y
        warp_pts = torch.einsum('bnc,bck->bnk', homo, pts)
        warp_pts = warp_pts.permute(0, 2, 1)
        warp_pts = warp_pts / warp_pts[:, :, 2:]  # /z, normalilzation
        warp_pts[:, :, 0] *= scale_x
        warp_pts[:, :, 1] *= scale_y

        diff = torch.linalg.norm(warp_pts[:, :, :2] - pts2, dim = 2)
        return diff.mean(1).data.cpu().numpy()

    err_1 = calc_pts_err(homo_21_inv, pts_1, pts_2, scale_x, scale_y)

    return err_1

def compute_losses(output, train_batch, params):
    losses = {}

    # compute losses
    if params.loss_type == "basic":
        imgs_patch = train_batch['imgs_gray_patch']

        # start = train_batch['start']
        # H_flow_f, H_flow_b = output['H_flow']
        # fea1_full, fea2_full = output["fea_full"]
        fea1_patch, fea2_patch = output["fea_patch"]
        img1_warp, img2_warp = output["img_warp"]
        fea1_warp, fea2_warp = output['fea_warp']
        fea1_patch_warp, fea2_patch_warp = output["fea_patch_warp"]

        batch_size, _, h_patch, w_patch = imgs_patch.size()

        # fea2_warp = net.get_warp_flow(fea2_full, H_flow_f, start=start)
        # fea1_warp = net.get_warp_flow(fea1_full, H_flow_b, start=start)

        im_diff_fw = imgs_patch[:, :1, ...] - img2_warp
        im_diff_bw = imgs_patch[:, 1:, ...] - img1_warp

        fea_diff_fw = fea1_warp - fea1_patch_warp
        fea_diff_bw = fea2_warp - fea2_patch_warp

        # loss
        losses["photo_loss_l1"] = photo_loss_function(
            diff=im_diff_fw, q=1, averge=True
        ) + photo_loss_function(diff=im_diff_bw, q=1, averge=True)

        losses["fea_loss_l1"] = photo_loss_function(
            diff=fea_diff_fw, q=1, averge=True
        ) + photo_loss_function(diff=fea_diff_bw, q=1, averge=True)

        losses["triplet_loss"] = (
            triplet_loss(fea1_patch, fea2_warp, fea2_patch).mean()
            + triplet_loss(fea2_patch, fea1_warp, fea1_patch).mean()
        )

        # loss toal: backward needed
        losses["total"] = (
            losses["triplet_loss"] + params.weight_fil * losses["fea_loss_l1"] + 0.7 * losses["photo_loss_l1"]
        )

        # losses["total"] = (
            # losses["triplet_loss"] + params.weight_fil * losses["fea_loss_l1"] + 0.2 * losses["photo_loss_l1"]
        # )
    
    elif params.loss_type == "L2":
        loss = F.mse_loss(output, train_batch["gt"])
        losses["total"] = loss
    elif params.loss_type == "L1":
        loss = F.l1_loss(output, train_batch["gt"])
        losses["total"] = loss
    elif params.loss_type == "VGG":
        losses["l1"] = F.l1_loss(output, train_batch["gt"])
        losses["vgg"] = vgg_loss_func(output, train_batch["gt"])
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
    errs = geometricDistance_v2(data_batch, output_batch, scale_x, scale_y)

    eval_results = {}
    eval_results["img1_full_warp"] = img1_full_warp
    eval_results["errs"] = errs

    return eval_results

if __name__ == "__main__":
    import ipdb
    ipdb.set_trace()
    input = torch.randn(1, 3, 448, 304)
    gt = torch.randn(1, 3, 448, 304)
    loss = vgg_loss_func(input.cuda(), gt.cuda())
    print(loss)
