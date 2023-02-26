import cv2
import torch
import numpy as np
from easydict import EasyDict

try:
    from model.net import Net
except:
    from .model.net import Net


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

    img = cv2.imread('dataset/AVM/10467_A.jpg', 0)
    img_patch = img[py : py + ph, px : px + pw]

    input = torch.from_numpy(img[np.newaxis, np.newaxis].astype(np.float32))
    input_patch = torch.from_numpy(img_patch[np.newaxis, np.newaxis].astype(np.float32))
    data_dict = {}
    data_dict["imgs_gray_full"] = torch.cat([input, input], dim=1).cuda()
    data_dict["imgs_gray_patch"] = torch.cat([input_patch, input_patch], dim=1).cuda()
    data_dict["imgs_full_rgb"] = data_dict["imgs_gray_full"]
    data_dict['img1_full_rgb'] = input.cuda()
    data_dict['img2_full_rgb'] = input.cuda()
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
    cv2.imwrite('dataset/AVM/test_res.jpg', res)


if __name__ == '__main__':

    util_test_net_forward()
