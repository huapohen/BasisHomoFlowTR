import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from model import net
from common import utils
import json

torch.backends.cuda.matmul.allow_tf32 = False

class DynamicAVMs(object):
    def __init__(self, model_file, params):
        self.model = net.fetch_net(params).cuda()
        self.load_checkpoint(model_file)
        self.model.eval()
        self.mean_I = torch.tensor([118.93, 113.97, 102.60]).reshape(1, 3, 1, 1).cuda()
        self.std_I = torch.tensor([69.85, 68.81, 72.45]).reshape(1, 3, 1, 1).cuda()
        self.crop_size = params.crop_size

    def load_checkpoint(self, model_file):
        state = torch.load(model_file, map_location=torch.device('cpu'))
        state_dict = state["state_dict"]
        new_dict = {}
        for key, value in state_dict.items():
            new_dict[key[7:]] = state_dict[key]
        self.model.load_state_dict(new_dict)

    def match_imgs(self, imgs):
        '''
        warp img2 to img1
        '''
        img_h, img_w = imgs[0].shape[:2]
        inputs = [None]*len(imgs)
        for idx, img in enumerate(imgs):
            inputs[idx] = torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)).cuda()
            inputs[idx] = F.interpolate(inputs[idx], self.crop_size, mode = 'bilinear')
            inputs[idx] = (inputs[idx] - self.mean_I) / self.std_I
            inputs[idx] = inputs[idx].mean(dim = 1, keepdim = True)

        # net forward
        with torch.no_grad():
            homo12 = self.model.compute_homo(inputs[0], inputs[1])

        # get img1 to img2 homo
        homo12[:, :, 0] *= self.crop_size[1] / float(img_w)
        homo12[:, :, 1] *= self.crop_size[0] / float(img_h)
        homo12[:, 0, :] *= float(img_w) / self.crop_size[1]
        homo12[:, 1, :] *= float(img_h) / self.crop_size[0]

        # warp image2
        img2 = torch.from_numpy(imgs[1].transpose(2, 0, 1)[np.newaxis].astype(np.float32)).cuda()
        warp2 = net.warp_image_from_H(homo12, img2, 1, img_h, img_w)

        warp2 = warp2[0].data.cpu().numpy()
        warp2 = warp2.transpose(1, 2, 0).astype(np.uint8)

        return warp2, homo12

if __name__ == '__main__':
    import ipdb
    # ipdb.set_trace()
    id = 8 
    model_dir = '/home/data/lwb/experiments/baseshomo'
    sv_dir = 'dataset/test'
    params = utils.Params(f'{model_dir}/exp_{id}/params.json')
    davms = DynamicAVMs(f'{model_dir}/exp_{id}/model_latest.pth', params)
    data_dir = '/home/data/lwb/data/dybev/v7/test/generate'
    # path1 = 'front/20221205141455_front_00006_p0010_1.jpg'
    # path2 = 'right/20221205141455_right_00006_p0010_0.jpg'
    path1 = 'right/20221205141455_right_00338_p0011_1.jpg'
    path2 = 'back/20221205141455_back_00338_p0011_1.jpg'
    img1 = cv2.imread(os.path.join(data_dir, path1))
    img2 = cv2.imread(os.path.join(data_dir, path2))
    cv2.imwrite(f'{sv_dir}/img1.jpg', img1)
    cv2.imwrite(f'{sv_dir}/img2.jpg', img2)
    warp2, homo12 = davms.match_imgs([img1, img2])
    cv2.imwrite(f'{sv_dir}/warp.jpg', warp2)
    utils.create_gif([img1, img2], f'{sv_dir}/bev_ori.gif')
    utils.create_gif([img1, warp2], f'{sv_dir}/bev_warp.gif')
