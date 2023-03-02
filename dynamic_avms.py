import os
import cv2
import json
import ipdb
import torch
import shutil
import numpy as np
import torch.nn.functional as F
from model import net
from common import utils

torch.backends.cuda.matmul.allow_tf32 = False


class DynamicAVMs(object):
    def __init__(self, model_file, params):
        self.model = net.fetch_net(params).cuda()
        self.load_checkpoint(model_file)
        self.model.eval()
        self.mean_I = torch.tensor([118.93, 113.97, 102.60]).reshape(1, 3, 1, 1).cuda()
        self.std_I = torch.tensor([69.85, 68.81, 72.45]).reshape(1, 3, 1, 1).cuda()
        self.crop_size = params.crop_size_dybev

    def load_checkpoint(self, model_file):
        state = torch.load(model_file, map_location=torch.device('cpu'))
        state_dict = state["state_dict"]
        new_dict = {}
        for key, value in state_dict.items():
            # module.
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
    
def single_img_inference(id):
    # ipdb.set_trace()
    model_dir = '/home/data/lwb/experiments/baseshomo'
    sv_dir = model_dir + f'/vis/exp_{id}'
    if os.path.exists(sv_dir):
        shutil.rmtree(sv_dir)
    os.makedirs(sv_dir, exist_ok=True)
    params = utils.Params(f'{model_dir}/exp_{id}/params.json')
    davms = DynamicAVMs(f'{model_dir}/exp_{id}/model_latest.pth', params)
    data_dir = {}
    set_name = 'v7' if id == 6 else 'v9'
    data_dir['test'] = f'/home/data/lwb/data/dybev/{set_name}/test/generate'
    data_dir['train'] = data_dir['test'].replace('test', 'train')
    path = {}
    path1 = 'front/20221205141455_front_00006_p0010_1.jpg'
    path2 = 'right/20221205141455_right_00006_p0010_0.jpg'
    path['fr'] = [path1, path2]
    path1 = 'right/20221205141455_right_00338_p0011_1.jpg'
    path2 = 'back/20221205141455_back_00338_p0011_1.jpg'
    path['rb'] = [path1, path2]
    path1 = 'left/20221205140548_left_00296_p0000_0.jpg'
    path2 = 'front/20221205140548_front_00296_p0000_0.jpg'
    path['lf'] = [path1, path2]
    for k, v in path.items():
        mode = 'train' if k == 'lf' else 'test'
        img1 = cv2.imread(os.path.join(data_dir[mode], path[k][0]))
        img2 = cv2.imread(os.path.join(data_dir[mode], path[k][1]))
        # cv2.imwrite(f'{sv_dir}/{id}_{k}_img1.jpg', img1)
        # cv2.imwrite(f'{sv_dir}/{id}_{k}_img2.jpg', img2)
        warp2, homo12 = davms.match_imgs([img1, img2])
        # cv2.imwrite(f'{sv_dir}/{id}_{k}_warp.jpg', warp2)
        utils.create_gif([img1, img2], f'{sv_dir}/{id}_{k}_ori.gif')
        utils.create_gif([img1, warp2], f'{sv_dir}/{id}_{k}_warp.gif')


        
def fblr_img_inference(id, loss_mode='all_fblr'):
    sequence = {
        'all_fblr': [0,1, 2,3, 4,5, 6,7],
        'front': [0,1,5,7],
        'back': [2,3,4,6],
        'left': [4,5,1,3],
        'right': [6,7,0,2],
    }
    seq = sequence[loss_mode]
    camera_list = ['front', 'back', 'left', 'right']
    # ipdb.set_trace()
    model_dir = '/home/data/lwb/experiments/baseshomo'
    sv_dir = model_dir + f'/vis/exp_{id}'
    if os.path.exists(sv_dir):
        shutil.rmtree(sv_dir)
    os.makedirs(sv_dir, exist_ok=True)
    params = utils.Params(f'{model_dir}/exp_{id}/params.json')
    davms = DynamicAVMs(f'{model_dir}/exp_{id}/model_latest.pth', params)
    data_dir = {}
    set_name = 'v7' if id == 6 else 'v9'
    data_dir = f'/home/data/lwb/data/dybev/{set_name}/test/generate'
    pair_path = data_dir + '/pair/all_fblr_pair_names_list.txt'
    with open(pair_path, 'r') as f:
        paths = f.readlines()
    for k, v in enumerate(paths):
        path_pair = paths[k].split(' ')
        path_pair = [os.path.join(data_dir, p.rsplit()[0]) for p in path_pair]
        imgs = [cv2.imread(path) for path in path_pair]
        name = path_pair[0].split(os.sep)[-1].split('.')[0]
        # ipdb.set_trace()
        warp_list, img_list = [], []
        for i in range(4):
            img1, img2 = imgs[i*2], imgs[i*2+1]
            name_ori = name.replace('front', camera_list[i])
            sv_path = f'{sv_dir}/{id}_{k}_ori_{i}_{name_ori}.gif'
            utils.create_gif([img1, img2], sv_path)
            warp2 = davms.match_imgs([img1, img2])[0]
            warp1 = davms.match_imgs([img2, img1])[0]
            warp_list.append(warp2)
            warp_list.append(warp1)
            img_list.append(img1)
            img_list.append(img2)
        for i in seq:
            warp = warp_list[i]
            img = img_list[i]
            name_warp = name.replace('front', camera_list[int(i/2)])
            sv_path = f'{sv_dir}/{id}_{k}_warp_{i}_{name_warp}.gif'
            utils.create_gif([img, warp], sv_path)


if __name__ == '__main__':
    
    loss_mode = ['all_fblr', 'all_fblr', 'front', 'back', 'left', 'right']
    
    for exp_id in range(6, 12):
        # single_img_inference(exp_id)
        fblr_img_inference(exp_id, loss_mode[exp_id-6])
