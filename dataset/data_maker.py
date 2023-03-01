import os
import sys
import cv2
import json
import ipdb
import shutil
import imageio
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

try:
    from .avm.overlap import extract_overlap_hw
except:
    from avm.overlap import extract_overlap_hw


class DataMakerOverlap:
    def __init__(self, mode='train', source='bev'):
        self.data_root_dir = '/home/data/lwb/data/dybev'
        self.src_data_dir = self.data_root_dir + f'/v8/{mode}'
        self.tgt_data_dir = self.data_root_dir + f'/v9/{mode}'
        self.src_bev_dir = os.path.join(self.src_data_dir, source)
        self.tgt_bev_dir = os.path.join(self.tgt_data_dir, source)
        self.name_list = sorted(os.listdir(self.src_bev_dir + '/front'))
        self.pts_bev_fblr_path = self.src_data_dir + '/detected_points.json'
        self.wh_bev_fblr = EasyDict(
            {
                "front": [1078, 336],
                "back": [1078, 336],
                "left": [1172, 439],
                "right": [1172, 439],
            }
        )
        with open(self.pts_bev_fblr_path, 'r') as f:
            self.pts_bev_fblr = EasyDict(json.load(f)['corner_points'])
        self.cam_list = ['front', 'back', 'left', 'right']
        self.hw_overlap_fblr = extract_overlap_hw()

    def make_pair(self):
        os.makedirs(self.tgt_bev_dir, exist_ok=True)
        txt_sv_path = self.tgt_bev_dir + '/name_list.txt'
        if os.path.exists(txt_sv_path):
            os.remove(txt_sv_path)
        with open(txt_sv_path, 'a+') as f:
            for name in self.name_list:
                f.write(name.split('.jpg')[0] + '\n')
        # return
        for cam in self.cam_list:
            with tqdm(total=len(self.name_list)) as t:
                for name in self.name_list:
                    name = name.replace('front', cam)
                    path = f'{self.src_bev_dir}/{cam}/{name}'
                    img = cv2.imread(path)
                    hw_info = self.hw_overlap_fblr[cam]
                    r_info = self.hw_overlap_fblr['instruction']['rotate']
                    sv_dir = self.tgt_bev_dir + f'/{cam}'
                    os.makedirs(sv_dir, exist_ok=True)
                    sv_name = sv_dir + f'/{name}'
                    for i in range(2):
                        h1, h2, w1, w2 = hw_info[i]
                        # ipdb.set_trace()
                        angle = hw_info[2]
                        subimg = img[h1:h2, w1:w2]
                        sv_path = sv_name.replace('.jpg', f'_{i}.jpg')
                        if angle != 0:
                            subimg = cv2.rotate(subimg, eval(r_info[str(angle)]))
                        cv2.imwrite(sv_path, subimg)
                    t.set_description(desc=f'{cam} >>> ')
                    t.update()
        pass

    def make_pair_v0(self):
        '''ori_vs_ori'''
        pass

    def make_pair_v1(self):
        '''pert_vs_ori'''
        pass

    def make_pair_v1(self):
        '''pert_vs_pert'''
        pass


if __name__ == '__main__':

    mode_list = ['train', 'test']
    source_list = ['bev', 'generate']
    # source_list = ['generate']
    # source_list = ['bev']

    for mode in mode_list:
        for source in source_list:
            print(mode, source)
            data_handle = DataMakerOverlap(mode, source)
            data_handle.make_pair()
