import logging
import os
import pickle
import random
import cv2
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class OffsetDataset(torch.utils.data.Dataset):
    def __init__(self, text_dir, mode='train'):
        self.keys = ["front", "back", "left", "right"]
        self.img_size = (width, height)
        self.data_list = []
        if isinstance(text_dir, list):
            list_files = text_dir
        else:
            list_files = glob.glob(os.path.join(text_dir, '*.txt'))
        for text_file in list_files:
            with open(text_file, 'r') as f:
                for line in f.readlines():
                    self.data_list.append(line.strip('\n'))
                    
        if mode == 'train':
            random.shuffle(self.data_list)
            
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        perfix = self.data_list[idx]
        imgs = []
        for key in self.keys:
            img = cv2.imread(perfix + f'_{key}.jpg')
            # img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis = 2)
        inputs = torch.from_numpy(imgs.transpose(2, 0, 1).astype(np.float32))
        inputs = (inputs / 255. - 0.5) * 2

        # gt
        gt = cv2.imread(perfix + '_avm.jpg')
        # gt = cv2.resize(gt, self.img_size, interpolation=cv2.INTER_LINEAR)
        gt = torch.from_numpy(gt.transpose(2, 0, 1).astype(np.float32))
        gt = (gt / 255. - 0.5) * 2
        
        # input resolution: 616, 880
        fl_pts, fr_pts = [(70, 80), (160, 180)], [(546, 80), (456, 180)]
        bl_pts, br_pts = [(70, 800), (160, 700)], [(456, 700), (546, 800)]
        pts_f = np.array([*fl_pts, *fr_pts], dtype=np.float32).reshape(1, 4, 2)
        pts_b = np.array([*bl_pts, *br_pts], dtype=np.float32).reshape(1, 4, 2)
        pts_l = np.array([*fl_pts, *bl_pts], dtype=np.float32).reshape(1, 4, 2)
        pts_r = np.array([*fr_pts, *br_pts], dtype=np.float32).reshape(1, 4, 2)

        data_dict = {}
        data_dict["inputs"] = inputs
        data_dict["gt"] = gt
        data_dict['points_f'] = torch.from_numpy(pts_f)
        data_dict['points_b'] = torch.from_numpy(pts_b)
        data_dict['points_l'] = torch.from_numpy(pts_l)
        data_dict['points_r'] = torch.from_numpy(pts_r)
        
        return data_dict

if __name__ == "__main__":
    import ipdb
    ipdb.set_trace()
    avm_dataset = OffsetDataset('/data/xingchen/dataset/AVM/b16_train/dataloader/train')
    inputs, gt = avm_dataset[0]
    print(inputs.size())
