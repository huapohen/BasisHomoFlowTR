import logging
import os
import pickle
import random
import cv2
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ipdb import set_trace as ip


class OffsetDataset(torch.utils.data.Dataset):
    def __init__(self, text_dir, mode='train'):

        self.data_list = []
        if isinstance(text_dir, list):
            list_files = text_dir
        else:
            list_files = glob.glob(text_dir + '.txt')
        for text_file in list_files:
            with open(text_file, 'r') as f:
                for line in f.readlines():
                    if 'avm' in line:
                        self.data_list.append(line.strip('\n'))
        self.data_dir = text_dir
        if mode == 'train':
            random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        perfix = '_'.join(self.data_list[idx].split('_')[:2])

        fl_pts, fr_pts = [(70, 80), (160, 180)], [(546, 80), (456, 180)]
        bl_pts, br_pts = [(70, 800), (160, 700)], [(456, 700), (546, 800)]
        pts = {
            'f': [*fl_pts, *fr_pts],
            'b': [*bl_pts, *br_pts],
            'l': [*fl_pts, *bl_pts],
            'r': [*fr_pts, *br_pts],
        }

        data_dict = {}

        for k in ['front', 'back', 'left', 'right', 'avm']:
            img = cv2.imread(os.path.join(self.data_dir, perfix + f'_{k}.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            img = (img / 255.0 - 0.5) * 2
            img = torch.from_numpy(img.astype(np.float32))
            if k != 'avm':
                data_dict[f'img_{k[0]}'] = img
                pt = np.array(pts[k[0]], dtype=np.float32).reshape(4, 2)
                data_dict[f'points_{k[0]}'] = torch.from_numpy(pt)

        return data_dict


if __name__ == "__main__":
    import ipdb

    # ipdb.set_trace()
    avm_dataset = OffsetDataset('/home/data/lwb/data/dybev/v10_avm/train')
    for k, v in avm_dataset[0].items():
        print(k, v.shape)
