import logging
import os
import pickle
import random
import cv2
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_logger = logging.getLogger(__name__)

class AVMDataset(torch.utils.data.Dataset):
    def __init__(self, text_dir, width = 304, height = 448, shuffle = True):
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
        if shuffle:
            random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        perfix = self.data_list[idx]
        imgs = []
        for key in self.keys:
            img = cv2.imread(perfix + f'_{key}.jpg')
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis = 2)
        inputs = torch.from_numpy(imgs.transpose(2, 0, 1).astype(np.float32))
        inputs = (inputs / 255. - 0.5) * 2

        # gt
        gt = cv2.imread(perfix + '_avm.jpg')
        gt = cv2.resize(gt, self.img_size, interpolation=cv2.INTER_LINEAR)
        gt = torch.from_numpy(gt.transpose(2, 0, 1).astype(np.float32))
        gt = (gt / 255. - 0.5) * 2

        data_dict = {}
        data_dict["inputs"] = inputs
        data_dict["gt"] = gt
        return data_dict

if __name__ == "__main__":
    import ipdb
    ipdb.set_trace()
    avm_dataset = AVMDataset('/data/xingchen/dataset/AVM/b16_train/dataloader/train')
    inputs, gt = avm_dataset[0]
    print(inputs.size())
