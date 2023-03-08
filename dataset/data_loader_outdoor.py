import os
import cv2
import ipdb
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class HomoData(Dataset):
    def __init__(self, params, mode='train'):
        self.params = params
        self.mode = mode
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.crop_size = params.crop_size_outdoor
        self.rho = params.rho_dybev
        self.normalize = True
        self.gray = True
        self.horizontal_flip_aug = True if mode == 'train' else False
        txt_list = []
        
        base_path = '/home/data/lwb/data/dybev/'
        # self.data_dir = base_path + '/tmp'
        self.data_dir = base_path + '/b16'
        path = os.path.join(self.data_dir, f'{mode}.txt')
        
        self.data_all = open(path, 'r').readlines()
        total_sample = len(self.data_all)

        random.seed(params.seed)
        np.random.seed(params.seed)
        random.shuffle(self.data_all)

        num = int(total_sample * params.train_data_ratio)
        self.data_infor = self.data_all[:num]

        self.sample_number = {}
        set_name = params.train_data_dir.split(os.sep)[-1]
        self.sample_number[set_name] = {
            'ratio': params.train_data_ratio,
            "samples": len(self.data_infor),
        }
        self.sample_number["total_samples"] = len(self.data_infor)
        # ipdb.set_trace()

    def __len__(self):
        return len(self.data_infor)

    def __getitem__(self, idx):
        '''
        debug dataloader: set num_workers=0
        '''
        img_names = self.data_infor[idx]
        img_names = img_names.split(' ')
        
        ph, pw = self.crop_size
        patch_list, full_list = [], []
        pts_1_list, pts_2_list = [], []
        imgs_ori_list = []
        
        for i in range(int(len(img_names) / 2)):
            img1 = cv2.imread(f'{self.data_dir}/{img_names[i * 2]}')
            img2 = cv2.imread(f'{self.data_dir}/{img_names[i * 2 + 1].rsplit()[0]}')
            # ipdb.set_trace()
            img1_full, img2_full, img1_patch, img2_patch, px, py = self.data_aug(img1, img2)
            patch_list += [img1_patch, img2_patch]
            full_list += [img1_full, img2_full]
            imgs_ori_list += [img1, img2]
            pts = [[px, py], [px + pw, py], [px, py + ph], [px + pw, py + ph]]
            pts_1_list.append(torch.from_numpy(np.array(pts)).float())
            pts_2_list.append(torch.from_numpy(np.array(pts)).float())
        
        patch = np.concatenate(patch_list, axis=2)
        full = np.concatenate(full_list, axis=2)
        imgs_ori = np.concatenate(imgs_ori_list, axis=2)
        
        data_dict = {}
        data_dict["imgs_ori"] = torch.tensor(imgs_ori).permute(2, 0, 1).float()
        data_dict["imgs_gray_full"] = torch.tensor(full).permute(2, 0, 1).float()
        data_dict["imgs_gray_patch"] = torch.tensor(patch).permute(2, 0, 1).float()
        data_dict['points_1'] = torch.cat(pts_1_list, dim=0)
        data_dict['points_2'] = torch.cat(pts_2_list, dim=0)
        frames_name = img_names[0].split(os.sep)[1].split('.')[0]
        frames_name += '_vs_'+ '_'.join(img_names[1].split('.')[0].split('_')[1:])
        # ipdb.set_trace()
        data_dict['frames_name'] = frames_name
        
        # ipdb.set_trace()
        return data_dict

    def data_aug(self, img1, img2):
        def random_crop(img1, img2):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.crop_size
            x = np.random.randint(self.rho, width - self.rho - patch_size_w)
            y = np.random.randint(self.rho, height - self.rho - patch_size_h)
            img1_patch = img1[y : y + patch_size_h, x : x + patch_size_w, :]
            img2_patch = img2[y : y + patch_size_h, x : x + patch_size_w, :]
            return img1_patch, img2_patch, x, y
        
        def resize(img1, img2):
            ch, cw = self.crop_size
            if img1.shape[0] != ch or img1.shape[1] != cw:
                img1_rs = cv2.resize(img1, (cw, ch))
                img2_rs = cv2.resize(img2, (cw, ch))
            return img1_rs, img2_rs, 0, 0

        if self.horizontal_flip_aug and random.random() <= 0.5:
            img1 = np.flip(img1, 1)
            img2 = np.flip(img2, 1)

        if self.normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        aug_func = random_crop if self.mode == 'train' else resize
        img1_aug, img2_aug, px, py = aug_func(img1, img2)
        
        if self.gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)
            img1_aug = np.mean(img1_aug, axis=2, keepdims=True)
            img2_aug = np.mean(img2_aug, axis=2, keepdims=True)

        return img1, img2, img1_aug, img2_aug, px, py


def fetch_dataloader(params):

    dataloaders = {}
    
    if params.dataset_type in ['basic', 'train']:
        train_ds = HomoData(params, 'train')
        
        dl = DataLoader(
            train_ds,
            batch_size=params.train_batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=params.cuda,
            drop_last=True,
            # prefetch_factor=3,
        )
        dl.sample_number = train_ds.sample_number
        dataloaders["train"] = dl

    if params.eval_type in ['val', 'test']:
        test_ds = HomoData(params, 'test')
        dl = DataLoader(
            test_ds,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=params.cuda,
            # prefetch_factor=3,
        )
        dl.sample_number = test_ds.sample_number
        dataloaders[params.eval_type] = dl

    return dataloaders
