import logging
import os
import pickle
import random
import cv2
import ipdb

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_logger = logging.getLogger(__name__)


class HomoTrainData(Dataset):
    def __init__(self, params):
        self.params = params

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.crop_size = params.crop_size

        self.rho = params.rho
        self.normalize = True
        self.horizontal_flip_aug = True

        self.data_dir = os.path.join(params.train_data_dir, 'train')
        self.list_path = os.path.join(params.train_data_dir, "train_list.txt")
        self.data_infor = open(self.list_path, 'r').readlines()
        total_sample = len(self.data_infor)
        sample_num = int(total_sample * params.train_ratio)

        # ipdb.set_trace()
        self.seed = 0
        random.seed(self.seed)
        random.shuffle(self.data_infor)
        self.data_infor = self.data_infor[:sample_num]

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):

        # img loading
        img_names = self.data_infor[idx]
        img_names = img_names.split(' ')
        img1 = cv2.imread(
            os.path.join(self.data_dir, img_names[0])
        )  # Read image according to data list
        img2 = cv2.imread(os.path.join(self.data_dir, img_names[1][:-1]))

        img1 = cv2.resize(img1, (640, 360), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (640, 360), interpolation=cv2.INTER_AREA)

        # ipdb.set_trace()
        # img aug
        img1, img2, img1_patch, img2_patch, start = self.data_aug(
            img1,
            img2,
            normalize=self.normalize,
            horizontal_flip=self.horizontal_flip_aug,
        )
        # array to tensor
        imgs_gray_full = (
            torch.tensor(np.concatenate([img1, img2], axis=2)).permute(2, 0, 1).float()
        )
        imgs_gray_patch = (
            torch.tensor(np.concatenate([img1_patch, img2_patch], axis=2))
            .permute(2, 0, 1)
            .float()
        )

        # output dict
        data_dict = {}
        # data_dict["imgs_gray_rs"] = imgs_gray_rs # 加resize提点
        data_dict["imgs_gray_full"] = imgs_gray_full
        data_dict["imgs_gray_patch"] = imgs_gray_patch
        data_dict["start"] = torch.tensor(start).reshape(2, 1, 1).float()
        data_dict["start_00"] = torch.tensor([0, 0]).reshape(2, 1, 1).float()

        return data_dict

    def data_aug(self, img1, img2, gray=True, normalize=True, horizontal_flip=True):
        def random_crop(img1, img2):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.crop_size
            dx = self.params.rho_x
            dy = self.params.rho_y
            x = np.random.randint(dx, width - patch_size_w - dx)
            y = np.random.randint(dy, height - patch_size_h - dy)
            start = [x, y]
            img1_patch = img1[y : y + patch_size_h, x : x + patch_size_w, :]
            img2_patch = img2[y : y + patch_size_h, x : x + patch_size_w, :]
            return img1_patch, img2_patch, start

        if horizontal_flip and random.random() <= 0.5:
            img1 = np.flip(img1, 1)
            img2 = np.flip(img2, 1)

        if normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        if gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)

        img1_patch, img2_patch, start = random_crop(img1, img2)

        return img1, img2, img1_patch, img2_patch, start


class HomoTestData(Dataset):
    def __init__(self, params):
        self.params = params
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.crop_size = params.crop_size

        if params.is_calc_point_err:
            self.npy_list = os.path.join(params.test_data_dir, "test_list.txt")
            self.npy_path = os.path.join(params.test_data_dir, "Coordinate-v2")
            self.files_path = os.path.join(params.test_data_dir, "test")
            self.data_infor = open(self.npy_list, 'r').readlines()
            sample_num = int(len(self.data_infor) * params.test_ratio)
            self.data_infor = self.data_infor[:sample_num]
        else:
            self.data_dir = os.path.join(params.train_data_dir, 'train')
            pth = os.path.join(params.test_data_dir, 'train_list.txt')
            self.data_infor = open(pth, 'r').readlines()
            sample_num = int(len(self.data_infor) * 0.1)
            random.seed(0)
            random.shuffle(self.data_infor)
            self.data_infor = self.data_infor[:sample_num]
        
    def __len__(self):
        return len(self.data_infor)

    def __getitem__(self, idx):
        data_dict = {}
        params = self.params
        
        if params.is_calc_point_err:
            img_pair = self.data_infor[idx]
            pari_id = img_pair.split(' ')
            npy_name = (
                pari_id[0].split('/')[1] + '_' + pari_id[1].split('/')[1][:-1] + '.npy'
            )
            video_name = img_pair.split('/')[0]
            img_names = img_pair.replace('LM', '').replace(video_name + '/', '').split(' ')
            img_names[-1] = img_names[-1][:-1]
            img_files = os.path.join(self.files_path, video_name)
            img1 = cv2.imread(os.path.join(img_files, img_names[0]))
            img2 = cv2.imread(os.path.join(img_files, img_names[1]))
        else:
            img_names = self.data_infor[idx].split(' ')
            img1 = cv2.imread(os.path.join(self.data_dir, img_names[0]))
            img2 = cv2.imread(os.path.join(self.data_dir, img_names[1][:-1]))

        img1 = cv2.resize(img1, (640, 360), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (640, 360), interpolation=cv2.INTER_AREA)

        ch, cw = self.crop_size
        img1_rs = cv2.resize(img1, (cw, ch))
        img2_rs = cv2.resize(img2, (cw, ch))
        img1_patch, img2_patch, start = self.random_crop(img1, img2)

        img1_gray_rs, img2_gray_rs = self.data_aug(img1_rs, img2_rs)
        img1_gray_patch, img2_gray_patch = self.data_aug(img1_patch, img2_patch)
        img1_gray_full, img2_gray_full = self.data_aug(img1, img2)
        imgs_ori = (
            torch.tensor(np.concatenate([img1, img2], axis=2)).permute(2, 0, 1).float()
        )
        imgs_gray_rs = (
            torch.tensor(np.concatenate([img1_gray_rs, img2_gray_rs], axis=2))
            .permute(2, 0, 1)
            .float()
        )
        imgs_gray_patch = (
            torch.tensor(np.concatenate([img1_gray_patch, img2_gray_patch], axis=2))
            .permute(2, 0, 1)
            .float()
        )
        imgs_gray_full = (
            torch.tensor(np.concatenate([img1_gray_full, img2_gray_full], axis=2))
            .permute(2, 0, 1)
            .float()
        )
        imgs_gray = imgs_gray_rs if params.is_resize_mode else imgs_gray_patch
        # train dataset, 不适合resize，loss大；train dataset 适合crop
        data_dict["imgs_ori"] = imgs_ori
        data_dict["imgs_gray_patch"] = imgs_gray
        data_dict["imgs_gray_full"] = imgs_gray_full
        data_dict["start"] = torch.tensor(start).reshape(2, 1, 1).float()
        data_dict["start_00"] = torch.tensor([0, 0]).reshape(2, 1, 1).float()

        if params.is_calc_point_err:
            point_dic = np.load(os.path.join(self.npy_path, npy_name), allow_pickle=True)
            points = str(point_dic.item())
            pts_info = eval(points)
            pts_1_all, pts_2_all = [], []
            for j in range(6):
                pts_pair = pts_info['matche_pts'][j]
                pts_1_all.append(torch.tensor(pts_pair[0]).float())
                pts_2_all.append(torch.tensor(pts_pair[1]).float())
            pts_1_all = torch.stack(pts_1_all)
            pts_2_all = torch.stack(pts_2_all)
            
            data_dict["points"] = points
            data_dict["video_name"] = video_name
            data_dict["npy_name"] = npy_name
            data_dict['points_1_all'] = pts_1_all
            data_dict['points_2_all'] = pts_2_all

        return data_dict

    def data_aug(self, img1, img2):
        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I
        img1 = np.mean(img1, axis=2, keepdims=True)
        img2 = np.mean(img2, axis=2, keepdims=True)
        return img1, img2
    
    def random_crop(self, img1, img2):
        height, width = img1.shape[:2]
        patch_size_h, patch_size_w = self.crop_size
        dx = self.params.rho_x
        dy = self.params.rho_y
        x = np.random.randint(dx, width - patch_size_w - dx)
        y = np.random.randint(dy, height - patch_size_h - dy)
        start = [x, y]
        img1_patch = img1[y : y + patch_size_h, x : x + patch_size_w, :]
        img2_patch = img2[y : y + patch_size_h, x : x + patch_size_w, :]
        return img1_patch, img2_patch, start


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    _logger.info(
        "Dataset type: {}, transform type: {}".format(
            params.dataset_type, params.transform_type
        )
    )

    dataloaders = {}

    # add train data loader
    if params.dataset_type in ["basic", 'train']:
        train_ds = HomoTrainData(params)
        train_dl = DataLoader(
            train_ds,
            batch_size=params.train_batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=params.cuda,
            drop_last=True,
            prefetch_factor=3,  # for pytorch >=1.5.0
        )
        dataloaders["train"] = train_dl

    # chose test data loader for evaluate
    if params.eval_type in ["test", 'val']:
        test_ds = HomoTestData(params)
        dl = DataLoader(
            test_ds,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=params.cuda,
            prefetch_factor=3,  # for pytorch >=1.5.0
        )
    else:
        dl = None
        raise ValueError("Unknown eval_type in params, should in [val, test]")

    dataloaders[params.eval_type] = dl

    return dataloaders
