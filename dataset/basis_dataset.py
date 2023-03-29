import logging
import os
import pickle
import random
import cv2

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class HomoTrainData(Dataset):
    def __init__(self, params):

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.crop_size = params.crop_size

        self.rho = params.rho
        self.normalize = True
        self.horizontal_flip_aug = True

        self.list_path = params.train_data_dir + '/train_list.txt'
        self.data_infor = open(self.list_path, 'r').readlines()
        if params.data_ratio < 1:
            random.seed(1)
            random.shuffle(self.data_infor)
            self.data_infor = self.data_infor[:int(len(self.data_infor) * params.data_ratio)]
        self.data_dir = params.train_data_dir + '/Train/'

        self.seed = 0
        random.seed(self.seed)
        random.shuffle(self.data_infor)

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):

        # img loading
        img_names = self.data_infor[idx]
        img_names = img_names.split(' ')
        img1 = cv2.imread(f'{self.data_dir}/{img_names[0]}')
        img2 = cv2.imread(f'{self.data_dir}/{img_names[1][:-1]}')

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
        px, py = start
        ph, pw = self.crop_size
        pts = [[px, py], [px + pw, py], [px, py + ph], [px + pw, py + ph]]
        pts_1 = torch.from_numpy(np.array(pts)).float()
        pts_2 = torch.from_numpy(np.array(pts)).float()

        start = torch.tensor(start).reshape(2, 1, 1).float()
        # output dict
        data_dict = {}
        data_dict["imgs_gray_full"] = imgs_gray_full
        data_dict["imgs_gray_patch"] = imgs_gray_patch
        data_dict["start"] = start
        data_dict['points_1'] = pts_1
        data_dict['points_2'] = pts_2

        return data_dict

    def data_aug(self, img1, img2, gray=True, normalize=True, horizontal_flip=True):
        def random_crop(img1, img2):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.crop_size

            x = np.random.randint(self.rho, width - self.rho - patch_size_w)
            y = np.random.randint(self.rho, height - self.rho - patch_size_h)
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

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        self.crop_size = params.crop_size

        self.normalize = True
        self.horizontal_flip_aug = False

        self.npy_list = os.path.join(params.test_data_dir, "test_list.txt")
        self.npy_path = os.path.join(params.test_data_dir, "Coordinate-v2")
        self.files_path = os.path.join(params.test_data_dir, "Test")

        self.data_infor = open(self.npy_list, 'r').readlines()

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):

        # img loading
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

        img1 = cv2.resize(img1, (640, 360))
        img2 = cv2.resize(img2, (640, 360))

        # img aug
        img1_rs, img2_rs = img1, img2
        if img1.shape[0] != self.crop_size[0] or img1.shape[1] != self.crop_size[1]:
            img1_rs = cv2.resize(img1_rs, (self.crop_size[1], self.crop_size[0]))
            img2_rs = cv2.resize(img2_rs, (self.crop_size[1], self.crop_size[0]))

        img1_gray, img2_gray = self.data_aug(
            img1_rs,
            img2_rs,
            normalize=self.normalize,
            horizontal_flip=self.horizontal_flip_aug,
        )
        img1_gray_full, img2_gray_full = self.data_aug(
            img1,
            img2,
            normalize=self.normalize,
            horizontal_flip=self.horizontal_flip_aug,
        )
        # array to tensor
        imgs_ori = (
            torch.tensor(np.concatenate([img1, img2], axis=2)).permute(2, 0, 1).float()
        )
        imgs_gray = (
            torch.tensor(np.concatenate([img1_gray, img2_gray], axis=2))
            .permute(2, 0, 1)
            .float()
        )
        imgs_gray_full = (
            torch.tensor(np.concatenate([img1_gray_full, img2_gray_full], axis=2))
            .permute(2, 0, 1)
            .float()
        )

        ph, pw = self.crop_size
        pts = [[0, 0], [pw, 0], [0, ph], [pw, ph]]
        pts_1 = torch.from_numpy(np.array(pts)).float()

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

        # output dict
        data_dict = {}
        data_dict["imgs_gray_full"] = imgs_gray
        data_dict["imgs_gray_patch"] = imgs_gray
        data_dict["start"] = torch.tensor([0, 0]).reshape(2, 1, 1).float()

        data_dict["imgs_ori"] = imgs_ori
        data_dict["points"] = points
        data_dict['points_1'] = pts_1
        data_dict['points_2'] = pts_1
        data_dict['points_1_all'] = pts_1_all
        data_dict['points_2_all'] = pts_2_all
        data_dict["video_name"] = video_name
        data_dict["npy_name"] = npy_name

        return data_dict

    def data_aug(self, img1, img2, gray=True, normalize=True, horizontal_flip=True):

        if horizontal_flip and random.random() <= 0.5:
            img1 = np.flip(img1, 1)
            img2 = np.flip(img2, 1)

        if normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        if gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)

        return img1, img2
