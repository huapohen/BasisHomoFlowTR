import os
import cv2
import copy
import ipdb
import torch
import random
import pickle
import logging
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset

_logger = logging.getLogger(__name__)


class HomoData(Dataset):
    def __init__(self, params, mode='train'):
        self.params = params

        self.mode = mode
        if mode == "train":
            self.data_ratio = params.train_data_ratio
        else:
            self.data_ratio = params.test_data_ratio

        self.is_gray = params.is_gray
        self.is_norm = params.is_norm
        self.is_h_flip = False
        if mode == 'train' and params.is_h_flip:
            self.is_h_flip = True

        def _bev_data(set_name):
            txt_list = []
            suffix = '_pair_names_list.txt'
            if len(params.camera_list) == 4:
                txt_list = ['all_fblr' + suffix]
            else:
                for cam in params.camera_list:
                    txt_list.append(cam + suffix)
            data_dir = os.path.join(
                params.data_dir, set_name, mode, params.data_source_type
            )
            data_bev = []
            for txt_name in txt_list:
                path = data_dir + '/pair/' + txt_name
                names = open(path, 'r').readlines()
                names = [set_name + ' ' + i for i in names]
                data_bev += names
            return data_bev

        data_nature = open(f'{params.data_dir}/{mode}_list.txt', 'r').readlines()

        self.seed = 0
        random.seed(self.seed)
        self.data_sample = []
        self.sample_number = {}
        for data_ratio in self.data_ratio:
            set_name = data_ratio[0]
            if set_name == 'nature':
                name_list = data_nature
            else:  # v6, v4
                name_list = _bev_data(set_name)
            self.data_sample += name_list
            percentage = int(len(name_list) * data_ratio[1])
            if mode in ['test', 'val']:
                name_list = sorted(name_list)
            else:  # train
                random.shuffle(name_list)
            name_list = name_list[:percentage]
            self.sample_number[set_name] = {
                'ratio': float(data_ratio[1]),
                "samples": len(name_list),
            }

        self.sample_number["total_samples"] = len(self.data_sample)

        if mode == 'train':
            random.shuffle(self.data_sample)

    def __len__(self):
        # return size of dataset
        return len(self.data_sample)

    def __getitem__(self, idx):
        '''
        debug dataloader: set num_workers=0
        '''
        params = self.params

        img_names = self.data_sample[idx]

        is_bev_sample = True if '_p' in img_names else False

        if is_bev_sample:
            img_names = img_names.split(' ')
            set_name = img_names[0]
            img1_name = img_names[1]
            img2_name = img_names[2][:-1]
            prefix_path = os.path.join(
                params.data_dir, set_name, self.mode, params.data_source_type
            )
            video_name = '_'
            npy_name = ''
            npy_path = ''
        else:  # nature
            img_pair = self.data_sample[idx]
            pari_id = img_pair.split(' ')
            npy_name = (
                pari_id[0].split('/')[1] + '_' + pari_id[1].split('/')[1][:-1] + '.npy'
            )
            video_name = img_pair.split('/')[0]
            img_names = (
                img_pair.replace('LM', '').replace(video_name + '/', '').split(' ')
            )
            img_names[-1] = img_names[-1][:-1]
            img1_name, img2_name = img_names
            nature_mode = 'Train' if self.mode == 'train' else 'Test'
            prefix_path = os.path.join(params.data_dir, 'nature', nature_mode)
            npy_path = os.path.join(params.data_dir, 'nature', "Coordinate-v2")

        img1 = cv2.imread(f'{prefix_path}/{img1_name}')
        img2 = cv2.imread(f'{prefix_path}/{img2_name}')

        # align nature, otherwise img_list can not make up batch
        if is_bev_sample:
            img1 = cv2.resize(img1, (640, 360))
            img2 = cv2.resize(img2, (640, 360))

        if params.is_test_assigned_img:
            img1 = cv2.imread('dataset/test/10467_A.jpg')
            img2 = cv2.imread('dataset/test/10467_B.jpg')

        # if not is_bev_sample and self.mode == 'test':
        if self.mode == 'test':
            img1 = cv2.resize(img1, (640, 360))
            img2 = cv2.resize(img2, (640, 360))

        # output dict
        data_dict = {}

        # crop_size = params.crop_size_dybev if is_bev_sample else params.crop_size
        crop_size = params.crop_size

        aug_params = [crop_size, params.rho, params.mean_I, params.std_I]
        aug_params += [self.is_gray, self.is_norm, self.is_h_flip]
        data_dict = data_aug(data_dict, aug_params, img1, img2, mode=self.mode)

        data_dict = get_crop_pts(data_dict['start'], crop_size)

        if self.mode == 'test':
            pts_params = npy_path, video_name, npy_name, is_bev_sample
            data_dict = get_manual_pts(data_dict, pts_params)

        return data_dict


def get_crop_pts(data_dict, crop_size):
    px, py = data_dict['start']
    ph, pw = crop_size
    pts = [[px, py], [px + pw, py], [px, py + ph], [px + pw, py + ph]]
    data_dict['points_1'] = torch.from_numpy(np.array(pts)).float()
    data_dict['points_2'] = torch.from_numpy(np.array(pts)).float()
    return data_dict


def get_manual_pts(data_dict, pts_params):
    npy_path, video_name, npy_name, is_bev_sample = pts_params

    if not is_bev_sample:  # nature
        point_dic = np.load(os.path.join(npy_path, npy_name), allow_pickle=True)
        points = str(point_dic.item())
        pts_info = eval(points)
        pts_1_all, pts_2_all = [], []
        for j in range(6):
            pts_pair = pts_info['matche_pts'][j]
            pts_1_all.append(torch.tensor(pts_pair[0]).float())
            pts_2_all.append(torch.tensor(pts_pair[1]).float())
        pts_1_all = torch.stack(pts_1_all)
        pts_2_all = torch.stack(pts_2_all)
    else:  # bev
        pts_1_all = torch.ones(6, 2)
        pts_2_all = torch.ones(6, 2)

    data_dict['points_1_all'] = pts_1_all
    data_dict['points_2_all'] = pts_2_all
    data_dict["video_name"] = video_name
    data_dict["npy_name"] = npy_name

    return data_dict


def data_aug(data_dict, aug_params, img1, img2, mode='train'):

    crop_size, rho, mean_I, std_I, is_gray, is_norm, is_h_flip = aug_params

    def _random_crop(img1, img2):
        height, width = img1.shape[:2]
        patch_size_h, patch_size_w = crop_size
        x = np.random.randint(rho, width - rho - patch_size_w)
        y = np.random.randint(rho, height - rho - patch_size_h)
        start = [x, y]
        img1_patch = img1[y : y + patch_size_h, x : x + patch_size_w, :]
        img2_patch = img2[y : y + patch_size_h, x : x + patch_size_w, :]
        return img1_patch, img2_patch, start

    def _resize(img1, img2):
        if img1.shape[0] != crop_size[0] or img1.shape[1] != crop_size[1]:
            img1_rs = cv2.resize(img1, (crop_size[1], crop_size[0]))
            img2_rs = cv2.resize(img2, (crop_size[1], crop_size[0]))
        return img1_rs, img2_rs, [0, 0]

    if is_h_flip and random.random() <= 0.5 and mode == 'train':
        img1 = np.flip(img1, 1)
        img2 = np.flip(img2, 1)

    if mode == 'test':
        img1_patch_bgr, img2_patch_bgr, start = _resize(img1, img1)
    elif mode == 'train':
        img1_patch_bgr, img2_patch_bgr, start = _random_crop(img1, img2)
    else:
        raise ValueError

    bgr_list = [img1, img2, img1_patch_bgr, img2_patch_bgr]
    rgb_list = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]
    if is_norm:
        bgr_list = [(bgr - mean_I) / std_I for bgr in bgr_list]
    gray_list = bgr_list
    if is_gray:
        gray_list = [np.mean(bgr, axis=2, keepdims=True) for bgr in bgr_list]
    imgs = (*gray_list, *rgb_list)
    imgs = [torch.tensor(img.transpose((2, 0, 1))).float() for img in imgs]

    # output
    data_dict["start"] = torch.tensor(start).reshape(2, 1, 1).float()
    key_list = [
        'img1_full_gray',
        'img2_full_gray',
        'img1_patch_gray',
        'img2_patch_gray',
        'img1_full_rgb',
        'img2_full_rgb',
        'img1_patch_rgb',
        'img2_patch_rgb',
    ]
    for i, key in enumerate(key_list):
        data_dict[key] = imgs[i]

    return data_dict


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test'
                depending on which data is required
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
    if params.dataset_type in ['basic', 'train']:
        train_ds = HomoData(params, 'train')
        dl = DataLoader(
            train_ds,
            batch_size=params.train_batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=params.cuda,
            drop_last=True,
            # prefetch_factor=3, # for pytorch >=1.5.0
        )
        dl.sample_number = train_ds.sample_number
        dataloaders["train"] = dl

    # chose test data loader for evaluate
    if params.eval_type in ['val', 'test']:
        test_ds = HomoData(params, 'test')
        dl = DataLoader(
            test_ds,
            batch_size=params.eval_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=params.cuda
            # prefetch_factor=3, # for pytorch >=1.5.0
        )
        dl.sample_number = test_ds.sample_number
    else:
        dl = None
        raise ValueError("Unknown eval_type in params, should in [val, test]")

    dataloaders[params.eval_type] = dl

    return dataloaders
