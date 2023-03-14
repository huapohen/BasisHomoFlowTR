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
        # mode = 'train'
        self.params = params
        self.mode = mode
        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
        if params.set_name == 'b16':
            self.crop_size = params.crop_size_outdoor
        elif params.set_name == 'b07':
            self.crop_size = params.crop_size_dybev
        else:
            raise
        self.rho = params.rho_dybev
        
        self.is_balance = params.is_balance
        self.normalize = True
        self.gray = True
        self.horizontal_flip_aug = True if mode == 'train' else False
        
        base_path = '/home/data/lwb/data/dybev/'
        self.data_dir = os.path.join(base_path, params.set_name)
        path = os.path.join(self.data_dir, f'{mode}.txt')
        
        # self.data_all = open(path, 'r').readlines()
        self.data_all = []
        
        # if 0 and params.is_include_dataset_nature:
        if params.is_include_dataset_nature:
            path = os.path.join(base_path, 'nature', f'{mode}_list.txt')
            data_nature = open(path, 'r').readlines()
            random.shuffle(data_nature)
            num_nature = len(data_nature)
            ratio = params.dataset_nature_ratio
            self.data_all += data_nature[:int(ratio*num_nature)]
            
        total_sample = len(self.data_all)
        
        random.seed(params.seed)
        np.random.seed(params.seed)
        
        if mode == 'train':
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
        
        data_nature_dir = self.data_dir.replace('b16', f'nature/{self.mode}')
        data_dir = self.data_dir if '2023' in img_names[0] else data_nature_dir
        for i in range(int(len(img_names) / 2)):
            img1 = cv2.imread(f'{data_dir}/{img_names[i * 2]}')
            img2 = cv2.imread(f'{data_dir}/{img_names[i * 2 + 1].rsplit()[0]}')
            img1 = cv2.resize(img1, (640, 360))
            img2 = cv2.resize(img2, (640, 360))
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
        data_dict['start'] = torch.tensor([px, py]).reshape(2, 1, 1).float()
        data_dict["imgs_ori"] = torch.tensor(imgs_ori).permute(2, 0, 1).float()
        data_dict["imgs_gray_full"] = torch.tensor(full).permute(2, 0, 1).float()
        data_dict["imgs_gray_patch"] = torch.tensor(patch).permute(2, 0, 1).float()
        data_dict['points_1'] = torch.cat(pts_1_list, dim=0)
        data_dict['points_2'] = torch.cat(pts_2_list, dim=0)
        # ipdb.set_trace()
        if self.params.set_name == 'b16':
            # 20230227113856/20230227113856_3_f_l.jpg
            frames_name = img_names[0].split(os.sep)[1].split('.')[0]
            frames_name += '_vs_'+ '_'.join(img_names[1].split('.')[0].split('_')[1:])
            data_dict['frames_name'] = frames_name
        else:
            # l_f_pair/00090/20221205135519_front_00090_p0008_0.jpg
            data_dict['frames_name'] = ''
        
        # ipdb.set_trace()
        return data_dict

    def data_aug(self, img1, img2):
        def img_balance(img1, img2): 
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV).astype(np.float32)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.float32)
            k_h= np.mean(img1[:, :, 0]) / np.mean(img2[:, :, 0]) 
            k_s= np.mean(img1[:, :, 1]) / np.mean(img2[:, :, 1]) 
            k_v= np.mean(img1[:, :, 2]) / np.mean(img2[:, :, 2]) 
            # img2[:, :, 0] = img2[:, :, 0] * k_h 
            img2[:, :, 1] = img2[:, :, 1] * k_s 
            img2[:, :, 2] = img2[:, :, 2] * k_v 
            img1 = np.clip(img1, 0, 255)
            img2 = np.clip(img2, 0, 255)  
            img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_HSV2BGR)
            img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_HSV2BGR) 
            img1 = np.clip(img1, 0, 255)
            img2 = np.clip(img2, 0, 255) 
            return img1, img2
        
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
        
        if self.is_balance:
            img1, img2 = img_balance(img1, img2)

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
    
def white_balance(img):
    '''
    完美反射白平衡
    STEP 1：计算每个像素的R\G\B之和
    STEP 2：按R+G+B值的大小计算出其前Ratio%的值作为参考点的的阈值T
    STEP 3：对图像中的每个点，计算其中R+G+B值大于T的所有点的R\G\B分量的累积和的平均值
    STEP 4：对每个点将像素量化到[0,255]之间
    依赖ratio值选取而且对亮度最大区域不是白色的图像效果不佳。
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    img = img.astype(np.float32)
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    img_sum = b + g + r

    hists, bins = np.histogram(img_sum.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1

    sum_b, sum_g, sum_r = 0, 0, 0
    index = img_sum >= key
    sum_b = np.sum(b[index])
    sum_g = np.sum(g[index])
    sum_r = np.sum(r[index]) 
    time = np.sum(index)

    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    maxvalue = np.max(img)
    img[:, :, 0] = img[:, :, 0] * maxvalue / avg_b
    img[:, :, 1] = img[:, :, 1] * maxvalue / avg_g
    img[:, :, 2] = img[:, :, 2] * maxvalue / avg_r
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


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



        
