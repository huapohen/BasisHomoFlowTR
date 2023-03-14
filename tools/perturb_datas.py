import os
import glob
import cv2
import numpy as np
import math
import json
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
import ipdb

def get_f2bev_gt(input_dir, out_dir):
    warpper = Fev2Bev('/data/xingchen/project/dynamicbev/dataset/data/homo2.json')
    b2f_map = warpper.get_bev_map()
    img_files = glob.glob(os.path.join(input_dir, '*.jpg'))
    for img_file in tqdm(img_files):
        basename = os.path.basename(img_file)
        basename = basename.split('.')[0] + '_f2bev.jpg'
        out_file = os.path.join(out_dir, basename)

        img = cv2.imread(img_file, 0)
        dst = warpper.warp_image(img, b2f_map)
        cv2.imwrite(out_file, dst)

def draw_circles(img, pts, color = (255, 0, 0)):
    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, color, -1)
    return img

def perturb_func(pts, gen_num, offset_pix_range = 10, mode='random'):
    assert mode in ['random', 'permutation', 'diffusion', 'learned', 'tiny_perspective']
    if mode == 'random':
        offset = (np.random.rand(gen_num, pts.shape[1], pts.shape[2]) - 0.5) * 2 * offset_pix_range
    if mode == 'tiny_perspective':
        offsets = []
        for i in range(gen_num):
            offset1 = np.random.rand() * offset_pix_range
            offset2 = -np.random.rand() * offset_pix_range
            pt_offset = np.zeros((4, 2), dtype = np.float32)
            part = np.random.randint(0, 4)
            if part == 0:
                pt_offset[0, 0] = offset1
                pt_offset[1, 0] = offset2
            elif part == 1:
                pt_offset[1, 1] = offset1
                pt_offset[2, 1] = offset2
            elif part == 2:
                pt_offset[3, 0] = offset1
                pt_offset[2, 0] = offset2
            elif part == 3:
                pt_offset[0, 1] = offset1
                pt_offset[3, 1] = offset2
            offsets.append(pt_offset[np.newaxis])
        offset = np.concatenate(offsets, 0)
    dst_pts = pts + offset
    return offset, dst_pts

def perturb_onekey_data(src_pts, img_file, gt_file, out_img_dir, per_num, idx, offset_pixel_range = 5):
    # print(f'process {idx} data')
    basename = os.path.basename(img_file).split('.')[0]
    key_postfix = basename[basename.rfind('_'):]
    basename = basename[:basename.rfind('_')]
    img = cv2.imread(img_file)
    gt = cv2.imread(gt_file)
    offset, dst_pts = perturb_func(src_pts[np.newaxis], per_num, mode = 'random', offset_pix_range = offset_pixel_range)

    src_pt = src_pts.reshape(-1, 1, 2)
    for i, dst_pt in enumerate(dst_pts):
        # dst_pt = np.array([[10, 0], [-10, 0], [50, -10], [-50, -10]], dtype = np.float32).reshape(-1, 1, 2) + src_pt  # for test
        dst_pt = dst_pt.reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pt, dst_pt)
        perturb_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        post_str = f'{basename}_p{i}'
        out_img_file = os.path.join(out_img_dir, post_str+key_postfix+'.jpg')
        cv2.imwrite(out_img_file, perturb_img)

        # gt
        cv2.imwrite(os.path.join(out_img_dir, post_str + '_avm.jpg'), gt)

def perturb_one_data(src_pts, data_prefix, out_img_dir, per_num, file_idx, offset_pixel_range = 5):
    print(f'process {file_idx} data')
    keys = ["front", "back", "left", "right"]

    gt_file = data_prefix + "_avm.jpg"
    for idx, key in enumerate(keys):
        img_file = data_prefix + "_{}.jpg".format(key)
        perturb_onekey_data(src_pts[idx], img_file, gt_file, out_img_dir, per_num, idx, offset_pixel_range = offset_pixel_range)

def perturb_datas(src_file_list, out_img_dir, per_num = 100, offset_pixel_range = 5):
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)

    pts_width = 100
    bev_size = (616, 880)
    fb_size = (616, 244)
    lr_size = (221, 880)

    front_pts = [[(bev_size[0]-pts_width)//2, (fb_size[1]-pts_width)//2], [(bev_size[0]-pts_width)//2 + pts_width, (fb_size[1]-pts_width)//2],
                 [(bev_size[0]-pts_width)//2, (fb_size[1]-pts_width)//2 + pts_width], [(bev_size[0]-pts_width)//2 + pts_width, (fb_size[1]-pts_width)//2 + pts_width]]
    front_pts = np.array(front_pts, dtype = np.float32)
    back_pts = deepcopy(front_pts)
    back_pts[:, 1] += bev_size[1] - fb_size[1]

    left_pts = [[(lr_size[0]-pts_width)//2, (lr_size[1]-pts_width)//2], [(lr_size[0]-pts_width)//2 + pts_width, (lr_size[1]-pts_width)//2],
                 [(lr_size[0]-pts_width)//2, (lr_size[1]-pts_width)//2 + pts_width], [(lr_size[0]-pts_width)//2 + pts_width, (lr_size[1]-pts_width)//2 + pts_width]]
    left_pts = np.array(left_pts, dtype = np.float32)
    right_pts = deepcopy(left_pts)
    right_pts[:, 0] += bev_size[0] - lr_size[0]

    src_pts = [front_pts, back_pts, left_pts, right_pts]

    # show points
    # pts_show = np.zeros((bev_size[1], bev_size[0], 3), np.uint8)
    # pts_show = draw_circles(pts_show, front_pts, (255, 0, 0))
    # pts_show = draw_circles(pts_show, back_pts, (0, 255, 0))
    # pts_show = draw_circles(pts_show, left_pts, (0, 0, 255))
    # pts_show = draw_circles(pts_show, right_pts, (255, 255, 0))
    # cv2.imwrite('pts_show.jpg', pts_show)

    # get file list
    data_list = []
    with open(src_file_list, 'r') as f:
        for line in f.readlines():
            data_list.append(line.strip('\n'))

    # for idx, data_prefix in enumerate(tqdm(data_list)):
        # perturb_one_data(src_pts, data_prefix, out_img_dir, per_num, idx, offset_pixel_range)
    
    Parallel(n_jobs = 4)(delayed(perturb_one_data)(src_pts, data_prefix, out_img_dir, per_num, idx, offset_pixel_range)
                                                    for idx, data_prefix in enumerate(data_list))

if __name__ == '__main__':
    # ipdb.set_trace()
    data_file = '/data/xingchen/dataset/AVM/b16_train/dataloader/valid/valid_list.txt'
    out_dir = '/data/xingchen/dataset/AVM/b16_train/valid_augment'
    perturb_datas(data_file, out_dir, per_num = 10, offset_pixel_range = 2)
