import os
import sys
import cv2
import shutil
import imageio
import numpy as np
from torchvision.utils import save_image


def vis_save_image(input, output, bs=1):
    svd = 'experiments/tmp'
    os.makedirs(svd, exist_ok=True)
    shutil.rmtree(svd)
    os.makedirs(svd)

    gif_pair = [
        ('img1_patch_rgb', 'img2_patch_rgb_warp'),
        ('img2_patch_rgb', 'img1_patch_rgb_warp'),
        ('img1_full_rgb', 'img2_full_rgb_warp'),
        ('img2_full_rgb', 'img1_full_rgb_warp'),
    ]

    for pair in gif_pair:
        save_image(input[pair[0]][:bs] / 255.0, f'{svd}/{pair[0]}.jpg')
        save_image(output[pair[1]][:bs] / 255.0, f'{svd}/{pair[1]}.jpg')
        i1 = cv2.imread(f'{svd}/{pair[0]}.jpg')
        i2 = cv2.imread(f'{svd}/{pair[1]}.jpg')
        txt_info = (cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
        cv2.putText(i1, f'b_{pair[0]}', (200, 150), *txt_info)
        cv2.putText(i2, f'b_{pair[1]}', (200, 200), *txt_info)
        imageio.mimsave(f'{svd}/b_{pair[0]}.gif', [i1, i2], 'GIF', duration=0.5)


def eval_save_result_kernel(save_file, save_name, manager, k=0):

    type_name = 'gif' if type(save_file) == list else 'jpg'
    save_dir_gif = os.path.join(manager.params.model_dir, type_name)
    if not os.path.exists(save_dir_gif):
        os.makedirs(save_dir_gif)

    save_dir_gif_epoch = os.path.join(save_dir_gif, str(manager.epoch_val))
    if k == manager.params.save_iteration and os.path.exists(save_dir_gif_epoch):
        shutil.rmtree(save_dir_gif_epoch)
    if not os.path.exists(save_dir_gif_epoch):
        os.makedirs(save_dir_gif_epoch)

    save_path = os.path.join(save_dir_gif_epoch, save_name + '.' + type_name)
    if type(save_file) == list:  # save gif
        imageio.mimsave(save_path, save_file, 'GIF', duration=0.5)
    elif type(save_file) == str:  # save string information
        f = open(save_path, 'w')
        f.write(save_file)
        f.close()
    else:  # save single image
        cv2.imwrite(save_path, save_file)


def eval_save_result(manager, j, k, input, output):
    img_pair = [
        ('img1_full_rgb', 'img2_full_rgb_warp'),
        ('img2_full_rgb', 'img1_full_rgb_warp'),
    ]
    for pair in img_pair:
        i1 = input[pair[0]][j]
        i2 = output[pair[1]][j]
        frames = [i.permute(1, 2, 0).cpu().numpy().astype(np.uint8) for i in [i1, i2]]
        svn = input["npy_name"][j] + "_" + str(output["errs"][j]) + f"_{pair[0]}"
        eval_save_result_kernel(frames, svn, manager, k)
