import cv2
import numpy as np
import math
import json
import os
from tqdm import tqdm
from fev2avm import Fev2AVM

def process_video(avm_video_dir, avmer, out_dir, step = 8):
    keys = ["front", "back", "left", "right"]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    basename = os.path.basename(avm_video_dir)
    data_name_list = []
    front_file = os.path.join(avm_video_dir, 'front.mp4')
    back_file = os.path.join(avm_video_dir, 'back.mp4')
    left_file = os.path.join(avm_video_dir, 'left.mp4')
    right_file = os.path.join(avm_video_dir, 'right.mp4')
    if not (os.path.exists(front_file) and os.path.exists(back_file) and os.path.exists(left_file) and os.path.exists(right_file)):
        return data_name_list

    video_files = [front_file, back_file, left_file, right_file]
    caps = []
    for file in video_files:
        caps.append(cv2.VideoCapture(file))

    frame_idx = 0
    while(True):
        inputs = []
        for idx, key in enumerate(keys):
            ret, frame = caps[idx].read()
            if ret:
                inputs.append(frame)
        if len(inputs) != len(keys):
            break
        if frame_idx % step == 0:
            # process
            frame_perfix = os.path.join(out_dir, basename + f"_{frame_idx}")
            bev_outs = []
            for idx, key in enumerate(keys):
                bev = avmer.warp_image(inputs[idx], avmer.b2f_map[key])
                bev_outs.append(bev)

            # merge avm
            avm_img = avmer.merge2avm(bev_outs)

            avm_out_file = frame_perfix + "_avm.jpg"
            cv2.imwrite(avm_out_file, avm_img)

            # get split part
            for idx, key in enumerate(keys):
                split_out = avmer.get_split_part(bev_outs[idx], key)
                split_out_file = frame_perfix + f"_{key}.jpg"
                cv2.imwrite(split_out_file, split_out)

            data_name_list.append(frame_perfix)

        frame_idx += 1

    return data_name_list

if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    warpper = Fev2AVM('b16param/config.json', 'b16param/camera_infor.json', 'b16param')

    out_dir = "/data/xingchen/dataset/AVM/b16_train/valid"
    # process_video('/data/xingchen/dataset/AVM/B16_data/20230227/20230227113817', warpper, out_dir, step = 5)

    step = 3
    # data_dirs = ["/data/xingchen/dataset/AVM/B16_data/20230227", 
                 # "/data/xingchen/dataset/AVM/B16_data/20230302"]
    data_dirs = ["/data/xingchen/dataset/AVM/B16_data/valid"]
    input_dirs = []
    for data_dir in data_dirs:
        datalist = os.listdir(data_dir)
        for one_dir in datalist:
            input_dirs.append(os.path.join(data_dir, one_dir))
    for datadir in tqdm(input_dirs):
        process_video(datadir, warpper, out_dir, step = step)
