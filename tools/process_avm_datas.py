import cv2
import numpy as np
import math
import json
import os
from fev2avm import Fev2AVM


def save_pair_imgs():
    return

def process_video(avm_video_dir, avmer, out_dir, step = 8):
    keys = ["front", "back", "left", "right"]

    os.makedirs(out_dir, exist_ok=True)

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
            frame_prefix = os.path.join(out_dir, basename + f"_{frame_idx}")
            bev_outs = []
            for idx, key in enumerate(keys):
                bev = avmer.warp_image(inputs[idx], avmer.b2f_map[key])
                bev_outs.append(bev)

            # merge avm
            avm_img = avmer.merge2avm(bev_outs)
            
            avmer.save_pair_imgs(bev_outs, frame_prefix)
            
            avm_out_file = frame_prefix + "_avm.jpg"
            cv2.imwrite(avm_out_file, avm_img)

            # get split part
            for idx, key in enumerate(keys):
                split_out = avmer.get_split_part(bev_outs[idx], key)
                split_out_file = frame_prefix + f"_{key}.jpg"
                cv2.imwrite(split_out_file, split_out)

            data_name_list.append(frame_prefix)

        frame_idx += 1

    return data_name_list



if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    warpper = Fev2AVM('b16param/config.json', 'b16param/camera_infor.json', 'b16param')
    
    vid_list = ['20230227113020',
                '20230227113709',
                '20230227113746',
                '20230227113817',
                '20230227113856',
                '20230227113937',
                '20230227114006',
                '20230227114024',
                '20230227114043',
                '20230227114111',
                '20230227114137',
                '20230227114205',
                '20230227114230',
                '20230227114304',
                '20230227114328',
                '20230227114403',
                '20230227114457']
    for vid in vid_list:
        out_dir = f"/home/data/lwb/data/dybev/b16/{vid}"
        process_video(f'/home/data/lwb/data/dybev/AVM_20230227/{vid}', warpper, out_dir, step = 1)


