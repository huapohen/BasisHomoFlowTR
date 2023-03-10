import cv2
import numpy as np
import math
import json
import os
import ipdb
import imageio
import shutil
from fev2avm import Fev2AVM
from tqdm import tqdm


def save_pair_imgs():
    return

def process_video(avm_video_dir, avmer, out_dir, step = 8):
    keys = ["front", "back", "left", "right"]
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
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
            avm_img = avmer.merge2avm(bev_outs, is_add_mask=True)
            # avm_gif = avmer.merge2avm(bev_outs, is_add_mask=False)
            
            avmer.save_pair_imgs(bev_outs, frame_prefix)
            
            cv2.imwrite(frame_prefix + "_avm.jpg", avm_img)
            
            # imageio.mimsave(frame_prefix + "_avm-gif.gif", avm_gif, 'GIF', duration=0.5)

            # get split part
            for idx, key in enumerate(keys):
                split_out = avmer.get_split_part(bev_outs[idx], key)
                split_out_file = frame_prefix + f"_a-{key}.jpg"
                cv2.imwrite(split_out_file, split_out)

            data_name_list.append(frame_prefix)

        frame_idx += 1

    return data_name_list



if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    warpper = Fev2AVM('b16param/config.json', 'b16param/camera_infor.json', 'b16param')
    
    vids_0227 = [
        '20230227113020',
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
        '20230227114457', # 17
    ]
    
    vids_0302 = [
        '20230302155357',
        '20230302155444',
        '20230302155508',
        '20230302155521',
        '20230302155534',
        '20230302155547',
        '20230302155600',
        '20230302155613',
        '20230302155626',
        '20230302155638',
        '20230302155657',
        '20230302155719',
        '20230302155732',
        '20230302155745',
        '20230302155758',
        '20230302155811',
        '20230302155823',
        '20230302155838',
        '20230302160235',
        '20230302160310',
        '20230302160343',
        '20230302160416',
        '20230302160449',
        '20230302160522',
        '20230302160555', # 25
    ]
    vids_all = vids_0227 + vids_0302
    
    bp = '/home/data/lwb/data/dybev'
    for vid in tqdm(vids_all):
        inp_dir = f'{bp}/videos_{vid[:8]}/{vid}'
        out_dir = f"{bp}/b16/{vid}"
        process_video(inp_dir, warpper, out_dir, step=1)


