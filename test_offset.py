import os
import cv2
import ipdb
import torch
import shutil
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from common import utils
import model.net as net
from model.offset_net_v2 import *
import dataset.data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/offset')
parser.add_argument('--restore_file', default='model_latest.pth')
parser.add_argument('--result_dir', default="experiments/offset/result")


def inference(model, params):
    model.eval()
    torch.set_grad_enabled(False)
    svd = params.result_dir
    dataloaders = data_loader.fetch_dataloader(params)
    with tqdm(total=len(dataloaders['test'])) as t:
        for i, inputs in enumerate(dataloaders['test']):
            inputs = utils.tensor_gpu(inputs)
            inp_path = inputs['input_avm_path'][0]
            output = model(inputs)
            mask_dict = mask_to_device(inputs)
            output = compute_homo(inputs, output)
            output = second_stage(inputs, output, mask_dict)
            imgs = [output['img_ga_m'], output['img_a_m'], output['img_a_pred']]
            imgs = [to_cv2_format(i) for i in imgs]
            draw_info = [cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2]
            cv2.putText(imgs[1], 'inp', (280, 400), *draw_info)
            cv2.putText(imgs[0], 'gt', (280, 450), *draw_info)
            cv2.putText(imgs[2], 'pred', (280, 500), *draw_info)
            pad = np.full((880, 50, 3), 30, np.uint8)
            versus = [imgs[0], pad, imgs[1], pad, imgs[2]]
            compare = np.concatenate(versus, axis=1)
            suff = inp_path.split(os.sep)[-1].split('.')[0]
            cv2.imwrite(f'{svd}/{i}_compare__{suff}.jpg', compare)
            imgs = [to_rgb(i) for i in imgs]
            i12, i01, i02 = [imgs[1], imgs[2]], [imgs[0], imgs[1]], [imgs[0], imgs[2]]
            imageio.mimsave(f'{svd}/{i}_m_pd.gif', i12, duration=0.5)
            imageio.mimsave(f'{svd}/{i}_gm_m.gif', i01, duration=0.5)
            imageio.mimsave(f'{svd}/{i}_gm_pd.gif', i02, duration=0.5)
            avm_inp_path = os.path.join(params.test_data_dir, inp_path)
            token = avm_inp_path.split('_p')
            avm_gt_path = token[0] + '_p' + token[1] + '_p0' + token[2][1:]
            shutil.copy2(avm_inp_path, f'{svd}/{i}_avm_inp.jpg')
            shutil.copy2(avm_gt_path, f'{svd}/{i}_avm_gt.jpg')
            to_save_mask(f'{svd}/{i}_avm_mask.jpg', output['ones_mask_w_avm'])

            t.set_description(desc='')
            t.update()


if __name__ == '__main__':
    # ipdb.set_trace()

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path)
    params = utils.Params(json_path)
    params.eval_batch_size = 1
    params.num_workers = 1
    torch.manual_seed(42)
    model = net.fetch_net(params)
    params.cuda = torch.cuda.is_available()
    if params.cuda:
        torch.cuda.manual_seed(42)
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
    restore_file = os.path.join(args.model_dir, args.restore_file)
    state = torch.load(restore_file, map_location=torch.device('cpu'))
    model.load_state_dict(state["state_dict"])
    if os.path.exists(args.result_dir):
        shutil.rmtree(args.result_dir)
    os.makedirs(args.result_dir, exist_ok=True)
    params.result_dir = args.result_dir
    inference(model, params)
