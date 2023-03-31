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
from model.offset_net import *
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
            output, temp = model(inputs)
            output = compute_homo(inputs, output)
            output = second_stage(inputs, output, temp)
            imgs = [output['img_ga_m'], output['img_a_m'], output['img_a_pred']]
            imgs = [to_cv2_format(i) for i in imgs]
            draw_info = [cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2]
            cv2.putText(imgs[1], 'inp', (280, 400), *draw_info)
            cv2.putText(imgs[0], 'gt', (280, 450), *draw_info)
            cv2.putText(imgs[2], 'pred', (280, 500), *draw_info)
            pad = np.full((880, 50, 3), 255, np.uint8)
            versus = [imgs[0], pad, imgs[1], pad, imgs[2]]
            compare = np.concatenate(versus, axis=1)
            cv2.imwrite(f'{svd}/{i}_compare.jpg', compare)
            imgs = [to_rgb(i) for i in imgs]
            imageio.mimsave(f'{svd}/{i}_m_pd.gif', [imgs[1], imgs[2]], duration=0.5)
            imageio.mimsave(f'{svd}/{i}_gm_m.gif', [imgs[0], imgs[1]], duration=0.5)
            imageio.mimsave(f'{svd}/{i}_gm_pd.gif', [imgs[0], imgs[2]], duration=0.5)

            avm_inp_path = os.path.join(
                params.test_data_dir, inputs['input_avm_path'][0]
            )
            token = avm_inp_path.split('_p')
            avm_gt_path = token[0] + '_p' + token[1] + '_p0' + token[2][1:]
            shutil.copy2(avm_inp_path, f'{svd}/{i}_avm_inp.jpg')
            shutil.copy2(avm_gt_path, f'{svd}/{i}_avm_gt.jpg')

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
