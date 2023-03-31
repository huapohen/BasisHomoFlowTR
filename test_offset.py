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
    with tqdm(total=len(dataloaders['train'])) as t:
        for i, data_batch in enumerate(dataloaders['train']):
            data_batch = utils.tensor_gpu(data_batch)
            output, temp = model(data_batch)
            output = compute_homo(data_batch, output)
            output = second_stage(data_batch, output, temp)
            imgs = [output['img_ga_m'], output['img_a_m'], output['img_a_pred']]
            imgs = [to_cv2_format(i) for i in imgs]
            compare = np.concatenate(imgs, axis=1)
            cv2.imwrite(f'{svd}/{i}_compare.jpg', compare)
            imgs = [to_rgb(i) for i in imgs]
            imageio.mimsave(f'{svd}/{i}_m_pd.gif', [imgs[1], imgs[2]], duration=0.5)
            imageio.mimsave(f'{svd}/{i}_gm_m.gif', [imgs[0], imgs[1]], duration=0.5)
            imageio.mimsave(f'{svd}/{i}_gm_pd.gif', [imgs[0], imgs[2]], duration=0.5)

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
