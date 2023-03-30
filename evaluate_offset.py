"""Evaluates the model"""

import argparse
import logging
import os
import cv2, imageio
import json

import numpy as np
import torch
from torch.autograd import Variable

from torch.utils.data import DataLoader
import model.net as net
from common import utils
from loss.offset_losses import compute_losses
from common.manager import Manager
from parameters import get_config, dictToObj
from easydict import EasyDict
from tqdm import tqdm
from dataset.avm_dataset import AVMDataset

torch.backends.cuda.matmul.allow_tf32 = False

parser = argparse.ArgumentParser()
parser.add_argument(
    '--params_path',
    default='experiments/base_model',
    help="Directory containing params.json",
)
parser.add_argument(
    '--model_dir',
    default='experiments/base_model',
    help="Directory containing params.json",
)

# parser.add_argument('--restore_file', default='experiments/base_model/best_0.5012.pth.tar', help="name of the file in --model_dir containing weights to load")
parser.add_argument(
    '--restore_file',
    default='experiments/base_model/model_latest.pth',
    help="name of the file in --model_dir containing weights to load",
)


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # print("eval begin!")

    # loss status and eval status initial
    manager.reset_loss_status()
    manager.reset_metric_status(manager.params.eval_type)
    model.eval()

    k = 0
    with torch.no_grad():
        # compute metrics over the dataset

        with tqdm(total=len(manager.dataloaders['test'])) as t:
            for data_batch in manager.dataloaders[manager.params.eval_type]:
                data_batch = utils.tensor_gpu(data_batch)
                output_batch = model(data_batch["inputs"])
                loss = compute_losses(output_batch, data_batch, manager.params)

                manager.update_metric_status(
                    metrics=loss,
                    split=manager.params.eval_type,
                    batch_size=manager.params.eval_batch_size,
                )
                eval_save_result(output_batch, data_batch["gt"], manager)
            t.update()
        # eval_save_result(output_batch, data_batch["gt"], manager)

        # update data to logger
        manager.logger.info(
            "Loss/valid epoch_{} {}: {:.3f}".format(
                manager.params.eval_type,
                manager.epoch_val,
                manager.test_status["total"].avg,
            )
        )

        # For each epoch, print the metric
        manager.print_metrics(
            manager.params.eval_type, title=manager.params.eval_type, color="green"
        )

        manager.update_epoch_val()
        model.train()


def eval_save_result(outs, gts, manager, out_dir="imgs"):

    # save dir: model_dir
    save_dir = os.path.join(manager.params.model_dir, out_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, out in enumerate(outs):
        out_img = (out.data.cpu().numpy() * 0.5 + 0.5) * 255
        out_img = out_img.transpose(1, 2, 0).astype(np.uint8)
        gt_img = (gts[idx].data.cpu().numpy() * 0.5 + 0.5) * 255
        gt_img = gt_img.transpose(1, 2, 0).astype(np.uint8)
        save_img = np.concatenate([out_img, gt_img], axis=1)

        cv2.imwrite(
            os.path.join(save_dir, "epoch{}_b{}.jpg").format(manager.epoch_val, idx),
            save_img,
        )


def save_full_result(outs, inputs, b_idx, save_dir="imgs"):

    # save dir: model_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, out in enumerate(outs):
        imgs = (inputs["inputs"][idx] * 0.5 + 0.5) * 255 / 3.0
        imgs = torch.clamp(imgs[:3] + imgs[3:6] + imgs[6:9] + imgs[9:], 0, 255)
        save_img = torch.cat([out, inputs["gt"][idx]], 2)
        save_img = (save_img * 0.5 + 0.5) * 255
        save_img = torch.cat([imgs, save_img], 2).data.cpu().numpy()
        save_img = save_img.transpose(1, 2, 0).astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, "b{}_{}.jpg").format(b_idx, idx), save_img)


def load_checkpoint(model_file, model):
    state = torch.load(model_file, map_location=torch.device('cpu'))
    state_dict = state["state_dict"]
    new_dict = {}
    for key, value in state_dict.items():
        new_dict[key[7:]] = state_dict[key]
    model.load_state_dict(new_dict)


def test_exp(test_lists, save_dir, max_num=100):
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    params = utils.Params(json_path)

    # Update args into params
    params.update(vars(args))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # dataloader
    dataloaders = {}
    test_ds = AVMDataset(test_lists, shuffle=False)
    dl = DataLoader(
        test_ds,
        batch_size=params.eval_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.cuda,
    )
    dataloaders["test"] = dl

    # net
    model = net.fetch_net(params).cuda()
    load_checkpoint(params.restore_file, model)
    model.eval()

    # inference
    num = 0
    with torch.no_grad():
        # compute metrics over the dataset

        with tqdm(total=len(dataloaders['test'])) as t:
            for idx, data_batch in enumerate(dataloaders["test"]):
                data_batch = utils.tensor_gpu(data_batch)
                output_batch = model(data_batch["inputs"])
                loss = compute_losses(output_batch, data_batch, params)

                save_full_result(output_batch, data_batch, idx, save_dir)
                num += output_batch.shape[0]
                if num >= max_num:
                    break
            t.update()


if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    test_files = [
        "/data/xingchen/dataset/AVM/b16_train/dataloader/valid/valid_aug_list.txt"
    ]
    save_dir = "/data/xingchen/results/AVMs/remove_stn_warp_l1vgg"
    test_exp(test_files, save_dir, max_num=500)
