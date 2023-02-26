"""Evaluates the model"""
import argparse
import logging
import os
import cv2
import time
import glob
import json
import random
import imageio
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from common import utils
import model.net as net
import dataset.data_loader as data_loader
from parameters import get_config, dictToObj
import ipdb

''' add this line, because GTX30 serials' default torch.matmul() on cuda is uncorrected '''
torch.backends.cuda.matmul.allow_tf32 = False


parser = argparse.ArgumentParser()
parser.add_argument(
    '--test_data_dir', default='dataset', help="Directory containing the dataset"
)
parser.add_argument(
    '--model_dir',
    default='experiments/base_model',
    help="Directory containing params.json",
)
parser.add_argument(
    '--restore_file',
    default='model_latest.pth',
    help="name of the file in --model_dir containing weights to load",
)
parser.add_argument(
    '--result_files', default="ours_v3_ICCV_final", help="file for store eval results"
)


def load_checkpoint(restore_file, model):
    state = torch.load(restore_file, map_location=torch.device('cpu'))
    model.load_state_dict(state["state_dict"])


def evaluate_main(model, args, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    print("eval begin!")
    result_files = args.result_files
    if not os.path.exists(result_files):
        os.makedirs(result_files)
    # set model to evaluation mode
    model.eval()
    torch.set_grad_enabled(False)
    k = 0
    times = 0
    crop_size = tuple(params.crop_size)
    imgs_dir = args.test_data_dir + '/AVM'
    # imgs_dir = args.test_data_dir
    # imgs_list = glob.glob(os.path.join(imgs_dir, "*A*")) # 相同名字*_A和*_B为一个pair
    # imgs_names = [name.split("/")[-1].split("_")[0] for name in imgs_list]
    # for name in imgs_names:
    #     name_src = os.path.join(imgs_dir, name + "_A.jpg")
    #     name_tg = os.path.join(imgs_dir, name + "_B.jpg")
    if 1:
        name = 'bev'
        name_src = imgs_dir + "/ori.jpg"
        name_tg = imgs_dir + "/pert.jpg"
        img1 = cv2.imread(name_src)
        img2 = cv2.imread(name_tg)
        # img1 = img1[500:,:-200,:]
        # img2 = img2[500:,200:,:]
        img1_rs, img2_rs = img1, img2
        if img1.shape[0] != crop_size[0] or img1.shape[1] != crop_size[1]:
            img1_rs = cv2.resize(img1_rs, (crop_size[1], crop_size[0]))
            img2_rs = cv2.resize(img2_rs, (crop_size[1], crop_size[0]))
        img1_gray, img2_gray = data_aug(
            img1_rs, img2_rs, normalize=True, horizontal_flip=False
        )
        img1_gray_full, img2_gray_full = data_aug(
            img1, img2, normalize=True, horizontal_flip=False
        )

        # array to tensor
        imgs_ori = (
            torch.tensor(np.concatenate([img1, img2], axis=2))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )
        imgs_gray = (
            torch.tensor(np.concatenate([img1_gray, img2_gray], axis=2))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )
        imgs_gray_full = (
            torch.tensor(np.concatenate([img1_gray_full, img2_gray_full], axis=2))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )
        eval_batch = {}
        eval_batch["imgs_ori"] = imgs_ori
        eval_batch["imgs_gray_full"] = imgs_gray_full
        eval_batch["imgs_gray_patch"] = imgs_gray
        eval_batch["start"] = torch.tensor([0, 0]).reshape(1, 2, 1, 1).float()
        eval_batch = utils.tensor_gpu(eval_batch, check_on=True)

        # forward
        eval_results = eval_forward_main(model, eval_batch)

        # get results
        img2_full_warp = (
            eval_results["img2_full_warp"][0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        img2_full_warp = cv2.cvtColor(img2_full_warp, cv2.COLOR_BGR2RGB)
        img1_full = imgs_ori[0, :3, ...].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img1_full = cv2.cvtColor(img1_full, cv2.COLOR_BGR2RGB)
        img2_full = imgs_ori[0, 3:, ...].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img2_full = cv2.cvtColor(img2_full, cv2.COLOR_BGR2RGB)
        vis_img = img2_full_warp.copy()
        vis_img[:, :, 0] = img1_full[:, :, 0]  # 在warped图中把原来的r通道替换为target图的
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        vis_img_ori = img2_full.copy()
        vis_img_ori[:, :, 0] = img1_full[:, :, 0]
        vis_img_ori = cv2.cvtColor(vis_img_ori, cv2.COLOR_BGR2RGB)

        # results saving
        frmes = [img1_full, img2_full_warp]
        svname = os.path.join(result_files, name + ".gif")
        imageio.mimsave(svname, frmes, 'GIF', duration=0.5)

        cv2.imwrite(os.path.join(result_files, name + "_vis.jpg"), vis_img)
        cv2.imwrite(os.path.join(result_files, name + "_vis_ori.jpg"), vis_img_ori)
        times += eval_results["times"]


def eval_forward_main(model, val_batch):
    imgs_full = val_batch["imgs_ori"]
    imgs_gray = val_batch["imgs_gray_patch"]
    batch_size, _, grid_h, grid_w = imgs_full.shape
    # ==================================================================== 网络推理 ===================================================================
    torch.cuda.synchronize()
    start = time.time()
    output = model(val_batch)
    torch.cuda.synchronize()
    end = time.time()
    times = end - start
    # print("times: ", times)
    # ==================================================================== warp ======================================================================
    H_flow_f, H_flow_b = output['H_flow']
    H_flow_f = net.upsample2d_flow_as(
        H_flow_f, imgs_full, mode="bilinear", if_rate=True
    )  # scale
    img2_full_warp = net.get_warp_flow(imgs_full[:, 3:, ...], H_flow_f, start=0)
    # ==================================================================== return ======================================================================
    eval_results = {}
    eval_results["img2_full_warp"] = img2_full_warp
    eval_results["times"] = times
    return eval_results


def data_aug(img1, img2, gray=True, normalize=True, horizontal_flip=True):
    mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
    std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)
    if horizontal_flip and random.random() <= 0.5:
        img1 = np.flip(img1, 1)
        img2 = np.flip(img2, 1)
    if normalize:
        img1 = (img1 - mean_I) / std_I
        img2 = (img2 - mean_I) / std_I
    if gray:
        img1 = np.mean(img1, axis=2, keepdims=True)  # 变均值，灰度
        img2 = np.mean(img2, axis=2, keepdims=True)
    return img1, img2


if __name__ == '__main__':
    """
    Evaluate the model on the test set.
    """
    # ipdb.set_trace()
    # Load the parameters from json file
    args = parser.parse_args()
    # args.gpu_used = '5'
    # args.model_dir = 'experiments/remove_lrr_and_only_photo_loss'

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    params = utils.Params(json_path)
    # use GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "_" in params.gpu_used:
        params.gpu_used = ",".join(params.gpu_used.split("_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used
    params.cuda = torch.cuda.is_available()
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
    # Define the model
    if params.cuda:
        model = net.Net(params).cuda()
        gpu_num = len(params.gpu_used.split(","))
        device_ids = range(gpu_num)
        # device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model = net.Net(params)
    logging.info("Starting evaluation")
    # Reload weights from the saved file
    load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    evaluate_main(model, args, params)
