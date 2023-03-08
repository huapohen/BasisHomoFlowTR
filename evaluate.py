"""Evaluates the model"""

import os
import sys
import cv2
import json
import ipdb
import shutil
import imageio
import logging
import argparse
import numpy as np

import torch
from torch.autograd import Variable
from tqdm import tqdm

# import dataset.data_loader_dybev as data_loader_dybev
import dataset.data_loader as data_loader
import model.net as net
from common import utils
from loss.losses import compute_losses, compute_eval_results
from common.manager import Manager
from parameters import get_config, dictToObj
from easydict import EasyDict


parser = argparse.ArgumentParser()
parser.add_argument(
    '--params_path',
    default=None,
    help="Directory containing params.json",
)
parser.add_argument(
    '--model_dir',
    default=None,
    help="Directory containing params.json",
)
parser.add_argument(
    '--restore_file',
    default=None,
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
    params = manager.params

    RE = [
        '0000011',
        '0000016',
        '00000147',
        '00000155',
        '00000158',
        '00000107',
        '00000239',
        '0000030',
    ]
    LT = [
        '0000038',
        '0000044',
        '0000046',
        '0000047',
        '00000238',
        '00000177',
        '00000188',
        '00000181',
    ]
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []
    
    kpr_list = []

    with torch.no_grad():
        # compute metrics over the dataset
        iter_max = len(manager.dataloaders[params.eval_type])
        with tqdm(total=iter_max) as t:
            for k, data_batch in enumerate(manager.dataloaders[params.eval_type]):
                # data parse
                imgs_full = data_batch["imgs_ori"]
                video_name = data_batch["video_name"]
                npy_name = data_batch["npy_name"]
                # move to GPU if available
                data_batch = utils.tensor_gpu(data_batch)
                # compute model output
                output_batch = model(data_batch)
                # compute all metrics on this batch
                eval_results = compute_eval_results(data_batch, output_batch, params)
                img1s_full_warp = eval_results["img1_full_warp"]
                err_avg = eval_results["errs"]

                for j in range(len(err_avg)):

                    if video_name[j] in RE:
                        MSE_RE.append(err_avg[j])
                    elif video_name[j] in LT:
                        MSE_LT.append(err_avg[j])
                    elif video_name[j] in LL:
                        MSE_LL.append(err_avg[j])
                    elif video_name[j] in SF:
                        MSE_SF.append(err_avg[j])
                    elif video_name[j] in LF:
                        MSE_LF.append(err_avg[j])

                    if k % params.save_iteration == 0 and params.is_save_gif:
                        img2_full = imgs_full[j, 3:, ...].permute(1, 2, 0)
                        img2_full = img2_full.cpu().numpy().astype(np.uint8)
                        img2_full = cv2.cvtColor(img2_full, cv2.COLOR_BGR2RGB)

                        img1_full_warp = img1s_full_warp[j].permute(1, 2, 0)
                        img1_full_warp = img1_full_warp.cpu().numpy().astype(np.uint8)
                        img1_full_warp = cv2.cvtColor(img1_full_warp, cv2.COLOR_BGR2RGB)

                        save_file = [img2_full, img1_full_warp]
                        # save_file = img1_full_warp
                        save_name = npy_name[j] + "_" + str(err_avg[j])
                        eval_save_result(save_file, save_name, manager, k)

                prt_str = f"{k}:{np.mean(err_avg):.4f} "
                kpr_list.append(prt_str)
                t.set_description(prt_str)
                t.update()
        
        kpr_dir = os.path.join(params.model_dir, 'kpr')
        os.makedirs(kpr_dir, exist_ok=True)
        kpr_name =  f'epoch={params.current_epoch:02d}_k_points_err_record.txt'
        kpr_path = os.path.join(kpr_dir, kpr_name)
        if os.path.exists(kpr_path):
            os.remove(kpr_path)
        kpr = open(kpr_path, 'a+')
        kpr.write(('\n').join(kpr_list))
        kpr.close()

        MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
        MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
        MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
        MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
        MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
        MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5

        Metric = {
            "MSE_RE_avg": MSE_RE_avg,
            "MSE_LT_avg": MSE_LT_avg,
            "MSE_LL_avg": MSE_LL_avg,
            "MSE_SF_avg": MSE_SF_avg,
            "MSE_LF_avg": MSE_LF_avg,
            "AVG": MSE_avg,
        }
        manager.update_metric_status(
            metrics=Metric, split=manager.params.eval_type, batch_size=1
        )

        # update data to logger
        manager.logger.info(
            "Loss/Test epoch_{} AVG: {:.4f}. "
            + "RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                manager.epoch_val,
                MSE_avg,
                MSE_RE_avg,
                MSE_LT_avg,
                MSE_LL_avg,
                MSE_SF_avg,
                MSE_LF_avg,
            )
        )
 

        # For each epoch, print the metric
        manager.print_metrics(
            manager.params.eval_type, title=manager.params.eval_type, color="green"
        )

        # manager.epoch_val += 1

        model.train()


def eval_save_result(save_file, save_name, manager, k=0):

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
        utils.create_gif(save_file, save_path)
    elif type(save_file) == str:  # save string information
        f = open(save_path, 'w')
        f.write(save_file)
        f.close()
    else:  # save single image
        cv2.imwrite(save_path, save_file)
    if manager.params.is_vis_and_exit:
        sys.exit()


def run_all_exps(exp_id):
    """
    Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    if args.params_path is not None and args.restore_file is None:
        # run test by DIY in designated diy_params.json
        '''python evaluate.py --params_path diy_param_json_path'''
        params = utils.Params(args.params_path)
    else:
        # run by python evaluate.py
        if exp_id is not None:
            args.exp_id = exp_id
        cfg = get_config(args, mode='test')
        dic_params = json.loads(json.dumps(cfg))
        obj_params = dictToObj(dic_params)
        params_default_path = os.path.join(obj_params.exp_current_dir, 'params.json')
        model_json_path = os.path.join(obj_params.model_dir, "params.json")
        assert os.path.isfile(
            model_json_path
        ), "No json configuration file found at {}".format(model_json_path)
        params = utils.Params(params_default_path)
        params_model = utils.Params(model_json_path)
        params.update(params_model.dict)
        params.update(obj_params)
        # ipdb.set_trace()

    # Only load model weights
    params.only_weights = True

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

    # Get the logger
    logger = utils.set_logger(os.path.join(params.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    # dataloaders = data_loader_dybev.fetch_dataloader(params)
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        gpu_num = len(params.gpu_used.split(","))
        device_ids = range(gpu_num)
        # device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(
        model=model,
        optimizer=None,
        scheduler=None,
        params=params,
        dataloaders=dataloaders,
        writer=None,
        logger=logger,
    )

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    evaluate(model, manager)


if __name__ == "__main__":

    # for i in range(5, 10):
    #     run_all_exps(i)
    run_all_exps(exp_id=None)
    # run_all_exps(exp_id=1)
