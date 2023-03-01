"""Perform hyperparemeters search"""

import os
import sys
import ipdb
import time
import shutil
import argparse
import itertools
import collections
from easydict import EasyDict
from subprocess import check_call

from common import utils
from experiment_dispatcher import dispatcher, tmux

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument(
    '--exp_current_dir', default='experiments',
    help='Directory containing params.json',
)
parser.add_argument(
    '--exp_root_dir', default='/home/data/lwb/experiments',
    help='Directory containing params.json',
)


def launch_training_job(
    exp_current_dir,
    exp_root_dir,
    exp_name,
    session_name,
    param_pool_dict,
    device_used,
    params,
    start_id=0,
):
    # Partition tmux windows automatically
    tmux_ops = tmux.TmuxOps()
    # Combining hyper-parameters and experiment ID automatically
    task_manager = dispatcher.Enumerate_params_dict(
        task_thread=0, if_single_id_task=True, **param_pool_dict
    )

    num_jobs = len([v for v in itertools.product(*param_pool_dict.values())])
    num_device = len(device_used)
    exp_cmds = []

    assert num_jobs == num_device

    for job_id in range(num_jobs):
        device_id = device_used[job_id % num_device]
        hyper_params = task_manager.get_thread(ind=job_id)[0]
        exp_id = job_id + start_id
        job_name = 'exp_{}'.format(exp_id)
        # ipdb.set_trace()
        for k in hyper_params.keys():
            params.dict[k] = hyper_params[k]

        params.dict['model_dir'] = os.path.join(exp_root_dir, exp_name, job_name)
        model_dir = params.dict['model_dir']

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Write parameters in json file
        json_path = os.path.join(model_dir, 'params.json')
        params.save(json_path)
        # ipdb.set_trace()

        # Launch training with this config
        tb_path = os.path.join(exp_root_dir, exp_name, "tf_log", job_name)
        cmd = (
            'python train.py '
            f'--gpu_used {device_id} '
            f'--exp_current_dir {exp_current_dir} '
            f'--exp_root_dir {exp_root_dir} '
            f'--model_dir {model_dir} '
            f'--exp_name {exp_name} '
            f'--exp_id {exp_id} '
            f'--tb_path {tb_path}'
        )

        exp_cmds.append(cmd)

    if_serial = num_jobs > num_device
    if if_serial:
        print("run task serially! ")
    else:
        print("run task parallelly! ")

    tmux_ops.run_task(
        exp_cmds, task_name=exp_name, session_name=session_name, start_id=start_id
    )

    if True:
        print(
            '\n terminal: \n'
            f'\t `tmux ls`: find all running tasks \n'
            f'\t `tmux attach -t {start_id}` to enter the tmux window \n'
            f'\t `Ctrl + C`: interupt the task \n'
            f'\t `Ctrl + D`: end the current subwindow \n'
            f'\t `Ctrl + B D`: exit the window and keep the taks running\n'
        )
        time.sleep(5)
        check_call(f'tmux ls', shell=True)
        time.sleep(3)
        check_call(f'tmux attach -t {start_id}', shell=True)


def experiment():
    # Load the "reference" parameters from experiment_dir json file
    args = parser.parse_args()
    exp_root_dir = args.exp_root_dir
    exp_current_dir = args.exp_current_dir
    json_path = os.path.join(exp_current_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    default_params = utils.Params(json_path)

    exp_name = 'baseshomo'
    exp_start_id = 8
    session_name = str(exp_start_id)  # tmux session name, need pre-create
    param_pool_dict = collections.OrderedDict()
    device_used = collections.OrderedDict()
    device_used = ['4', '5', '6', '7']
    param_pool_dict["train_batch_size"] = [16]
    param_pool_dict["eval_batch_size"] = [16]
    param_pool_dict["num_workers"] = [8]
    param_pool_dict['train_data_ratio'] = [1.0]
    param_pool_dict['train_data_dir'] = ['/home/data/lwb/data/dybev/v9']
    param_pool_dict['loss_type'] = ['front_first', 'back_first', 
                                    'left_first', 'right_first']
    param_pool_dict['exp_description'] = [
        f'exp_8-11: dataset v9, loss_type = [f, b, l, r]'
        ]

    # '0', '1', '2', '3', '4', '5', '6', '7'
    # device_used = ['6']
    # device_used = ['0', '0']
    # device_used = ['0_1_2_3'] # for one experiment
    # device_used = ['0', '1', '2', '3'] # for four experiments

    for id in range(exp_start_id, exp_start_id + len(device_used)):
        try:  # cover exp
            bp = os.path.join(exp_root_dir, exp_name)
            shutil.rmtree(os.path.join(bp, f"exp_{id}"))
            path = os.path.join(bp, "tf_log", f"{exp_name}_exp_{id}")
            shutil.rmtree(path)
        except:
            pass

    launch_training_job(
        exp_current_dir,
        exp_root_dir,
        exp_name,
        session_name,
        param_pool_dict,
        device_used,
        default_params,
        exp_start_id,
    )


if __name__ == "__main__":
    experiment()
