import os
import sys
import ipdb
from easydict import EasyDict
from yacs.config import CfgNode as CN


def train_config(cfg):
    cfg.exp_id = 41
    cfg.exp_description = f' exp_{cfg.exp_id}: '
    cfg.exp_description += ' offset: b16 + 320x576 + 300 epoch + 1e-3 + 0.97 + img_balance '
    # cfg.exp_description += ' outdoor b16 img_balance + add edge loss + nature=0.02 '
    cfg.gpu_used = '2'
    cfg.train_data_ratio = 1.0
    cfg.is_img_balance = True
    cfg.eval_freq = 20
    # cfg.is_add_edge_loss = True
    # cfg.is_add_photo_loss = False
    # cfg.is_include_dataset_nature = True
    # cfg.is_only_nature_dataset = True
    cfg.dataset_nature_ratio = 0.02
    cfg.num_workers = 8
    cfg.crop_size_outdoor = [320, 576]
    cfg.rho_dybev = 16
    # cfg.learning_rate = 1e-4
    # cfg.gamma = 0.8
    # cfg.num_epochs = 20
    # cfg.set_name = 'b16_seq'
    cfg.set_name = 'b16'
    # cfg.set_name = 'b16_cp'
    # cfg.set_name = 'b07'""
    cfg.train_data_dir = f'/home/data/lwb/data/dybev/{cfg.set_name}'
    cfg.test_data_dir = cfg.train_data_dir
    cfg.camera_list = ['front']
    cfg.train_batch_size = 16
    # cfg.eval_batch_size = 2
    # cfg.forward_version = 'basis'
    cfg.forward_version = 'offset'
    # cfg.pair_loss_type = 'front_first'
    cfg = continue_train(cfg)
    # cfg.gpu_used = '0_1_2_3_4_5_6_7' # use 8 GPUs
    return cfg


def test_config(cfg, args=None):

    cfg.exp_id = 38
    # cfg.gpu_used = '6'
    cfg.gpu_used = '5'
    cfg.test_pipeline_dataset = 'test'
    cfg.test_pipeline_mode = 'resize'
    # cfg.calc_gt_photo_loss = True
    # cfg.set_name = 'b16_cp'
    cfg.set_name = 'b16'
    cfg.is_add_ones_mask = True
    # cfg.is_img_balance = False
    cfg.is_img_balance = True
    # cfg.crop_size_outdoor = [192, 192]
    cfg.is_save_intermediate_results = True
    # cfg.loss_sequence = '21' # img1 - img2_warp
    # cfg.loss_sequence = '12' # img2 - img1_warp
    cfg.loss_sequence = 'all'
    # cfg.crop_size_outdoor = [320, 576]
    # cfg.rho_dybev = 16
    # cfg.is_include_dataset_nature = True
    cfg.is_include_dataset_nature = False
    # 推理时开启photo_loss
    cfg.is_add_photo_loss = True
    # cfg.is_only_nature_dataset = True
    cfg.save_iteration = 1
    cfg.is_save_gif = True
    cfg.eval_batch_size = 1
    # cfg.eval_batch_size = 16
    cfg.num_workers = 0
    # cfg.num_workers = 8
    # cfg.is_vis_and_exit = True
    # cfg.is_debug_dataloader = True
    # cfg.eval_visualize_save = False
    cfg.dataset_type = "test"
    cfg.restore_file = "model_latest.pth"

    if 'exp_id' in vars(args):
        cfg.exp_id = args.exp_id

    return cfg


def continue_train(cfg):
    if 'is_continue_train' in vars(cfg) and cfg.is_continue_train:
        # cfg.restore_file = 'test_model_best.pth'
        cfg.restore_file = "model_latest.pth"
        cfg.only_weights = True
        # cfg.only_weights = False
    return cfg


def common_config(cfg):
    if "linux" in sys.platform:
        cfg.data_dir = "/home/data/lwb/data/dybev"
    else:  # windows
        cfg.data_dir = ""
    if not os.path.exists(cfg.data_dir):
        raise ValueError
    cfg.exp_root_dir = '/home/data/lwb/experiments'
    cfg.exp_current_dir = 'experiments'
    cfg.exp_name = 'baseshomo'
    cfg.extra_config_json_dir = os.path.join(cfg.exp_current_dir, 'config')
    exp_dir = os.path.join(cfg.exp_root_dir, cfg.exp_name)
    cfg.model_dir = os.path.join(exp_dir, f"exp_{cfg.exp_id}")
    cfg.tb_path = os.path.join(exp_dir, 'tf_log', f'exp_{cfg.exp_id}')
    if 'restore_file' in cfg and cfg.restore_file is not None:
        cfg.restore_file = os.path.join(cfg.model_dir, cfg.restore_file)
    if (
        'is_exp_rm_protect' in vars(cfg)
        and cfg.is_exp_rm_protect
        and os.path.exists(cfg.model_dir)
        and not cfg.is_continue_train
    ):
        print("Existing experiment, exit.")
        sys.exit()
    if 'is_debug_dataloader' in dictToObj(cfg) and cfg.is_debug_dataloader:
        cfg.num_workers = 0
        
    return cfg


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


def get_config(args=None, mode='train'):
    """Get a yacs CfgNode object with debug params values."""
    cfg = CN()

    assert mode in ['train', 'test', 'val', 'evaluate']

    if mode == 'train':
        cfg = train_config(cfg)
    else:
        cfg = test_config(cfg, args)

    cfg = common_config(cfg)

    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = cfg.clone()
    config.freeze()

    return config


'''
cfg = get_config(None, 'train'))
dic = json.loads(json.dumps(cfg))
'''
