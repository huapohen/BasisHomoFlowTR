import os
import sys
import ipdb
from easydict import EasyDict
from yacs.config import CfgNode as CN


def train_config(cfg):
    cfg.exp_id = 6
    cfg.gpu_used = '7'
    cfg.num_workers = 8
    cfg.train_batch_size = 8
    cfg.train_data_ratio = 0.1
    # cfg.is_add_lrr_module = False
    cfg.is_add_lrr_module = True
    # cfg.loss_func_type = 'photo'
    # cfg.loss_func_type = 'feature'
    cfg.loss_func_type = 'all'
    cfg.is_exp_rm_protect = False
    cfg.is_continue_train = False
    # cfg.exp_description = 'only_photo_loss'
    # cfg.exp_description = 'only_feature_loss'
    # cfg.exp_description = 'all_loss'
    # cfg.exp_description = 'photo_loss + lrr'
    # cfg.exp_description = 'feature_loss + lrr'
    cfg.exp_description = 'all_loss + lrr'
    cfg = continue_train(cfg)
    # cfg.gpu_used = '0_1_2_3_4_5_6_7' # use 8 GPUs
    return cfg


def test_config(cfg, args=None):

    cfg.exp_id = 1
    cfg.gpu_used = '6'
    cfg.eval_batch_size = 8
    cfg.is_exp_rm_protect = False
    cfg.dataset_type = "test"
    # cfg.eval_visualize_save = False
    # cfg.restore_file = 'test_model_best.pth'
    cfg.restore_file = "model_latest.pth"

    if 'exp_id' in vars(args):
        cfg.exp_id = args.exp_id

    return cfg


def continue_train(cfg):
    if cfg.is_continue_train:
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
    cfg.exp_root = 'experiments'
    cfg.exp_name = 'baseshomo'
    exp_dir = os.path.join(cfg.exp_root, cfg.exp_name)
    cfg.model_dir = os.path.join(exp_dir, f"exp_{cfg.exp_id}")
    cfg.tb_path = os.path.join(exp_dir, 'tf_log', f'exp_{cfg.exp_id}')
    if (
        cfg.is_exp_rm_protect
        and os.path.exists(cfg.model_dir)
        and not cfg.is_continue_train
    ):
        print("Existing experiment, exit.")
        sys.exit()
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
