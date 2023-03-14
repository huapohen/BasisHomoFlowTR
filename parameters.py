import os
import sys
import ipdb
from easydict import EasyDict
from yacs.config import CfgNode as CN


def train_config(cfg):
    cfg.exp_id = 6
    cfg.is_exp_rm_protect = True
    # cfg.lr_baseline = 1e-16
    cfg.optimizer = 'adamw'
    cfg.num_epochs = 40
    cfg.learning_rate = 1e-3
    cfg.lr_baseline = cfg.learning_rate
    # cfg.lr_baseline = 1e-4
    cfg.gamma = 0.80
    cfg.weight_decay = 0.05
    cfg.gpu_used = '1'
    cfg.train_ratio = 0.1
    # cfg.is_share_backbone = False
    cfg.is_share_backbone = True
    cfg.is_share_feature = False
    # cfg.encoder_layers = 1
    cfg.encoder_layers = 2
    # cfg.exp_description = f' exp_{cfg.exp_id}: backbone=origin, share_module=new, ratio=0.1 '
    cfg.exp_description = f' exp_{cfg.exp_id}: origin encoder_layers=2 '
    # cfg.optimizer = 'adam' # 先用adam快速实验，最后切adamw刷点
    cfg.exp_description += ' '
    # cfg.proj_hidden_chans = 16
    # cfg.proj_out_chans = 1
    # cfg.backbone_type = 'swin'
    cfg.backbone_type = 'origin'
    # cfg.backbone_type = 'light'
    # cfg.share_module = 'unet'
    cfg.share_module = 'origin'
    # cfg.share_module = 'new'
    cfg.is_load_pretrained = False
    # cfg.is_load_pretrained = True
    cfg.eval_freq = 1
    cfg.sv_epoch_sequence = 1
    # cfg.swin_pth_name = 'swin_base_patch4_window12_384.pth'
    cfg.is_save_best_checkpoint = True
    # cfg.is_save_best_checkpoint = False
    # cfg.basesnet_new = False
    # cfg.basesnet_ratio = 1
    # cfg.basesnet_dim = 64
    # cfg.basesnet_layers = [3, 4, 6, 3]
    # cfg.basesnet_is_attn = False

    cfg = continue_train(cfg)
    return cfg


def test_config(cfg, args=None):

    cfg.exp_id = 2
    cfg.gpu_used = '0'
    cfg.is_calc_point_err = False
    # cfg.is_calc_point_err = True
    cfg.is_resize_mode = False
    # cfg.is_resize_mode = True
    cfg.is_save_gif = True
    # cfg.save_iteration = 10
    cfg.save_iteration = 1000
    cfg.is_exp_rm_protect = False
    cfg.dataset_type = "test"
    # cfg.eval_visualize_save = False
    # cfg.restore_file = 'test_model_best.pth'
    cfg.restore_file = "model_latest.pth"

    if 'exp_id' in vars(args):
        cfg.exp_id = args.exp_id

    return cfg


def sample():
    '''
    车: 00000104_10010.jpg_00000104_10017.jpg.npy_42.06744161391.gif
    车: 00000104_10036.jpg_00000104_10042.jpg.npy_21.89122305297559.gif
    水: 00000181_10046.jpg_00000181_10049.jpg.npy_3.6407653322951847.gif
    '''
    return


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
    cfg.exp_name = 'homotr'
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
    if 'is_debug_dataloader' in vars(cfg) and cfg.is_debug_dataloader:
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
