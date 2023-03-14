from .basenet import Net
from .osnet import OSNet
from .swin_multi import SwinTransformer
from .pix2pixSTN import Pix2PixSTN

def fetch_net(params):

    if params.net_type == "basic":
        net = Net(params.crop_size, params.use_LRR)
    elif params.net_type == 'osnet':
        net = OSNet(params, backbone=SwinTransformer, init_mode = 'swin')
    elif params.net_type == "pix2pix_stn":
        net = Pix2PixSTN(params.inch, params.outch)
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return net
