import os
import sys
import cv2
import ipdb
import random
import numpy as np
from easydict import EasyDict
from tqdm import tqdm


class PairMaker:
    def __init__(self):
        super().__init__()
        self.vid_list = [
            '20230227113020', # 1
            '20230227113709',
            '20230227113746',
            '20230227113817',
            '20230227113856', # 5
            '20230227113937',
            '20230227114006',
            '20230227114024',
            '20230227114043',
            '20230227114111', # 10
            '20230227114137',
            '20230227114205',
            '20230227114230',
            '20230227114304',
            '20230227114328',
            '20230227114403',
            '20230227114457', # 17
            ]
        self.bev_ind = ['f_l', 'f_r', 'b_l', 'b_r', 'l_f', 'l_b', 'r_f', 'r_b',]
        '''
        front/20221205141455_front_00001_p0011_0.jpg left/20221205141455_left_00001_p0011_0.jpg 
        back/20221205141455_back_00001_p0010_1.jpg right/20221205141455_right_00001_p0010_1.jpg 
        left/20221205141455_left_00001_p0011_1.jpg back/20221205141455_back_00001_p0010_0.jpg 
        right/20221205141455_right_00001_p0010_0.jpg front/20221205141455_front_00001_p0011_1.jpg
        '''
        # 4304 0:20 frames
        
    def every_unaligned_bad_case(self):
        des = {'20230227114230': 'lan gan (lf+fl)', 
               '20230227114205': 'jie ti, lv hua(rf+fr)',
               '20230227114137': '!',
               '20230227113746': 'zhuan, but distorted car',
               '20230227113020': 'zhuan, tree, car',
               '20230227113709': 'zhuan, tree, car',
               }
        
        return
    
    def make_pair(self):
        # vid = '20230227114304'
        # vid = '20230227114457'
        # vid = '20230227114328'
        # vid = '20230227113937'
        vid = '20230227113856' # zhuan
        # sign = [['f_l', 'l_f'], ['b_l', 'l_b'], ['r_b', 'b_r']]
        # sign = [['f_l', 'l_f'], ['r_b', 'b_r']]
        # sign = [['f_l', 'l_f']]
        # sign = [['l_b', 'b_r'], ['r_b', 'b_r']]
        sign = [['f_l', 'l_f'], ['l_b', 'b_r']]
        bp = '/home/data/lwb/data/dybev/b16'
        svp = bp + f'/train_{vid}.txt'
        if os.path.exists(svp):
            os.remove(svp)
        f = open(svp, 'a+')
        # fs, fe = 0, 20
        # fs, fe = 100, 110
        # fs, fe = 0, 13
        fs, fe = 0, 30
        for i in range(fs, fe + 1):
            index = [[i, i], [i, i+1]]
            for k1 in index:
                for k2 in sign:
                    n1 = f'{vid}/{vid}_{k1[0]}_{k2[0]}.jpg'
                    n2 = f'{vid}/{vid}_{k1[1]}_{k2[1]}.jpg'
                    f.write(n1 + ' ' + n2 + '\n')
        f.close()
        pass
    
    def merge_txt(self):
        bp = '/home/data/lwb/data/dybev/b16'
        txt_list = [
            'train_20230227113856.txt',
            'train_20230227113937.txt',
            'train_20230227114304.txt',
            'train_20230227114328.txt',
            'train_20230227114457.txt',
        ]
        name_list = []
        for txt in txt_list:
            with open(f'{bp}/{txt}', 'r') as f:
                name_list += f.readlines()
        random.shuffle(name_list)
        p1 = f'{bp}/train.txt'
        p2 = f'{bp}/test.txt'
        for p in [p1, p2]:
            if os.path.exists(p):
                os.remove(p)
        num = len(name_list)
        with open(p1, 'a+') as f:
            for name in name_list[:int(0.9*num)]:
                f.write(name)
        with open(p2, 'a+') as f:
            for name in name_list[int(0.9*num):]:
                f.write(name)
    
    
if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    pm = PairMaker()
    # pm.make_pair()
    pm.merge_txt()
    pass