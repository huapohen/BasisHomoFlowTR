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
        
        self.vids_info = {
            '20230227113020': [[], []],
            '20230227113709': [[0, 30], [['f_l', 'l_f']]],
            '20230227113746': [[0, 100], [['f_r', 'r_f'], ['b_r', 'r_b']]],
            '20230227113817': [[], []],
            '20230227113856': [[], []],
            '20230227113937': [[130, 160], [['f_l', 'l_f'], ['b_l', 'l_b']]],
            '20230227114006': [[150, 240], [['b_r', 'r_b']]],
            '20230227114024': [[0, 30], [['f_r', 'r_f']]],
            '20230227114043': [[0, 20], [['f_r', 'r_f']]],
            '20230227114111': [[22, 90], [['b_l', 'l_b']]],
            '20230227114137': [[0, 20], [['f_l', 'l_f']]],
            '20230227114205': [[261, 273], [['f_l', 'l_f'], ['f_r', 'r_f']]],
            '20230227114230': [[], []], # lan gan !!!  very difficult
            '20230227114304': [[0, 20], [['b_l', 'l_b'], ['b_r', 'r_b']]],
            # 4328 heng tiao xiao shi , and 4403
            '20230227114328': [[0, 20], [['f_l', 'l_f'], ['b_l', 'l_b'], ['b_r', 'r_b']]],
            '20230227114403': [[10, 30], [['b_r', 'r_b']]],
            '20230227114457': [[0, 35], [['f_r', 'r_f'], ['b_r', 'r_b']]],
            '20230227114457': [[35, 50], [['b_r', 'r_b']]], # double
            ##  ['f_l', 'l_f'], ['f_r', 'r_f'], ['b_l', 'l_b'], ['b_r', 'r_b']
            '20230302155357': [[100, 150], [['f_r', 'r_f']]], 
            # ying zi, unaligned, ? he 5444 yiyang, you ban ma xian # choose
            '20230302155444': [[0, 74], [['f_l', 'l_f'], ['b_r', 'r_b']]],
            '20230302155444': [[0, 40], [['b_l', 'l_b'], ['b_r', 'r_b']]],
            '20230302155508': [[0, 15], [['f_l', 'l_f'], ['f_r', 'r_f'], ['b_r', 'r_b']]],
            '20230302155521': [[80, 120], [['f_l', 'l_f'], ['f_r', 'r_f'], ['b_r', 'r_b']]],
            '20230302155534': [[0, 70], [['f_l', 'l_f'], ['f_r', 'r_f'], ['b_r', 'r_b']]],
            '20230302155547': [[0, 80], [['f_l', 'l_f'], ['f_r', 'r_f'], ['b_l', 'l_b'], ['b_r', 'r_b']]],
            '20230302155600': [[0, 20], [['f_l', 'l_f'], ['f_r', 'r_f'], ['b_r', 'r_b']]],
            '20230302155613': [[20, 35], [['f_r', 'r_f']]],
            '20230302155613': [[150, 160], [['f_l', 'l_f'], ['f_r', 'r_f']]],
            '20230302155613': [[180, 188], [['f_l', 'l_f'], ['b_l', 'l_b'], ['b_r', 'r_b']]],
            '20230302155626': [[10, 50], [['f_l', 'l_f']]],
            '20230302155638': [[160, 180], [['f_r', 'r_f'], ['b_r', 'r_b']]],
            '20230302155657': [[40, 100], [['f_l', 'l_f'], ['f_r', 'r_f'], ['b_l', 'l_b'], ['b_r', 'r_b']]],
            '20230302155719': [[], []],
            '20230302155732': [[], []],
            '20230302155745': [[], []],
            '20230302155758': [[], []],
            '20230302155811': [[], []],
            '20230302155823': [[], []],
            '20230302155838': [[], []],
            '20230302160235': [[], []],
            '20230302160310': [[], []],
            '20230302160343': [[], []],
            '20230302160416': [[], []],
            '20230302160449': [[], []],
            '20230302160522': [[], []],
            '20230302160555': [[], []],
        }
        
        # ['f_l', 'l_f'], ['f_r', 'r_f'], ['b_l', 'l_b'], ['b_r', 'r_b']
        self.pair_id = {'f_l': 'p1', 'f_r': 'p2', 'b_l': 'p3', 'b_r': 'p4'}
                
    
    def make_pair(self):
        self.valid_list = []
        for vid, info in self.vids_info.items():
            if len(info[0]) == 0:
                continue
            self.valid_list.append(vid)
            bp = '/home/data/lwb/data/dybev/b16'
            svp = bp + f'/train_{vid}.txt'
            if os.path.exists(svp):
                os.remove(svp)
            f = open(svp, 'a+')
            fs, fe = info[0]
            sign = info[1]
            for i in range(fs, fe + 1):
                index = [[i, i], [i, i+1]]
                for k1 in index:
                    for k2 in sign:
                        n1 = f'{vid}/{vid}_{k1[0]}_{self.pair_id[k2[0]]}-{k2[0]}.jpg'
                        n2 = f'{vid}/{vid}_{k1[1]}_{self.pair_id[k2[0]]}-{k2[1]}.jpg'
                        f.write(n1 + ' ' + n2 + '\n')
        f.close()
        pass
    
    def merge_txt(self):
        bp = '/home/data/lwb/data/dybev/b16'
        name_list = []
        for txt in self.valid_list:
            with open(f'{bp}/train_{txt}.txt', 'r') as f:
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
    pm.make_pair()
    pm.merge_txt()
    pass