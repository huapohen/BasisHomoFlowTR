import os
import sys
import ipdb
import json
import shutil
import numpy as np
from tqdm import tqdm


class MakePair:
    '''same frame version'''

    def __init__(self, mode='train', source='generate'):
        version = 'v7'
        base_path = f'/home/data/lwb/data/dybev/{version}'
        self.src_path = f'{base_path}/{mode}/{source}'
        self.pair_dir = self.src_path + '/pair'
        if os.path.exists(self.pair_dir):
            shutil.rmtree(self.pair_dir)
        os.makedirs(self.pair_dir)
        self.cam_list = ['front', 'back', 'left', 'right']
        with open(self.src_path + f'/name_list.txt') as f:
            self.name_list = f.readlines()
        self.sv_txt_name_fblr = 'all_fblr_pair_names_list.txt'
        self.sv_txt_name_suff = 'pair_names_list.txt'
        np.random.seed(1)
        self.pert_number = 10 if source == 'generate' else 1
        self.cam_align = {
            'front': ['left', 'right'],
            'back': ['left', 'right'],
            'left': ['front', 'back'],
            'right': ['front', 'back'],
        }
        # f 0-> l 1-> b 1-> r 0->
        # f(0,1), b(2,3), l(4,5), r(6,7)
        # [0,3,5,6]
        # f 1-> r 1-> b 0-> l 0->
        # [1,7,2,4]
        # self.pair_idx_list = [[0, 3, 5, 6], [1, 7, 2, 4]]
        self.pair_idx_list = [[0, 3, 5, 6], [1, 2, 4, 7]]

    def pert_to_ori(self):
        pass

    def pert_to_pert_two_camera(self):
        for cam1 in self.cam_list:
            fw = open(self.pair_dir + f'/{cam1}_{self.sv_txt_name_suff}', 'a+')
            iter_max = len(self.name_list)
            k = 0
            with tqdm(total=len(self.name_list)) as t:
                for name in self.name_list:
                    name = name.rsplit()[0]
                    name1 = name.replace('front', cam1)
                    for i, cam2 in enumerate(self.cam_align[cam1]):
                        name2 = name.replace('front', cam2)
                        pidx = name1.split('_')[-1]
                        jj = np.random.choice(self.pert_number, 1)[0]
                        name22 = name2.replace(pidx, f'p{jj:04d}')
                        cam2_id_for_pair = self.cam_align[self.cam_align[cam1][i]]
                        ii = 0 if cam2_id_for_pair[0] == cam1 else 1
                        sample_pair = (
                            f'{cam1}/{name1}_{i}.jpg'
                            + " "
                            + f'{cam2}/{name22}_{ii}.jpg'
                        )
                        suff = '\n'
                        if k == iter_max - 1 and i == 1:
                            suff = ''
                        fw.write(sample_pair + suff)
                        # print(sample_pair)
                    t.set_description(desc=f'{cam1} >>> ')
                    t.update()
                    k += 1
                    # ipdb.set_trace()
            fw.close()

    def ori_to_ori(self):
        pass

    def pert_to_pert_four_camera(self):
        fw_all = open(f'{self.pair_dir}/{self.sv_txt_name_fblr}', 'a+')
        iter_max = len(self.name_list)
        k = 0
        with tqdm(total=iter_max) as t:
            for name in self.name_list:
                sample_pair = []
                name = name.rsplit()[0]
                pidx = name.split('_')[-1]
                for i, cam in enumerate(self.cam_list):
                    j = np.random.choice(self.pert_number, 1)[0]
                    name2 = name.replace('front', cam)
                    name2 = name2.replace(pidx, f'p{j:04d}')
                    sample_pair.append(f'{cam}/{name2}_0.jpg')
                    sample_pair.append(f'{cam}/{name2}_1.jpg')
                for i in range(2):
                    pair_str = []
                    for idx in self.pair_idx_list[i]:
                        pair_str.append(sample_pair[idx])
                    suff = '\n'
                    if k == iter_max - 1 and i == 1:
                        suff = ''
                    fw_all.write(' '.join(pair_str) + suff)
                    # print(pair_str)
                    # ipdb.set_trace()
                t.set_description()
                t.update()
                k += 1
        fw_all.close()

if __name__ == '__main__':

    mode_list = ['train', 'test']
    source_list = ['bev', 'generate']
    # source_list = ['generate']
    # source_list = ['bev']

    for mode in mode_list:
        for source in source_list:
            print(mode, source)
            mp = MakePair(mode, source)
            mp.pert_to_pert_two_camera()
            mp.pert_to_pert_four_camera()
