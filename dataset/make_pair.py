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
        src_path = f'{base_path}/{mode}/{source}'
        self.mode = mode
        self.pair_dir = src_path + '/pair'
        # if os.path.exists(self.pair_dir):
            # shutil.rmtree(self.pair_dir)
        os.makedirs(self.pair_dir, exist_ok=True)
        self.cam_list = ['front', 'back', 'left', 'right']
        with open(src_path + f'/name_list.txt') as f:
            self.name_list = f.readlines()
        self.sv_txt_name_suff = 'pair_names_list.txt'
        np.random.seed(1)
        self.pert_number = 10 if source == 'generate' else 1
        if mode == 'test':
            self.pert_number = 2
            self.pert_number_train = 10
        self.cam_align = {
            'front': ['left', 'right'],
            'back': ['left', 'right'],
            'left': ['front', 'back'],
            'right': ['front', 'back'],
        }
        # f(0,1), b(2,3), l(4,5), r(6,7)
        # f 0-> l 1-> b 1-> r 0->
        # f 1-> r 1-> b 0-> l 0->
        # fblr (f0,l0), (b0,l1), (r0,f1), (r1,b1)
        #   => (f1,r0), (b0,l1), (l0,f0), (r1,b1)
        #   or (f0,l0), (b1,r1), (l1,b0), (r0,f1)
        pair_idx_order1 = [1,6,2,5,4,0,7,3]
        pair_idx_order2 = [0,4,3,7,5,2,6,1]
        self.pair_idx_order = pair_idx_order2

    def pert_to_pert_two_camera(self):
        for cam1 in self.cam_list:
            cam_pair_path = self.pair_dir + f'/{cam1}_{self.sv_txt_name_suff}'
            if os.path.exists(cam_pair_path):
                os.remove(cam_pair_path)
            fw = open(cam_pair_path, 'a+')
            with tqdm(total=len(self.name_list)) as t:
                for name in self.name_list:
                    name = name.rsplit()[0]
                    name1 = name.replace('front', cam1)
                    for i, cam2 in enumerate(self.cam_align[cam1]):
                        name2 = name.replace('front', cam2)
                        pidx = name1.split('_')[-1]
                        jj = np.random.choice(self.pert_number, 1)[0]
                        if self.mode == 'test':
                            jj += self.pert_number_train
                        name22 = name2.replace(pidx, f'p{jj:04d}')
                        cam2_id_for_pair = self.cam_align[self.cam_align[cam1][i]]
                        ii = 0 if cam2_id_for_pair[0] == cam1 else 1
                        sample_pair = (
                            f'{cam1}/{name1}_{i}.jpg'
                            + " "
                            + f'{cam2}/{name22}_{ii}.jpg'
                        )
                        fw.write(sample_pair + '\n')
                        # print(sample_pair)
                    t.set_description(desc=f'{cam1} >>> ')
                    t.update()
                    # ipdb.set_trace()
            fw.close()

    def pert_to_pert_four_camera(self):
        fblr_pair_path = f'{self.pair_dir}/all_fblr_pair_names_list.txt'
        if os.path.exists(fblr_pair_path):
            os.remove(fblr_pair_path)
        fw_all = open(fblr_pair_path, 'a+')
        with tqdm(total=len(self.name_list)) as t:
            for name in self.name_list:
                sample_pair = []
                name = name.rsplit()[0]
                pidx = name.split('_')[-1]
                for i, cam in enumerate(self.cam_list):
                    j = np.random.choice(self.pert_number, 1)[0]
                    if self.mode == 'test':
                        j += self.pert_number_train
                    name2 = name.replace('front', cam)
                    name2 = name2.replace(pidx, f'p{j:04d}')
                    sample_pair.append(f'{cam}/{name2}_0.jpg')
                    sample_pair.append(f'{cam}/{name2}_1.jpg')
                pair_str = []
                for idx in self.pair_idx_order:
                    pair_str.append(sample_pair[idx])
                fw_all.write(' '.join(pair_str) + '\n')
                # print(pair_str)
                # ipdb.set_trace()
                t.set_description()
                t.update()
        fw_all.close()
        
    def pert_to_ori(self):
        pass
    
    def ori_to_ori(self):
        pass
    

if __name__ == '__main__':

    mode_list = ['train', 'test']
    # mode_list = ['test']
    source_list = ['bev', 'generate']
    # source_list = ['generate']
    # source_list = ['bev']

    for mode in mode_list:
        for source in source_list:
            print(mode, source)
            mp = MakePair(mode, source)
            mp.pert_to_pert_two_camera()
            mp.pert_to_pert_four_camera()
