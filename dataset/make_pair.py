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
        base_path = '/home/data/lwb/data/dybev/v6'
        self.src_path = f'{base_path}/{mode}/{source}'
        self.pair_dir = self.src_path + '/pair'
        self.cam_list = ['front', 'back', 'left', 'right']
        with open(self.src_path + f'/name_list.txt') as f:
            self.name_list = f.readlines()
        self.sv_txt_name = 'all_fblr_pair_names_list.txt'
        self.sv_txt_name_suff = 'pair_names_list.txt'
        np.random.seed(1)
        self.pert_number = 15 if source == 'generate' else 1
        self.cam_align = {
            'front': ['left', 'right'],
            'back': ['left', 'right'],
            'left': ['front', 'back'],
            'right': ['front', 'back'],
        }

    def pert_to_ori(self):
        pass

    def pert_to_pert(self):
        if os.path.exists(self.pair_dir):
            shutil.rmtree(self.pair_dir)
        os.makedirs(self.pair_dir)
        k = self.pert_number
        fw_all = open(f'{self.pair_dir}/{self.sv_txt_name}', 'a+')
        for cam1 in self.cam_list:
            fw = open(self.pair_dir + f'/{cam1}_{self.sv_txt_name_suff}', 'a+')
            with tqdm(total=len(self.name_list)) as t:
                for name in self.name_list:
                    name = name.rsplit()[0]
                    name1 = name.replace('front', cam1)
                    for i, cam2 in enumerate(self.cam_align[cam1]):
                        name2 = name.replace('front', cam2)
                        for j in range(k):
                            # ipdb.set_trace()
                            name11 = name1.replace('p0000', f'p{j:04d}')
                            jj = np.random.choice(k, 1)[0]
                            name22 = name2.replace('p0000', f'p{jj:04d}')
                            cam2_id_for_pair = self.cam_align[self.cam_align[cam1][i]]
                            ii = 0 if cam2_id_for_pair[0] == cam1 else 1
                            sample_pair = (
                                f'{cam1}/{name11}_{i}.jpg'
                                + " "
                                + f'{cam2}/{name22}_{ii}.jpg'
                            )
                            fw.write(sample_pair + '\n')
                            fw_all.write(sample_pair + '\n')

                    t.set_description(desc=f'{cam1} >>> ')
                    t.update()
            fw.close()
        fw_all.close()

    def ori_to_ori(self):
        pass


if __name__ == '__main__':

    mode_list = ['train', 'test']
    source_list = ['bev', 'generate']
    # source_list = ['generate']
    # source_list = ['bev']

    for mode in mode_list:
        for source in source_list:
            print(mode, source)
            mp = MakePair(mode, source)
            mp.pert_to_pert()
