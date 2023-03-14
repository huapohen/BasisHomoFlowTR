import os
import sys
import glob
import ipdb

def filter_data_list(data_dirs, out_file):
    with open(out_file, 'w') as f:
        for data_dir in data_dirs:
            avm_files = glob.glob(os.path.join(data_dir, '*avm.jpg'))
            for file in avm_files:
                perfix = file[:file.rfind('_')]
                f.write(perfix + '\n')

if __name__ == "__main__":
    # ipdb.set_trace()
    data_dirs = ['/data/xingchen/dataset/AVM/b16_train/train_augment']
    filter_data_list(data_dirs, '/data/xingchen/dataset/AVM/b16_train/train_aug_list.txt')
