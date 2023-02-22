import cv2
import os
from tqdm import tqdm

def save_img(video_path, out_dir):
    videos = os.listdir(video_path)
    for video_name in tqdm(videos):
        file_name = video_name.split('.')[0]
        folder_name = os.path.join(out_dir, file_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
        vc = cv2.VideoCapture(video_path+video_name)
        c = 0
        rval = vc.isOpened()

        while rval:
            c = c + 1
            rval, frame = vc.read()
            if rval:
                frames = c + 10000
                cv2.imwrite(os.path.join(folder_name, file_name + '_' + str(frames) + '.jpg'), frame)
                # cv2.waitKey(1)
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)

if __name__ == '__main__':
    # path to video folds eg: video_path = './Test/'
    video_root = '/data/xingchen/dataset/AVM/DeepH/Contant-Aware-DeepH-Data/Data'

    video_path = video_root + '/Test/'
    out_dir = '/data/xingchen/dataset/AVM/DeepH/Test'
    save_img(video_path, out_dir)

    # video_path = video_root + '/Train/'
    # save_img(video_path)
