import os
import cv2
import json
import shutil
import imageio


def create_gif(image_list, gif_name, duration=0.5):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)


def extract_overlap_hw():
    info = [
        '5 numbers: h1, h2, w1, w2, angle.',
        'h1:h2, w1:w2, angle_couterclockwise.',
        'Crop first and then rotate.',
    ]
    for inf in info:
        print('\t', inf)
    hw_overlap_fblr = {
        'info': info,
        'input_shape': {
            "front": [1078, 336],
            "back": [1078, 336],
            "left": [1172, 439],
            "right": [1172, 439],
        },
        'input_direction': 'camera',
        'output_shape': [320, 320],
        'output_direction': 'align forward(front)',
        'angle_fblr': [0, 180, 90, -90],
        'instruction': {
            'sequence': '*f *b *l *r',
            'front': {'fl': [], 'fr': [], 'angle': ''},
            'back': {'bl': [], 'br': [], 'angle': ''},
            'left': {'lf': [], 'lb': [], 'angle': ''},
            'right': {'rf': [], 'rb': [], 'angle': ''},
            '5 numbers': ['h1', 'h2', 'w1', 'w2', 'angle'],
            'crop': ['h1:h1', 'w1:w2'],
            'rotate': {
                '90': 'cv2.ROTATE_90_COUNTERCLOCKWISE',
                '-90': 'cv2.ROTATE_90_CLOCKWISE',
                '180': 'cv2.ROTATE_180',
            },
            'usage': 'eval(string)',
        },
        'front': [
            [216 - 200, 316 + 20, 309 - 200, 409 + 20],
            [216 - 200, 316 + 20, 669 - 20, 769 + 200],
            0,
        ],
        'back': [
            [216 - 200, 316 + 20, 669 - 20, 769 + 200],
            [216 - 200, 316 + 20, 309 - 200, 409 + 20],
            180,
        ],
        'left': [
            [319 - 200, 419 + 20, 856 - 20, 956 + 200],
            [314 - 200, 414 + 20, 216 - 200, 316 + 20],
            90,
        ],
        'right': [
            [314 - 200, 414 + 20, 216 - 200, 316 + 20],
            [314 - 200, 414 + 20, 856 - 20, 956 + 200],
            -90,
        ],
    }
    return hw_overlap_fblr


def save_info():
    hw_overlap_info = extract_overlap_hw()
    with open('overlap_info.json', 'w') as f:
        json.dump(hw_overlap_info, f, indent=4)


def unit_test_overlap():
    sv_dir = 'overlap'
    if os.path.exists(sv_dir):
        shutil.rmtree(sv_dir)
    os.makedirs(sv_dir, exist_ok=True)

    src_dir = 'bev/fev2bev'
    front_ori = cv2.imread(f'{src_dir}/front.png')
    back_ori = cv2.imread(f'{src_dir}/back.png')
    left_ori = cv2.imread(f'{src_dir}/left.png')
    right_ori = cv2.imread(f'{src_dir}/right.png')
    print('front & back:', front_ori.shape, back_ori.shape)
    print('left & right:', left_ori.shape, right_ori.shape)
    wh_bev_fblr = {
        "front": [1078, 336],
        "back": [1078, 336],
        "left": [1172, 439],
        "right": [1172, 439],
    }
    print(wh_bev_fblr)
    print()

    # 309,316, 409,316, 336 - 316
    # 856,319, 856,419, 319 - 419
    # 1172 - 856 = 316
    # 409,216, 409,316
    # 956,419, 856,419
    # 10:440

    # f: w=0:409+20, h=0:316+20
    # l: h=10:419+20, w=856-20:956+216
    lf_left = left_ori[319 - 200 : 419 + 20, 856 - 20 : 956 + 200]
    lf_front = front_ori[216 - 200 : 316 + 20, 309 - 200 : 409 + 20]
    lf_left = cv2.rotate(lf_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    txt_info = (cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
    cv2.putText(lf_left, 'left', (200, 200), *txt_info)
    cv2.putText(lf_front, 'front', (200, 150), *txt_info)
    cv2.imwrite(f'{sv_dir}/lf_f.jpg', lf_front)
    cv2.imwrite(f'{sv_dir}/lf_l.jpg', lf_left)
    create_gif([lf_left, lf_front], f'{sv_dir}/lf.gif')
    print('lf_front', lf_front.shape)
    print('lf_left', lf_left.shape)

    # r: 956,314, 956,414  vs  b: 309,216, 409,216
    # r: 856,414, 956,414  vs  b: 409,316, 409,216
    # r: w=856-216:956+20, h=314-309:414+25
    # b: w=:409+25, h=0:316+20
    rb_right = right_ori[314 - 200 : 414 + 20, 856 - 20 : 956 + 200]
    rb_back = back_ori[216 - 200 : 316 + 20, 309 - 200 : 409 + 20]
    rb_right = cv2.rotate(rb_right, cv2.ROTATE_90_CLOCKWISE)
    rb_back = cv2.rotate(rb_back, cv2.ROTATE_180)
    txt_info = (cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
    cv2.putText(rb_right, 'right', (150, 100), *txt_info)
    cv2.putText(rb_back, 'back', (150, 50), *txt_info)
    cv2.imwrite(f'{sv_dir}/rb_r.jpg', rb_right)
    cv2.imwrite(f'{sv_dir}/rb_b.jpg', rb_back)
    create_gif([rb_back, rb_right], f'{sv_dir}/rb.gif')
    print('rb_right', rb_right.shape)
    print('rb_back', rb_back.shape)

    # r: 216,414, 316,414  vs  f: 769,216, 769,316
    # r: 316,414, 316,314  vs  f: 669,316, 769,316
    # r: w=:316+20, h=314-216:414+20
    # f: w=669-20:769+216, h=:316+20
    rf_front = front_ori[216 - 200 : 316 + 20, 669 - 20 : 769 + 200]
    rf_right = right_ori[314 - 200 : 414 + 20, 216 - 200 : 316 + 20]
    rf_right = cv2.rotate(rf_right, cv2.ROTATE_90_CLOCKWISE)
    txt_info = (cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
    cv2.putText(rf_right, 'right', (100, 200), *txt_info)
    cv2.putText(rf_front, 'front', (100, 150), *txt_info)
    cv2.imwrite(f'{sv_dir}/rf_r.jpg', rf_right)
    cv2.imwrite(f'{sv_dir}/rf_b.jpg', rf_front)
    create_gif([rf_right, rf_front], f'{sv_dir}/rf.gif')
    print('rf_front', rf_front.shape)
    print('rf_right', rf_right.shape)

    # l: 216,314, 216,414  vs  b: 769,216, 769,316
    # l: 216,414, 316,414  vs  b: 669,216, 409,316
    # l: w=:316+20, h=314-216:414+20
    # b: w=669-216:769+20, h=:316+20
    lb_left = left_ori[314 - 200 : 414 + 20, 216 - 200 : 316 + 20]
    lb_back = back_ori[216 - 200 : 316 + 20, 669 - 20 : 769 + 200]
    lb_left = cv2.rotate(lb_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lb_back = cv2.rotate(lb_back, cv2.ROTATE_180)
    txt_info = (cv2.FONT_HERSHEY_SIMPLEX, 1, (192, 192, 192), 2)
    cv2.putText(lb_left, 'left', (100, 200), *txt_info)
    cv2.putText(lb_back, 'back', (100, 150), *txt_info)
    cv2.imwrite(f'{sv_dir}/lb_l.jpg', lb_left)
    cv2.imwrite(f'{sv_dir}/lb_b.jpg', lb_back)
    create_gif([lb_left, lb_back], f'{sv_dir}/lb.gif')
    print('lb_left', lb_left.shape)
    print('lb_back', lb_back.shape)


if __name__ == '__main__':

    save_info()
    unit_test_overlap()
