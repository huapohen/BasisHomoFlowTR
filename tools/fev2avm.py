import cv2
import ipdb
import numpy as np
import math
import json
from copy import deepcopy
import os

class Fev2AVM(object):
    def __init__(self, config_json, camera_info_json, mask_dir):
        self.fish_scale = 1.0
        with open(config_json, 'r') as f:
            datas = json.load(f)["AVM"]
        camera_param = datas["cameraParam"]

        self.avm_size = (datas["2D"]["img_w"], datas["2D"]["img_h"])
        self.fev_size = (1280, 960)
        self.hardware = camera_param["hardware"]
        self.undistort_param = camera_param["undistort"]
        self.undistort_inv_param = camera_param["undis_params_inverse_0"]

        self.init_undistored_parameters()
        self.undist_map = self.init_undist_map()

        self.bev_fb_size = (616, 244)
        self.bev_lr_size = (880, 221)

        # get keypoints
        camera_keys = ["front", "back", "left", "right"]
        with open(camera_info_json, 'r') as f:
            camera_info = json.load(f)
        self.detected_pts = {}
        for key in camera_keys:
            self.detected_pts[key] = np.array(camera_info[key]["detected_points"], dtype = np.float32).reshape(-1, 2)

        # warp detected points to undist
        self.undist_pts = {}
        for key in camera_keys:
            pts = self.detected_pts[key]
            warp_pts = []
            for pt in pts:
                warp_pt = self.warpFisheye2Undist(pt[0], pt[1])
                warp_pts.extend(warp_pt)
            self.undist_pts[key] = np.array(warp_pts, dtype = np.float32).reshape(-1, 2)

        # get corners
        self.corner_pts = {}
        for key in camera_keys:
            self.corner_pts[key] = np.array(camera_info[key]["corner_points"], dtype = np.float32).reshape(-1, 2)

        # get bev homo
        self.homo_b2u = {}
        for key in camera_keys:
            self.homo_b2u[key], _ = cv2.findHomography(self.corner_pts[key].reshape(-1, 1, 2), self.undist_pts[key].reshape(-1, 1, 2), cv2.RANSAC, 5.0)

        # get bev map
        self.b2f_map = {}
        for key in camera_keys:
            if key in ["back", "front"]:
                bev_w = self.bev_fb_size[0]
                bev_h = self.bev_fb_size[1]
            else:
                bev_w = self.bev_lr_size[0]
                bev_h = self.bev_lr_size[1]
            self.b2f_map[key] = self.get_bev_map(bev_h, bev_w, self.homo_b2u[key])

        # get mask
        self.bev_mask = {}
        self.bev_mask["front"] = cv2.imread(os.path.join(mask_dir, "maskFront.jpg"), 0)
        self.bev_mask["back"] = cv2.imread(os.path.join(mask_dir, "maskBack.jpg"), 0)
        self.bev_mask["left"] = cv2.imread(os.path.join(mask_dir, "maskLeft.jpg"), 0)
        self.bev_mask["right"] = cv2.imread(os.path.join(mask_dir, "maskRight.jpg"), 0)
        for key in camera_keys:
            self.bev_mask[key] = self.bev_mask[key].astype(np.float32)[:, :, np.newaxis] / 255.

    def init_undistored_parameters(self):
        hardware = self.hardware
        fish = {"scale": self.fish_scale, "width": self.fev_size[0], "height": self.fev_size[1]}
        focal_len = hardware["focal_length"]
        self.dx = dx = hardware["dx"] / fish["scale"]
        self.dy = dy = hardware["dy"] / fish["scale"]
        self.fish_width = distort_width = int(fish["width"] * fish["scale"])
        self.fish_height = distort_height = int(fish["height"] * fish["scale"])
        undis_scale = self.undistort_param["undis_scale"]
        self.center_w = center_w = distort_width / 2
        self.center_h = center_h = distort_height / 2
        self.intrinsic = intrinsic = [
            [focal_len / dx, 0, center_w],
            [0, focal_len / dy, center_h],
            [0, 0, 1],
        ]
        self.intrinsic_undis = intrinsic_undis = [
            [focal_len / dx, 0, center_w * undis_scale],
            [focal_len / dy, 0, center_h * undis_scale],
            [0, 0, 1],
        ]
        self.undist_w = u_w = int(distort_width * undis_scale)
        self.undist_h = u_h = int(distort_height * undis_scale)
        self.d = math.sqrt(
            pow(intrinsic_undis[0][2] / intrinsic[0][0], 2)
            + pow(intrinsic_undis[1][2] / intrinsic[1][1], 2)
        )

    def get_grids(self, h, w, H = None):
        grids = np.meshgrid(np.arange(w), np.arange(h))
        grids = np.stack(grids, axis=2).astype(np.float32)
        if H is not None:
            # y = Hx
            grids = np.concatenate((grids, np.ones((h, w, 1))), axis = 2)
            grids = grids.reshape((-1, 3)).transpose(1, 0)
            grids = np.matmul(H, grids)
            grids = grids.transpose(1, 0).reshape((h, w, 3))
            grids /= grids[:, :, 2:]
        return grids[:, :, 0:2]

    def calc_angle_undistorted(self, r_undist):
        angle_undistorted = np.arctan(r_undist)
        angle_undistorted_p2 = angle_undistorted * angle_undistorted
        angle_undistorted_p3 = angle_undistorted_p2 * angle_undistorted
        angle_undistorted_p5 = angle_undistorted_p2 * angle_undistorted_p3
        angle_undistorted_p7 = angle_undistorted_p2 * angle_undistorted_p5
        angle_undistorted_p9 = angle_undistorted_p2 * angle_undistorted_p7
        r_distort = (
            angle_undistorted
            + self.undistort_param['Opencv_k0'] * angle_undistorted_p3
            + self.undistort_param['Opencv_k1'] * angle_undistorted_p5
            + self.undistort_param['Opencv_k2'] * angle_undistorted_p7
            + self.undistort_param['Opencv_k3'] * angle_undistorted_p9
        )
        scale = r_distort / (r_undist + 0.00001)

        return scale

    def init_undist_map(self, out_h = None, out_w = None, H = None):
        if out_h is None or out_w is None:
            out_h = self.undist_h
            out_w = self.undist_w
        undist_center = np.array([self.undist_w / 2, self.undist_h / 2]).reshape(
            1, 1, 2
        )
        dist_center = np.array([self.center_w, self.center_h]).reshape(1, 1, 2)
        f = np.array([self.intrinsic[0][0], self.intrinsic[1][1]]).reshape(1, 1, 2)

        grids = self.get_grids(out_h, out_w, H)
        grids = grids - undist_center
        grids_norm = grids / f
        r_undist = np.linalg.norm(grids_norm, axis=2)
        scale = self.calc_angle_undistorted(r_undist)
        scale = scale[..., np.newaxis]
        grids = grids * scale + dist_center
        return grids.astype(np.float32)

    def warpFisheye2Undist(self, x, y):
        y_ = (y - self.center_h) / self.intrinsic[1][1]
        x_ = (x - self.center_w) / self.intrinsic[0][0]
        r_distorted = math.sqrt(pow(x_, 2) + pow(y_, 2))

        r_distorted_p2 = r_distorted * r_distorted
        r_distorted_p3 = r_distorted_p2 * r_distorted
        r_distorted_p4 = r_distorted_p2 * r_distorted_p2
        r_distorted_p5 = r_distorted_p2 * r_distorted_p3
        angle_undistorted = (r_distorted 
                            + self.undistort_inv_param["k0"] * r_distorted_p2
                            + self.undistort_inv_param["k1"] * r_distorted_p3
                            + self.undistort_inv_param["k2"] * r_distorted_p4
                            + self.undistort_inv_param["k3"] * r_distorted_p5)
        # scale
        r_undistorted = np.tan(angle_undistorted)

        scale = r_undistorted / (r_distorted + 0.00001)  # scale = r_dis on the camera img plane divide r_undis on the normalized plane

        xx = (x - self.center_w) * self.fish_scale
        yy = (y - self.center_h) * self.fish_scale

        undis_center_h = self.undist_h / 2
        undis_center_w = self.undist_w / 2
        warp_xy = self.warpPointInverse(undis_center_h, undis_center_w, xx, yy, scale)

        return warp_xy

    def warpPointInverse(self, map_center_h, map_center_w, x_, y_, scale):
        warp_x = x_ * scale + map_center_w;
        warp_y = y_ * scale + map_center_h;
        return (warp_x, warp_y)

    def get_bev_map(self, bev_h, bev_w, homo_b2u, deta_H = None):
        H_b2u = deepcopy(homo_b2u)
        if deta_H is not None:
            H_b2u = np.matmul(H_b2u, deta_H)
        b2f_map = self.init_undist_map(bev_h, bev_w, H_b2u)
        return b2f_map

    def warp_image(self, src, dst2src_map):
        dst = cv2.remap(src, dst2src_map[:, :, 0], dst2src_map[:, :, 1], cv2.INTER_LINEAR)
        return dst

    def img_rot90(self, input, isclock = True):
        if isinstance(input, str):
            img = cv2.imread(input)
        else:
            img = input

        if isclock:
            img = img.transpose(1, 0, 2)[:, ::-1]
        else:
            img = img.transpose(1, 0, 2)[::-1]

        return np.ascontiguousarray(img)

    def merge2avm(self, inputs):
        for idx, input in enumerate(inputs):
            inputs[idx] = input.astype(np.float32)
        front, back, left, right = inputs

        front = front * self.bev_mask["front"]
        back = np.ascontiguousarray(back[::-1, ::-1]) * self.bev_mask["back"]
        left = self.img_rot90(left, isclock = False) * self.bev_mask["left"]
        right = self.img_rot90(right, isclock = True) * self.bev_mask["right"]
        
        avm_w, avm_h = self.avm_size
        out = np.zeros((avm_h, avm_w, front.shape[2]), dtype = np.float32)
        out[:front.shape[0]] += front
        out[avm_h - back.shape[0]:] += back
        out[:, :left.shape[1]] += left
        out[:, avm_w - right.shape[1]:] += right
        out = np.clip(out, 0, 255).astype(np.uint8)

        return out
    
    def save_pair_imgs(self, inputs, prefix):
        bevs = {}
        for idx, input in enumerate(inputs):
            bevs[idx] = input.astype(np.float32)
        bevs[1] = np.ascontiguousarray(bevs[1][::-1, ::-1])
        bevs[2] = self.img_rot90(bevs[2], isclock = False)
        bevs[3] = self.img_rot90(bevs[3], isclock = True)
        # ipdb.set_trace()
        # (616, 244)
        # (880, 221)
        fb_hw = [244, 616]
        lf_hw = [880, 221]
        sz = 192
        crop = {
            'f_l': bevs[0][52:, 29:221],
            'f_r': bevs[0][52:, 395:587],
            'b_l': bevs[1][:192, 29:221],
            'b_r': bevs[1][:192, 395:587],
            'l_f': bevs[2][52:244, 29:221],
            'l_b': bevs[2][636:828, 29:221],
            'r_f': bevs[3][52:244, :192],
            'r_b': bevs[3][636:828, :192]
        }
        for k, v in crop.items():
            svp = f'{prefix}_{k}.jpg'
            cv2.imwrite(svp, v)
        
        return crop

    def get_split_part(self, bev, key):
        avm_w, avm_h = self.avm_size
        out = np.zeros((avm_h, avm_w, bev.shape[2]), dtype = np.uint8)

        if key == "front":
            out[:bev.shape[0]] = bev
        elif key == "back":
            out[avm_h - bev.shape[0]:] = np.ascontiguousarray(bev[::-1, ::-1])
        elif key == "left":
            left = self.img_rot90(bev, isclock = False)
            out[:, :left.shape[1]] = left
        elif key == "right":
            right = self.img_rot90(bev, isclock = True)
            out[:, avm_w - right.shape[1]:] = right
        return out

def draw_pts(img, pt):
    cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    keys = ["front", "back", "left", "right"]
    warpper = Fev2AVM('b16param/config.json', 'b16param/camera_infor.json', 'b16param')

    outs = []
    for key in keys:
        img = cv2.imread('../test_datas/avm/{}.png'.format(key))
        dst = warpper.warp_image(img, warpper.undist_map)

        # draw points
        for pt in warpper.undist_pts[key]:
            draw_pts(dst, pt)
        cv2.imwrite(f'f2undist2{key}.jpg', dst)

        # get bev
        bev = warpper.warp_image(img, warpper.b2f_map[key])
        cv2.imwrite(f'bev_{key}.jpg', bev)

        outs.append(bev)

    avm_img = warpper.merge2avm(outs)
    cv2.imwrite('avm.jpg', avm_img)
