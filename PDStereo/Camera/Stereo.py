import cv2
import numpy as np
import PDStereo.Camera.Calibration as cb
import PDStereo.Camera.Utils as ut
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

class Stereo():
    def __init__(self, root_dir):
        self.ROOT_DIR = root_dir + "/"
        if not os.path.exists(self.ROOT_DIR):
            os.makedirs(self.ROOT_DIR)
        self.cam_points_left_path = self.ROOT_DIR + 'points_left.json'
        self.cam_points_right_path = self.ROOT_DIR + 'points_right.json'
        self.cam_params_left_path = self.ROOT_DIR + 'params_left.json'
        self.cam_params_right_path = self.ROOT_DIR + 'params_right.json'
        self.cam_params_stereo_path = self.ROOT_DIR + 'params_stereo.json'

        self.objPoints_left, self.imgPoints_left, self.imgPathes_left \
            = cb.loadPoints(self.cam_points_left_path)
        self.objPoints_right, self.imgPoints_right, self.imgPathes_right \
            = cb.loadPoints(self.cam_points_right_path)

        if self.objPoints_left == None \
            or self.imgPoints_left == None \
            or self.imgPathes_left == None:
            self.objPoints_left, self.imgPoints_left, self.imgPathes_left \
                = [], [], []
        if self.objPoints_right == None \
            or self.imgPoints_right == None \
            or self.imgPathes_right == None:
            self.objPoints_right, self.imgPoints_right, self.imgPathes_right \
                = [], [], []

        self.isStereoCalibrated = False

    def calibrateCamAndSave(self, leftFrame, rightFrame, corner_size, filename):
        assert type(corner_size) is tuple
        img_gray_left, objp_left, corner_left, ret_left \
            = cb.findChessboardCorners(leftFrame, corner_size)
        img_gray_right, objp_right, corner_right, ret_right \
            = cb.findChessboardCorners(rightFrame, corner_size)

        path = self.ROOT_DIR + 'images/'
        cal_left, cal_right = None, None
        if not os.path.exists(path):
            os.mkdir(path)
        if ret_left == True and ret_right == True:
            imgPath = path + 'left-{}.png'.format(filename)
            cv2.imwrite(imgPath, img_gray_left)
            self.imgPathes_left.append(imgPath)
            self.objPoints_left.append(objp_left.tolist())
            self.imgPoints_left.append(corner_left.tolist())
            cb.savePoints(self.objPoints_left, self.imgPoints_left, self.imgPathes_left, self.cam_points_left_path)
            cal_left = cv2.drawChessboardCorners(leftFrame, corner_size, corner_left, ret_left)

            imgPath = path + 'right-{}.png'.format(filename)
            cv2.imwrite(imgPath, img_gray_right)
            self.imgPathes_right.append(imgPath)
            self.objPoints_right.append(objp_right.tolist())
            self.imgPoints_right.append(corner_right.tolist())
            cb.savePoints(self.objPoints_right, self.imgPoints_right, self.imgPathes_right, self.cam_points_right_path)
            cal_right = cv2.drawChessboardCorners(rightFrame, corner_size, corner_right, ret_right)

        objps, imgps = np.array(self.objPoints_left, dtype=np.float32), np.array(self.imgPoints_left, dtype=np.float32)
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objps, imgps, leftFrame.shape[:2][::-1], None, None)
        objps, imgps = np.array(self.objPoints_right, dtype=np.float32), np.array(self.imgPoints_right, dtype=np.float32)
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objps, imgps, rightFrame.shape[:2][::-1], None, None)

        f = open(self.cam_params_left_path, 'w')
        f.write(json.dumps({
            'ret': ret_left,
            'camera_matrix': mtx_left.tolist(),
            'distortion_coefficients': dist_left.tolist(),
            'rotation_vectors': [i.tolist() for i in rvecs_left],
            'translation_vectors': [i.tolist() for i in tvecs_left]
        }))
        f.close()

        f = open(self.cam_params_right_path, 'w')
        f.write(json.dumps({
            'ret': ret_right,
            'camera_matrix': mtx_right.tolist(),
            'distortion_coefficients': dist_right.tolist(),
            'rotation_vectors': [i.tolist() for i in rvecs_right],
            'translation_vectors': [i.tolist() for i in tvecs_right]
        }))
        f.close()

        return cal_left, cal_right

    def stereoCalibrate(self, imgSize, rectifyScale = 0): # 0=full crop, 1=no crop
        if not (os.path.exists(self.cam_params_left_path)\
                and os.path.exists(self.cam_params_right_path)):
            return False

        f = open(self.cam_params_left_path, 'r')
        json_cam_param_left = json.loads(f.read())
        f.close()
        f = open(self.cam_params_right_path, 'r')
        json_cam_param_right = json.loads(f.read())
        f.close()
        camMatrix_left = np.asarray(json_cam_param_left['camera_matrix'])
        camMatrix_right = np.asarray(json_cam_param_right['camera_matrix'])
        distCoeffs_left = np.asarray(json_cam_param_left['distortion_coefficients'])
        distCoeffs_right = np.asarray(json_cam_param_right['distortion_coefficients'])
        rvecs_left = json_cam_param_left['rotation_vectors']
        rvecs_right = json_cam_param_right['rotation_vectors']
        tvecs_left = json_cam_param_left['translation_vectors']
        tvecs_right = json_cam_param_right['translation_vectors']

        ret, self.cam_mtx1, self.distCoeffs1, self.cam_mtx2, self.distCoeffs2, \
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            np.asarray(self.objPoints_left, np.float32),
            np.asarray(self.imgPoints_left, np.float32),
            np.asarray(self.imgPoints_right, np.float32),
            camMatrix_left, distCoeffs_left,
            camMatrix_right, distCoeffs_right,
            imgSize)
        
        self.R1, self.R2, self.P1, self.P2, self.Q, \
        self.validPixROI1, self.validPixROI2 = cv2.stereoRectify(
            self.cam_mtx1, self.distCoeffs1,
            self.cam_mtx2, self.distCoeffs2,
            imgSize,
            self.R,
            self.T,
            alpha=rectifyScale)

        self.isStereoCalibrated = True
        return True

    def stereoMatching(self, leftFrame, rightFrame, filename='untitled', savePointClound=False, rectifyScale=0):
        imgSize = leftFrame.shape[:2][::-1]
        if not self.isStereoCalibrated:
            if not self.stereoCalibrate(imgSize, rectifyScale):
                return None

        leftMaps = cv2.initUndistortRectifyMap(
            self.cam_mtx1,
            self.distCoeffs1,
            self.R1,
            self.P1,
            imgSize,
            cv2.CV_16SC2)
        rightMaps = cv2.initUndistortRectifyMap(
            self.cam_mtx2,
            self.distCoeffs2,
            self.R2,
            self.P2,
            imgSize,
            cv2.CV_16SC2)

        left_img_remap = cv2.remap(leftFrame, leftMaps[0], leftMaps[1], cv2.INTER_LANCZOS4)
        right_img_remap = cv2.remap(rightFrame, rightMaps[0], rightMaps[1], cv2.INTER_LANCZOS4)

        left_img_remap_blur = cv2.medianBlur(left_img_remap, 5)
        right_img_remap_blur = cv2.medianBlur(right_img_remap, 5)
        left_img_remap_gray = cv2.cvtColor(left_img_remap_blur, cv2.COLOR_BGR2GRAY)
        right_img_remap_gray = cv2.cvtColor(right_img_remap_blur, cv2.COLOR_BGR2GRAY)
        
        window_size = 11
        lmbda = 20000
        sigma = 1.0
        num_disp = 96
        #disparity, visibleDisparity = ut.get_disparity_map(
        disparity = ut.get_disparity_map(
            left_img_remap_gray, right_img_remap_gray,
            window_size=window_size,
            lmbda=lmbda,
            sigma=sigma,
            num_disp=num_disp)
        #min = disparity.min()
        #max = disparity.max()
        #disparity = np.uint8(255 * (disparity - min) / (max - min))
        #disparity
        disparity = cv2.medianBlur(disparity, 5)
        disparity_ROI = cv2.getValidDisparityROI(self.validPixROI1, self.validPixROI2, disparity.min(),
            num_disp, window_size)
        
        x1, y1, x2, y2 = disparity_ROI
        img_left_ROI = left_img_remap[y1:y1+y2, x1:x1+x2]
        img_disparity_ROI = disparity[y1:y1+y2, x1:x1+x2]

        newImgSize = np.asarray(img_disparity_ROI.shape) - 30
        newImgSize = (
            img_disparity_ROI.shape[0] * 0.5 - newImgSize[0] / 2,
            img_disparity_ROI.shape[1] * 0.5 - newImgSize[1] / 2,
            newImgSize[0] + img_disparity_ROI.shape[0] * 0.5 - newImgSize[0] / 2,
            newImgSize[1] + img_disparity_ROI.shape[1] * 0.5 - newImgSize[1] / 2
        )
        x1, y1, x2, y2 = [int(i) for i in np.round(newImgSize)]
        img_left_ROI = img_left_ROI[y1:y1+y2, x1:x1+x2]
        img_disparity_ROI = img_disparity_ROI[y1:y1+y2, x1:x1+x2]

        if savePointClound:
            ut.savePointClound(filename, left_img_remap, disparity, Q, imgSize)

        edge_rgb = self.create_edge_image(cv2.cvtColor(img_left_ROI, cv2.COLOR_BGR2GRAY))
        edge_depth = self.create_edge_image(img_disparity_ROI)
        weighted_edge = self.create_weighted_image(edge_rgb, edge_depth)
        weighted_depth = self.create_weighted_image(img_disparity_ROI, weighted_edge)

        #rgbd = np.dstack((img_left_ROI, img_disparity_ROI))
        rgbd = np.dstack((img_left_ROI, weighted_depth))

        return {
            "original": {
                "left_remap": left_img_remap,
                "right_remap": right_img_remap,
                "disparity": disparity,
            },
            "result": {
                "image": img_left_ROI,
                "disparity": img_disparity_ROI,
                'weighted_depth': weighted_depth,
                "rgbd": rgbd
            }
        }

    def create_edge_image(self, image, ksize=3):
        edgeImage_X = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=ksize)
        edgeImage_Y = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=ksize)
        edgeImage = cv2.addWeighted(edgeImage_X, 1, edgeImage_Y, 1, 0)
        return edgeImage
    
    def create_weighted_image(self, img1, img2):
        return cv2.addWeighted(img1, 1, img2, 1, 0)