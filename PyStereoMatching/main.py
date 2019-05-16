import cv2
import numpy as np
import calibration as cb
import utils as ut
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

left = cv2.VideoCapture(0)
right = cv2.VideoCapture(1)

isFlipped = True
isCallibrating = False

ROOT_DIR = './'
IMAGE_DATA_DIR = ROOT_DIR + 'ImageData/'

corner_size = (9, 6)
cam_points_left_path = ROOT_DIR + 'points_left.json'
cam_points_right_path = ROOT_DIR + 'points_right.json'
cam_params_left_path = ROOT_DIR + 'params_left.json'
cam_params_right_path = ROOT_DIR + 'params_right.json'
cam_params_stereo_path = ROOT_DIR + 'params_stereo.json'

objPoints_left, imgPoints_left, imgPathes_left = cb.loadPoints(cam_points_left_path)
objPoints_right, imgPoints_right, imgPathes_right = cb.loadPoints(cam_points_right_path)

if objPoints_left == None or imgPoints_left == None or imgPathes_left == None:
    objPoints_left, imgPoints_left, imgPathes_left = [], [], []
if objPoints_right == None or imgPoints_right == None or imgPathes_right == None:
    objPoints_right, imgPoints_right, imgPathes_right = [], [], []

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('f'):
        isFlipped = not isFlipped
    elif key == ord('c'):
        isCallibrating = True

    if not (left.grab() and right.grab()):
        print("No grabbed frames")
        continue
    
    _, leftFrame = left.retrieve()
    _, rightFrame = right.retrieve()

    if(isFlipped):
        tempFrame = leftFrame
        leftFrame = rightFrame
        rightFrame = tempFrame

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)

    dt = datetime.now()
    filename = dt.strftime('%y%m%d-%H%M%S') + str(dt.microsecond)[:3]

    if isCallibrating:
        img_gray_left, objp_left, corner_left, ret_left = cb.findChessboardCorners(leftFrame, corner_size)
        img_gray_right, objp_right, corner_right, ret_right = cb.findChessboardCorners(rightFrame, corner_size)

        path = ROOT_DIR + 'images/'
        if not os.path.exists(path):
            os.mkdir(path)
        if ret_left == True and ret_right == True:
            imgPath = path + 'left-{}.png'.format(filename)
            cv2.imwrite(imgPath, img_gray_left)
            imgPathes_left.append(imgPath)
            objPoints_left.append(objp_left.tolist())
            imgPoints_left.append(corner_left.tolist())
            cb.savePoints(objPoints_left, imgPoints_left, imgPathes_left, cam_points_left_path)
            cal_left = cv2.drawChessboardCorners(leftFrame, corner_size, corner_left, ret_left)
            cv2.imshow('cal_left', cal_left)

            imgPath = path + 'right-{}.png'.format(filename)
            cv2.imwrite(imgPath, img_gray_right)
            imgPathes_right.append(imgPath)
            objPoints_right.append(objp_right.tolist())
            imgPoints_right.append(corner_right.tolist())
            cb.savePoints(objPoints_right, imgPoints_right, imgPathes_right, cam_points_right_path)
            cal_right = cv2.drawChessboardCorners(rightFrame, corner_size, corner_right, ret_right)
            cv2.imshow('cal_right', cal_right)

        objps, imgps = np.array(objPoints_left, dtype=np.float32), np.array(imgPoints_left, dtype=np.float32)
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objps, imgps, leftFrame.shape[:2][::-1], None, None)
        objps, imgps = np.array(objPoints_right, dtype=np.float32), np.array(imgPoints_right, dtype=np.float32)
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objps, imgps, rightFrame.shape[:2][::-1], None, None)

        f = open(cam_params_left_path, 'w')
        f.write(json.dumps({
            'ret': ret_left,
            'camera_matrix': mtx_left.tolist(),
            'distortion_coefficients': dist_left.tolist(),
            'rotation_vectors': [i.tolist() for i in rvecs_left],
            'translation_vectors': [i.tolist() for i in tvecs_left]
        }))
        f.close()

        f = open(cam_params_right_path, 'w')
        f.write(json.dumps({
            'ret': ret_right,
            'camera_matrix': mtx_right.tolist(),
            'distortion_coefficients': dist_right.tolist(),
            'rotation_vectors': [i.tolist() for i in rvecs_right],
            'translation_vectors': [i.tolist() for i in tvecs_right]
        }))
        f.close()
        
        isCallibrating = False


    #if key == ord('r')\
    #    and os.path.exists(cam_params_left_path)\
    #    and os.path.exists(cam_params_right_path):
    if os.path.exists(cam_params_left_path)\
        and os.path.exists(cam_params_right_path):

        f = open(cam_params_left_path, 'r')
        json_cam_param_left = json.loads(f.read())
        f.close()
        f = open(cam_params_right_path, 'r')
        json_cam_param_right = json.loads(f.read())
        f.close()
        camMatrix_left = np.asarray(json_cam_param_left['camera_matrix'])
        camMatrix_right = np.asarray(json_cam_param_right['camera_matrix'])
        distCoeffs_left = np.asarray(json_cam_param_left['distortion_coefficients'])
        distCoeffs_right = np.asarray(json_cam_param_right['distortion_coefficients'])
        imgSize = leftFrame.shape[:2][::-1]
        rvecs_left = json_cam_param_left['rotation_vectors']
        rvecs_right = json_cam_param_right['rotation_vectors']
        tvecs_left = json_cam_param_left['translation_vectors']
        tvecs_right = json_cam_param_right['translation_vectors']

        ret, cam_mtx1, distCoeffs1, cam_mtx2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
            np.asarray(objPoints_left, np.float32),
            np.asarray(imgPoints_left, np.float32),
            np.asarray(imgPoints_right, np.float32),
            camMatrix_left, distCoeffs_left,
            camMatrix_right, distCoeffs_right,
            imgSize)
        
        rectifyScale = 0 # 0=full crop, 1=no crop
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            cam_mtx1, distCoeffs1,
            cam_mtx2, distCoeffs2,
            imgSize,
            R,
            T,
            alpha=rectifyScale)

        leftMaps = cv2.initUndistortRectifyMap(cam_mtx1, distCoeffs1, R1, P1, imgSize, cv2.CV_16SC2)
        rightMaps = cv2.initUndistortRectifyMap(cam_mtx2, distCoeffs2, R2, P2, imgSize, cv2.CV_16SC2)

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
        disparity_ROI = cv2.getValidDisparityROI(validPixROI1, validPixROI2, disparity.min(),
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

        #cv2.imshow("Left - Remap", left_img_remap)
        #cv2.imshow("Right - Remap", right_img_remap)
        #cv2.imshow("disparity", disparity)
        cv2.imshow('ROI - Left image', img_left_ROI)
        cv2.imshow('ROI - disparity', img_disparity_ROI)
        
        if key == ord('p'):
            ut.savePointClound(filename, left_img_remap, disparity, Q, imgSize)
        
        if key == ord('s'):
            rgbd = ut.saveRGBD(IMAGE_DATA_DIR, filename, img_left_ROI, img_disparity_ROI)
            #plt.imshow(rgbd)
            #plt.show()



left.release()
right.release()
cv2.destroyAllWindows()
