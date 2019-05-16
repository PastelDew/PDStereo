import numpy as np
import cv2
import glob
import json
import os

def findChessboardCorners(img, corner_size):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((corner_size[0]*corner_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_size[0],0:corner_size[1]].T.reshape(-1,2)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (corner_size[0],corner_size[1]),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        return gray, objp, corners2, ret
    else:
        return None, [], [], ret

def savePoints(object_points, image_points, images_path, path):
    jstr = json.dumps({"objp": object_points, "imgp": image_points, "path": images_path})
    f = open(path, 'w')
    f.write(jstr)
    f.close()

def loadPoints(path):
    if not os.path.exists(path):
        return None, None, None
    f = open(path, 'r')
    jstr = f.read()
    f.close()
    jstr = json.loads(jstr)
    return jstr['objp'], jstr['imgp'], jstr['path']