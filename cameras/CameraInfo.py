import cv2

def getAvailableCameraList():
    idx = 0
    while True:
        cam = cv2.VideoCapture(idx)
        if not cam.isOpened():
            break
    return idx

