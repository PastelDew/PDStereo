import cv2

def getAvailableCameraCount():
    idx = 0
    while True:
        cam = cv2.VideoCapture(idx)
        if not cam.isOpened():
            break
        idx = idx + 1
    return idx

