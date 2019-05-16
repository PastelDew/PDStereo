import cv2
from datetime import datetime

from QtApp.QtUI.QtMain import Ui_MainWindow
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from PDStereo.Camera import CameraInfo
from PDStereo.Camera.Stereo import Stereo

import qimage2ndarray

class MainWindow(QMainWindow):
    def initialize(self, mainForm: Ui_MainWindow):
        self.mainForm = mainForm

        #Connect events
        mainForm.spinBox_cam_left.valueChanged.connect(self.event_spinBox_cam_left_changed)
        mainForm.spinBox_cam_right.valueChanged.connect(self.event_spinBox_cam_right_changed)
        mainForm.btn_callibration.clicked.connect(self.event_callibration_clicked)
        mainForm.btn_stereoCallibration.clicked.connect(self.event_stereoCallibration_clicked)
        mainForm.spinBox_corner_X.valueChanged.connect(self.event_spinBox_chessboard_size_changed)
        mainForm.spinBox_corner_Y.valueChanged.connect(self.event_spinBox_chessboard_size_changed)

        #Init Corner Size
        self.event_spinBox_chessboard_size_changed()

        self.available_cam_count = CameraInfo.getAvailableCameraCount()
        mainForm.label_available_cam_count.setText(str(self.available_cam_count))
        mainForm.spinBox_cam_left.setMaximum(self.available_cam_count)
        mainForm.spinBox_cam_right.setMaximum(self.available_cam_count)

        self.left_cam = None
        self.right_cam = None
        self.isProcessing = False
        self.isCallibrating = False
        self.isStereoCallibrating = False
        self.imgVisibleCnt = 0
        self.leftFrame = None
        self.rightFrame = None

        self.stereo = Stereo("temp/StereoFiles")

        #Threads
        self.thread_cam = Worker(target=self.thread_cam_run)
        self.thread_cam.start()

        self.thread_mapping = Worker(target=self.thread_mapping_run)
        self.thread_mapping.start()
    
    def closeEvent(self, e):
        self.thread_cam.isWorking = False
        self.thread_mapping.isWorking = False
        if self.left_cam != None:
            self.left_cam.release()
        if self.right_cam != None:
            self.right_cam.release()
        cv2.destroyAllWindows()

    def event_spinBox_cam_left_changed(self):
        if self.left_cam != None:
            self.left_cam.release()
        idx = self.mainForm.spinBox_cam_left.value()
        right_idx = self.mainForm.spinBox_cam_right.value()
        
        if idx == 0 or idx == right_idx:
            self.left_cam = None
            return

        self.left_cam = cv2.VideoCapture(idx - 1)

    def event_spinBox_cam_right_changed(self):
        if self.right_cam != None:
            self.right_cam.release()
        idx = self.mainForm.spinBox_cam_right.value()
        left_idx = self.mainForm.spinBox_cam_left.value()
        if idx == 0 or idx == left_idx:
            self.right_cam = None
            return

        self.right_cam = cv2.VideoCapture(idx - 1)

    def event_spinBox_chessboard_size_changed(self):
        self.corner_size = (self.mainForm.spinBox_corner_X.value(),
                            self.mainForm.spinBox_corner_Y.value())

    def event_callibration_clicked(self):
        self.isCallibrating = True

    def event_stereoCallibration_clicked(self):
        self.isStereoCallibrating = True

    def generateFileName(self):
        dt = datetime.now()
        self.filename = dt.strftime('%y%m%d-%H%M%S') + str(dt.microsecond)[:3]
        return self.filename

    def retrieveFrames(self):
        if self.leftFrame is None or self.rightFrame is None:
            return None, None
        self.isProcessing = True
        leftFrame = self.leftFrame
        rightFrame = self.rightFrame
        self.leftFrame = None
        self.rightFrame = None
        self.isProcessing = False
        return leftFrame, rightFrame

    def thread_cam_run(self):
        leftFrame, rightFrame = None, None
        if self.left_cam != None and self.left_cam.grab():
            _, leftFrame = self.left_cam.retrieve()
            if leftFrame is not None:
                pixmap = QtGui.QPixmap(
                    self.convert_image_to_QImage(
                        self.resize_image(leftFrame, 320, 240)))
                if self.imgVisibleCnt == 0:
                    self.mainForm.cam_left.setPixmap(pixmap)
                    self.mainForm.cam_left.update()

        if self.right_cam != None and self.right_cam.grab():
            _, rightFrame = self.right_cam.retrieve()
            if rightFrame is not None:
                pixmap = QtGui.QPixmap(
                    self.convert_image_to_QImage(
                        self.resize_image(rightFrame, 320, 240)))
                if self.imgVisibleCnt == 0:
                    self.mainForm.cam_right.setPixmap(pixmap)
                    self.mainForm.cam_right.update()

        if self.imgVisibleCnt > 0:
            self.imgVisibleCnt = self.imgVisibleCnt - 1

        if leftFrame is None or rightFrame is None:
            return
        
        while self.isProcessing:
            QThread.sleep(1)
        self.leftFrame = leftFrame
        self.rightFrame = rightFrame

    def thread_mapping_run(self):
        leftFrame, rightFrame = self.retrieveFrames()
        if leftFrame is None or rightFrame is None:
            return

        filename = self.generateFileName()

        if self.isCallibrating:
            cal_left, cal_right = self.stereo.callibrateCamAndSave(leftFrame, rightFrame, self.corner_size, filename)
            if cal_left is not None and cal_right is not None:
                cal_left = self.convert_image_to_QImage(
                    self.resize_image(cal_left, 320, 240))
                cal_right = self.convert_image_to_QImage(
                    self.resize_image(cal_right, 320, 240))
                pxm_left = QtGui.QPixmap(cal_left)
                pxm_right = QtGui.QPixmap(cal_right)
                self.mainForm.cam_left.setPixmap(pxm_left)
                self.mainForm.cam_left.update()
                self.mainForm.cam_right.setPixmap(pxm_right)
                self.mainForm.cam_right.update()
                self.imgVisibleCnt = 120
            self.isCallibrating = False

        if self.isStereoCallibrating:
            self.stereo.stereoCallibrate(leftFrame.shape[:2][::-1])
            self.isStereoCallibrating = False

        result = self.stereo.stereoMatching(leftFrame, rightFrame)
        if result is None:
            return
        result = result['result']
        pixmap = QtGui.QPixmap(
            self.convert_image_to_QImage(
                self.resize_image(result["disparity"], 320, 240)))
        self.mainForm.cam_disparity.setPixmap(pixmap)
        self.mainForm.cam_disparity.update()

    def crop_image(self, img, width, height):
        size = img.shape[:2][::-1]
        x = int(size[0] / 2 - width / 2)
        y = int(size[1] / 2 - height / 2)
        return img[x:x+width,y:y+height]

    def resize_image(self, img, width, height):
        return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

    def convert_image_to_QImage(self, img):
        channel = 0
        if img.ndim < 3:
            channel = 1
            height, width = img.shape
        else:
            height, width, channel = img.shape

        bytesPerLine = width * channel
        format = QtGui.QImage.Format_RGB888
        if channel == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            format = QtGui.QImage.Format_ARGB32
        elif channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif channel == 1:
            format = QtGui.QImage.Format_Grayscale8
            #Bug exists in Qt_
            #return QtGui.QImage(img.data, width, height, format=format)
            #So, use this line instead.
            return qimage2ndarray.array2qimage(img)

        return QtGui.QImage(img.data, width, height, bytesPerLine, format)

class Worker(QThread):
    def __init__(self, target):
        assert callable(target)
        super().__init__()
        self.func = target
        self.isWorking = True

    def run(self):
        while self.isWorking:
            self.func()
            self.sleep(0)