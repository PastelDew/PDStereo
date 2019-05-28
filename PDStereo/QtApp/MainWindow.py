import cv2
from datetime import datetime
import os
import numpy as np

from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from PDStereo.QtApp.QtUI.QtMain import Ui_MainWindow
from PDStereo.Camera import CameraInfo
from PDStereo.Camera.Stereo import Stereo
from PDStereo.QtApp.Detector import Detector
from PDStereo.InjeAI import InjeAI

import qimage2ndarray

class MainWindow(QMainWindow):
    def initialize(self, mainForm: Ui_MainWindow):
        self.mainForm = mainForm
        
        #Connect events
        mainForm.actionExit.triggered.connect(self.close)

        mainForm.spinBox_cam_left.valueChanged.connect(self.event_spinBox_cam_left_changed)
        mainForm.spinBox_cam_right.valueChanged.connect(self.event_spinBox_cam_right_changed)

        mainForm.btn_callibration.clicked.connect(self.event_callibration_clicked)
        mainForm.btn_stereoCallibration.clicked.connect(self.event_stereoCallibration_clicked)
        mainForm.btn_camcnt_refresh.clicked.connect(self.event_btn_camcnt_refresh)
        mainForm.btn_detect_rgb.clicked.connect(self.event_btn_detect_rgb_clicked)
        mainForm.btn_detect_rgbd.clicked.connect(self.event_btn_detect_rgbd_clicked)

        mainForm.spinBox_corner_X.valueChanged.connect(self.event_spinBox_chessboard_size_changed)
        mainForm.spinBox_corner_Y.valueChanged.connect(self.event_spinBox_chessboard_size_changed)

        #Init Scroll View
        mainForm.scrollArea = QtWidgets.QScrollArea(mainForm.groupBox_Colors)
        itemRect = QRect(
            10, 20,
            mainForm.groupBox_Colors.width() - 20,
            mainForm.groupBox_Colors.height() - 30
        )
        mainForm.scrollArea.setGeometry(itemRect)
        mainForm.scrollArea.setWidgetResizable(True)
        mainForm.scrollArea.setObjectName("scrollArea")
        mainForm.gridLayoutWidget = QtWidgets.QWidget()
        mainForm.gridLayoutWidget.setGeometry(itemRect)
        mainForm.gridLayoutWidget.setObjectName("gridLayoutWidget")
        mainForm.gridLayout_Colors = QtWidgets.QGridLayout(mainForm.gridLayoutWidget)
        mainForm.gridLayout_Colors.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        mainForm.gridLayout_Colors.setContentsMargins(0, 0, 0, 0)
        mainForm.gridLayout_Colors.setVerticalSpacing(0)
        mainForm.gridLayout_Colors.setObjectName("gridLayout_Colors")
        mainForm.scrollArea.setWidget(mainForm.gridLayoutWidget)
        mainForm.gridLayout_Colors.setColumnStretch(0, 2)
        mainForm.gridLayout_Colors.setColumnStretch(2, 3)

        #Init Corner Size
        self.event_spinBox_chessboard_size_changed()

        #Init Cam Count
        self.event_btn_camcnt_refresh()

        self.left_cam = None
        self.right_cam = None
        self.leftFrame = None
        self.rightFrame = None

        self.fileDialog = QtWidgets.QFileDialog()
        self.fileDialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.fileDialog.setNameFilter("Weights File (*.h5)")
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        self.isDisparityProcessing = False
        self.isCallibrating = False
        self.isStereoCallibrating = False
        self.isRemapping = False
        self.imgVisibleCnt = 0
        self.detectMode = 0
        self.remapResult = None
        self.weightPath_RGB = None
        self.weightPath_RGBD = None

        self.detector_rgb = Detector(InjeAI.load_class("DL/datasets/objects/rgb/"))
        self.detector_rgbd = Detector(InjeAI.load_class("DL/datasets/objects/rgbd/"))
        self.detector = None

        self.stereo = Stereo("DL/StereoFiles")
        self.savedImageDir = "DL/SavedImages"
        

        #Threads
        self.thread_cam = Worker(target=self.thread_cam_run)
        self.thread_cam.start()

        self.thread_mapping = Worker(target=self.thread_mapping_run)
        self.thread_mapping.start()

        self.thread_detecting = Worker(target=self.thread_detecting_run)
        self.thread_detecting.start()
    
    def closeEvent(self, e):
        self.thread_cam.isWorking = False
        self.thread_mapping.isWorking = False
        if self.left_cam != None:
            self.left_cam.release()
        if self.right_cam != None:
            self.right_cam.release()
        cv2.destroyAllWindows()

    def event_btn_camcnt_refresh(self):
        self.available_cam_count = CameraInfo.getAvailableCameraCount()
        self.mainForm.label_available_cam_count.setText(str(self.available_cam_count))
        self.mainForm.spinBox_cam_left.setMaximum(self.available_cam_count)
        self.mainForm.spinBox_cam_right.setMaximum(self.available_cam_count)

    def event_btn_detect_rgb_clicked(self):
        if self.detectMode == 1:
            return
        if self.weightPath_RGB is None:
            if not self.fileDialog.exec_():
                return
            self.weightPath_RGB = self.fileDialog.selectedFiles()[0]
            self.detector_rgb.load_weights(self.weightPath_RGB, channel=3)

        self.detectMode = 0
        prevDetector = None
        if self.detector:
            prevDetector = self.detector
            while self.detector.isDetecting:
                QThread.sleep(1)
            prevDetector.isDetecting = True

        self.detector = self.detector_rgb
        labels = self.detector.class_names[1:]
        colors = self.detector.colors
        self.drawColorMapToGridLayout(labels, colors)
        
        self.mainForm.label_detecting.setText(os.path.basename(self.weightPath_RGB))
        if prevDetector:
            prevDetector.isDetecting = False
        self.detectMode = 1

    def event_btn_detect_rgbd_clicked(self):
        if self.detectMode == 2:
            return
        if self.weightPath_RGBD is None:
            if not self.fileDialog.exec_():
                return
            self.weightPath_RGBD = self.fileDialog.selectedFiles()[0]
            self.detector_rgbd.load_weights(self.weightPath_RGBD, channel=4)
        self.detectMode = 0
        prevDetector = None
        if self.detector:
            prevDetector = self.detector
            while self.detector.isDetecting:
                QThread.sleep(1)
            prevDetector.isDetecting = True
        
        self.detector = self.detector_rgbd
        labels = self.detector.class_names[1:]
        colors = self.detector.colors
        self.drawColorMapToGridLayout(labels, colors)

        self.mainForm.label_detecting.setText(os.path.basename(self.weightPath_RGBD))
        if prevDetector:
            prevDetector.isDetecting = False
        self.detectMode = 2

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

    def drawColorMapToGridLayout(self, labels, colors):
        for i in reversed(range(self.mainForm.gridLayout_Colors.count())):
            item = self.mainForm.gridLayout_Colors.itemAt(i).widget()
            self.mainForm.gridLayout_Colors.removeWidget(item)
            item.deleteLater()
        
        for i in range(len(labels)):
            label = labels[i]
            color = colors[label]
            color_for_cv = tuple([int(255 * c) for c in color[::-1]])
            blank_image = np.ones((8, 8, 3), np.uint8)
            blank_image = blank_image * color_for_cv
            blank_image = qimage2ndarray.array2qimage(blank_image)
            pixmap = QtGui.QPixmap(blank_image)

            label_color = QtWidgets.QLabel()
            label_color.setPixmap(pixmap)
            label_text = QtWidgets.QLabel()
            label_text.setText(label)

            self.mainForm.gridLayout_Colors.addWidget(label_color,i,0, Qt.AlignCenter)
            self.mainForm.gridLayout_Colors.addWidget(label_text,i,2)

    def generateFileName(self):
        dt = datetime.now()
        self.filename = dt.strftime('%y%m%d-%H%M%S') + str(dt.microsecond)[:3]
        return self.filename

    def retrieveFrames(self):
        if self.leftFrame is None or self.rightFrame is None:
            return None, None
        self.isDisparityProcessing = True
        leftFrame = self.leftFrame
        rightFrame = self.rightFrame
        self.leftFrame = None
        self.rightFrame = None
        self.isDisparityProcessing = False
        return leftFrame, rightFrame

    def thread_cam_run(self):
        leftFrame, rightFrame = None, None
        if self.left_cam != None and self.left_cam.grab():
            _, leftFrame = self.left_cam.retrieve()
            if leftFrame is not None and self.imgVisibleCnt == 0 \
                and not self.mainForm.checkBox_show_remap.isChecked():
                pixmap = QtGui.QPixmap(
                    self.convert_image_to_QImage(
                        self.resize_image(leftFrame, 320, 240)))
                self.mainForm.cam_left.setPixmap(pixmap)
                self.mainForm.cam_left.update()

        if self.right_cam != None and self.right_cam.grab():
            _, rightFrame = self.right_cam.retrieve()
            if rightFrame is not None and self.imgVisibleCnt == 0 \
                and not self.mainForm.checkBox_show_remap.isChecked():
                pixmap = QtGui.QPixmap(
                    self.convert_image_to_QImage(
                        self.resize_image(rightFrame, 320, 240)))
                self.mainForm.cam_right.setPixmap(pixmap)
                self.mainForm.cam_right.update()

        if self.imgVisibleCnt > 0:
            self.imgVisibleCnt = self.imgVisibleCnt - 1

        if leftFrame is None or rightFrame is None:
            return
        
        while self.isDisparityProcessing:
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
            rectifyScale = 0
            if self.mainForm.checkBox_show_remap.isChecked():
                rectifyScale = 1
            self.stereo.stereoCallibrate(leftFrame.shape[:2][::-1], rectifyScale=rectifyScale)
            self.isStereoCallibrating = False
        
        result = self.stereo.stereoMatching(leftFrame, rightFrame)
        if result is None:
            return
        
        if self.mainForm.checkBox_show_remap.isChecked():
            orig = result['original']
            pixmap = QtGui.QPixmap(
                self.convert_image_to_QImage(
                    self.resize_image(orig["left_remap"], 320, 240)))
            self.mainForm.cam_left.setPixmap(pixmap)
            self.mainForm.cam_left.update()

            pixmap = QtGui.QPixmap(
                self.convert_image_to_QImage(
                    self.resize_image(orig["right_remap"], 320, 240)))
            self.mainForm.cam_right.setPixmap(pixmap)
            self.mainForm.cam_right.update()

        result = result['result']
        pixmap = QtGui.QPixmap(
            self.convert_image_to_QImage(
                self.resize_image(result["disparity"], 320, 240)))
        self.mainForm.cam_disparity.setPixmap(pixmap)
        self.mainForm.cam_disparity.update()

        if self.mainForm.checkBox_saveImages.isChecked():
            if not os.path.exists(self.savedImageDir):
                os.mkdir(self.savedImageDir)
            filename = self.generateFileName()
            cv2.imwrite('{}/color-{}.png'.format(self.savedImageDir, filename), result['image'])
            cv2.imwrite('{}/depth-{}.png'.format(self.savedImageDir, filename), result['disparity'])
            cv2.imwrite('{}/merged-{}.png'.format(self.savedImageDir, filename), result['rgbd'])

        while self.isRemapping:
            QThread.sleep(1)
        
        self.isRemapping = True
        self.remapResult = result
        self.isRemapping = False

    def thread_detecting_run(self):
        if self.detectMode == 0:
            return

        while self.isRemapping:
            QThread.sleep(1)

        if self.remapResult is None:
            return

        self.isRemapping = True
        result = self.remapResult
        self.isRemapping = False

        if self.detectMode == 1:
            img = result['image']
        elif self.detectMode == 2:
            img = result['rgbd']
        else:
            return
            
        while self.detector.isDetecting:
            QThread.sleep(100)
        if not (self.detectMode == 1 and self.detector is self.detector_rgb) \
            and not (self.detectMode == 2 and self.detector is self.detector_rgbd):
            return
            
        r = self.detector.detect([img])[0]
        if r['rois'].shape[0] == r['masks'].shape[-1] == r['class_ids'].shape[0] > 0:
            img = self.detector.get_instances_image(img[...,:3], r['rois'],
                r['masks'], r['class_ids'],
                r['scores'])
            if self.mainForm.checkBox_save_detected_image.isChecked():
                filename = self.generateFileName()
                path = "DL/Detected/"
                if self.detectMode == 1:
                    path += "rgb/"
                else:
                    path += "rgbd/"
                if not (os.path.exists(path)):
                    os.makedirs(path)
                cv2.imwrite(path + filename + ".png", img)
        pixmap = QtGui.QPixmap(
            self.convert_image_to_QImage(
                self.resize_image(img, 320, 240)))
        self.mainForm.cam_detect.setPixmap(pixmap)
        self.mainForm.cam_detect.update()
        return


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