# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './QtApp\ui/QtMain.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cam_left = QtWidgets.QGraphicsView(self.centralwidget)
        self.cam_left.setGeometry(QtCore.QRect(0, 0, 320, 240))
        self.cam_left.setObjectName("cam_left")
        self.cam_right = QtWidgets.QGraphicsView(self.centralwidget)
        self.cam_right.setGeometry(QtCore.QRect(320, 0, 320, 240))
        self.cam_right.setObjectName("cam_right")
        self.spinBox_cam_left = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_cam_left.setGeometry(QtCore.QRect(280, 240, 42, 22))
        self.spinBox_cam_left.setObjectName("spinBox_cam_left")
        self.spinBox_cam_right = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_cam_right.setGeometry(QtCore.QRect(600, 240, 42, 22))
        self.spinBox_cam_right.setObjectName("spinBox_cam_right")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFiles = QtWidgets.QMenu(self.menubar)
        self.menuFiles.setObjectName("menuFiles")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFiles.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RGB-D Detector"))
        self.menuFiles.setTitle(_translate("MainWindow", "Files"))

