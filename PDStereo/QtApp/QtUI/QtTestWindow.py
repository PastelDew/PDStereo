# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './PDStereo\QtApp\ui/QtTestWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TestWindow(object):
    def setupUi(self, TestWindow):
        TestWindow.setObjectName("TestWindow")
        TestWindow.resize(640, 480)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(TestWindow.sizePolicy().hasHeightForWidth())
        TestWindow.setSizePolicy(sizePolicy)
        TestWindow.setMinimumSize(QtCore.QSize(640, 480))
        TestWindow.setMaximumSize(QtCore.QSize(640, 480))
        self.tabWidget = QtWidgets.QTabWidget(TestWindow)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 511, 481))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.frame = QtWidgets.QFrame(TestWindow)
        self.frame.setGeometry(QtCore.QRect(510, 0, 131, 481))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.btn_openValidationSet = QtWidgets.QPushButton(self.frame)
        self.btn_openValidationSet.setGeometry(QtCore.QRect(0, 0, 131, 31))
        self.btn_openValidationSet.setObjectName("btn_openValidationSet")

        self.retranslateUi(TestWindow)
        QtCore.QMetaObject.connectSlotsByName(TestWindow)

    def retranslateUi(self, TestWindow):
        _translate = QtCore.QCoreApplication.translate
        TestWindow.setWindowTitle(_translate("TestWindow", "Test Window"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("TestWindow", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("TestWindow", "Tab 2"))
        self.btn_openValidationSet.setText(_translate("TestWindow", "Validation Set"))

