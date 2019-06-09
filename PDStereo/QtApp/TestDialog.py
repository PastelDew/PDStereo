# -*- coding: utf-8 -*-


import numpy
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PDStereo.QtApp.QtUI.QtTestWindow import Ui_TestWindow


class TestDialog(QDialog):
    def __init__(self, parent=None):
        super(TestDialog, self).__init__(parent)

    def initialize(self, mainForm: Ui_TestWindow):
        self.mainForm = mainForm
        