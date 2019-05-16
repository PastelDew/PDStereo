# -*- coding: utf-8 -*-


import numpy
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class ImageDialog(QDialog):
    def __init__(self, qImage, parent=None):
        super(ImageDialog, self).__init__(parent)

        self.mQImage = qImage

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self.mQImage)
        painter.end()