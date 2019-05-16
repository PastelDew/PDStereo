from QtUI.QtMain import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
import RGB-D

class CtrlMain():
    def __init__(self, mainForm: Ui_MainWindow):
        assert type(mainForm) is Ui_MainWindow
        self.mainForm = mainForm
        mainForm.spinBox_cam_left.valueChanged.connect(self.event_spinBox_cam_left_changed)

    def event_spinBox_cam_left_changed(self):
        print("TEST")