import sys
from QtUI import QtMain
from PyQt5.QtWidgets import QMainWindow, QApplication

from CtrlMain import CtrlMain as CMain

def runApplication():
    app = QApplication(sys.argv)

    mainWindow = QMainWindow()
    mainForm = QtMain.Ui_MainWindow()
    mainForm.setupUi(mainWindow)
    mainForm.retranslateUi(mainWindow)

    cMain = CMain(mainForm)

    mainWindow.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(runApplication())