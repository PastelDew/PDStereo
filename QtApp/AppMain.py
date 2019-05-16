import sys
from QtUI import QtMain
from QtApp.MainWindow import MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication

def runApplication():
    app = QApplication(sys.argv)

    mainWindow = MainWindow()
    mainForm = QtMain.Ui_MainWindow()
    mainForm.setupUi(mainWindow)
    mainForm.retranslateUi(mainWindow)
    mainWindow.initialize(mainForm)

    mainWindow.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(runApplication())