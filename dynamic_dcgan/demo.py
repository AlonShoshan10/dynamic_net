from PySide2.QtWidgets import QApplication
from gui.main_dynamic_dcgan_widget import MainDynamicDcGanWidget

app = QApplication([])
ex = MainDynamicDcGanWidget()
app.exec_()

