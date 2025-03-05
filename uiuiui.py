import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import  QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from myui_1 import Ui_MainWindow
import tensorflow as tf
from tensorflow.keras.preprocessing import image

class ui_main(QMainWindow,Ui_MainWindow):
    #初始化构建窗口
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.img_path = None
        self.model = tf.keras.models.load_model("models/moblieV2_5.h5")
        self.class_names = ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']
        self.resize(700,500)
        self.pushButton_3.clicked.connect(lambda: self.exit())
        self.pushButton.clicked.connect(self.openimage)
        self.pushButton_2.clicked.connect(self.predict_image)
    def exit(self):
        self.close()

    # 拖动
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    # 读取图片,将路径存入txt文件中
    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image files(*.jpg , *.png)')
        jpg = QtGui.QPixmap(imgName).scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(jpg)
        self.img_path = imgName #得到推理的图片上显示

    def predict_image(self):
        img = image.load_img(self.img_path, target_size=(224, 224))  # 加载并调整图像大小
        img_array = image.img_to_array(img)  # 转换为数组
        images = np.array(img_array)
        images = np.expand_dims(img_array, axis=0)
        try:
            outputs = self.model.predict(images)
            result_index = np.argmax(outputs)
            predict_number = np.max(outputs)*100
            result = self.class_names[result_index]
            self.label_2.setText(f"预测结果为{result}的概率为{predict_number:.2f}%")
        except Exception as e:
            print(f"Error during prediction: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ui_main()
    window.show()
    sys.exit(app.exec_())
