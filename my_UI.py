import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import cv2
from PIL import Image
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于机器学习的花卉识别系统")

        #模型
        # self.model = tf.keras.models.load_model("models/mobileV2.h5")
        self.to_predict_name = "test.jpg"
        self.class_names = ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']
        self.resize(800,600)
        self.initUI()

    def initUI(self):
        #主窗口
        main_widget = QWidget()
        #主要控件
        main_layout=QHBoxLayout()
        font = QFont('宋体',15)

        #左端控件
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        img_title = QLabel("测试图像")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec())


