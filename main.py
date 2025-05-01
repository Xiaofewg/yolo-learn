#!/usr/bin python

'''
此main.py文件运行于树莓派5, 
实时检测在yolo11n(last_ncnn_model)上有15帧, 
为了提高识别精度采用yolo11s(best_shensai_ncnn_model)模型, 帧率为8帧左右
'''

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                              QTextEdit, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QFileDialog, QLabel)
from PySide6.QtCore import Signal, QUrl, Qt, QTimer
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtGui import QImage, QPixmap, QFont
import cv2
from ultralytics import YOLO
from gpiozero import DigitalOutputDevice, DigitalInputDevice
import time
import threading
import serial

ser = serial.Serial(
    port='/dev/ttyAMA0',  # UART 设备文件路径
    baudrate=9600,        # 波特率
    timeout=1             # 超时时间（秒）
)

# 定义要读取电平的引脚编号
pin_number1 = 17
pin_number2 = 18
pin_number3 = 27
pin_number4 = 22

# 创建DigitalInputDevice对象，用于读取指定引脚的电平
input_pin1 = DigitalInputDevice(pin_number1)
input_pin2 = DigitalInputDevice(pin_number2)
input_pin3 = DigitalInputDevice(pin_number3)
input_pin4 = DigitalInputDevice(pin_number4)
pin_types = [input_pin1, input_pin2, input_pin3, input_pin4]

garbage_flag_count = {"可回收垃圾": 0, "有害垃圾": 0, "厨余垃圾": 0, "其他垃圾": 0}
garbage_count = {"可回收垃圾": 0, "有害垃圾": 0, "厨余垃圾": 0, "其他垃圾": 0}
garbage_images = {
    "可回收垃圾": "/home/pi/yolov5-USB/Recyclable.png",
    "有害垃圾": "/home/pi/yolov5-USB/HazardousWaste.png",
    "厨余垃圾": "/home/pi/yolov5-USB/FoodWaste.png",
    "其他垃圾": "/home/pi/yolov5-USB/ResidualWaste.png"
}

flag = 0
index = 0
same_name = ''
same_count = 0
csb_flag = 0
gpio16 = DigitalOutputDevice(16)

class App(QMainWindow):

    detected = Signal(list)
    serial_data_received = Signal(bytes)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("多功能检测系统")
        self.setStyleSheet("background-image: url(bjt3.jpg); background-repeat: no-repeat; background-position: center;")
        self.setFixedSize(1024, 600)
        self.move(0, 0)
        #self.showFullScreen()
        # self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

        # 创建主布局
        main_layout = QHBoxLayout()

        # 用于显示视频帧的 QLabel
        self.video_label = QLabel()

        # 全局字体设置
        self.font = QFont()
        self.font.setPointSize(20)

        self.text_edit = QTextEdit("检测结果")
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(self.font)
        self.text_edit.setFixedSize(400, 240)

        # 控制按钮的垂直&水平布局
        control_layout = QVBoxLayout()
        picture_count_layout1 = QHBoxLayout()
        picture_count_layout2 = QHBoxLayout()

        # 创建一个字典来存储所有的 count_label 另一个字典存储所有warning_label
        self.garbage_labels = {}
        self.warning_labels = {}
        
        garbage_types = ["可回收垃圾", "有害垃圾", "厨余垃圾", "其他垃圾"]
        for i, garbage_type in enumerate(garbage_types):
            picture_warning, count_label = self.create_garbage_widgets(garbage_type)
            if i < 2:
                picture_count_layout1.addLayout(picture_warning)
                picture_count_layout1.addWidget(count_label)
            else:
                picture_count_layout2.addLayout(picture_warning)
                picture_count_layout2.addWidget(count_label)

        control_layout.addWidget(self.text_edit)
        control_layout.addLayout(picture_count_layout1)
        control_layout.addLayout(picture_count_layout2)

        # 主要布局设置
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.video_label)  # 显示视频帧

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 连接信号
        self.detected.connect(self.update_text_edit)

        # 初始化变量
        # Load the YOLO model
        self.model = YOLO("best_shensai2_ncnn_model", task="detect")
        self.cap_init_detect = cv2.VideoCapture(0)
        self.is_detecting = False

        # 启动串口读取线程
        self.serial_thread = threading.Thread(target=self.read_serial_data)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        self.serial_data_received.connect(self.handle_serial_data)

        # 超声波满载检测线程
        self.read_handle_pin_thread = threading.Thread(target=self.read_handle_pin)
        self.read_handle_pin_thread.daemon = True
        self.read_handle_pin_thread.start()

        #LINK START!!!!!
        self.replay_count = 0
        self.toggle_video()    #程序启动自动无限循环播放视频
        self.timer3 = QTimer(self)
        self.timer3.timeout.connect(self.only_detect)
        self.timer3.start(1000)
    
    def create_garbage_widgets(self, garbage_type):
        image_label = QLabel()
        image_label.setFixedSize(82, 100)
        pixmap = QPixmap(garbage_images[garbage_type])
        image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.KeepAspectRatio))
        image_label.setAlignment(Qt.AlignCenter)

        warning_label = QLabel()
        warning_label.setFixedSize(82, 50)
        warning_label.setFont(self.font)
        self.warning_labels[garbage_type] = warning_label

        count_label = QLabel()
        count_label.setFont(self.font)
        self.garbage_labels[garbage_type] = count_label

        picture_warning = QVBoxLayout()
        picture_warning.addWidget(image_label)
        picture_warning.addWidget(warning_label)

        return picture_warning, count_label

    def handle_serial_data(self, data):
        # 将接收到的字节数据转换为布尔值
        print(self.is_detecting)
        if data == b'1':
            self.is_detecting = True
        else:
            self.is_detecting = False

    def read_serial_data(self):
        while True:
            # 从串口读取一个字节的数据
            data = ser.read(1)
            ser.reset_input_buffer()
            # 发送信号到主线程处理数据
            self.serial_data_received.emit(data)
            time.sleep(0.065)

    def read_handle_pin(self):
        global input_pin1, input_pin2, input_pin3, input_pin4
        global pin_types
        garbage_types = [ "厨余垃圾","可回收垃圾","其他垃圾","有害垃圾"]
        pin_values = [0, 0, 0, 0]
        try:
            while True:
                for i in range(len(pin_types)):
                    pin_values[i] = pin_types[i].value  # 读取引脚当前的电平值，高电平返回1，低电平返回0
                    if pin_values[i] == 1:
                        self.warning_labels[garbage_types[i]].setText(f"已满载")
                    else:
                        self.warning_labels[garbage_types[i]].setText(f"")
                '''if sum(pin_values) >= 1:
                    gpio16.on()     #满载蜂鸣器报警
                    time.sleep(1.5)
                    gpio16.off()'''
                time.sleep(0.5)
        except Exception as e:
            print(f"read_handle_pin 线程出现异常: {e}")

    def only_detect(self):
        success, frame = self.cap_init_detect.read()
        if success:
            results = self.model(frame)
            if  results[0].summary():
                self.timer2.stop()
                self.timer3.stop()
                self.cap.release()
                self.cap_init_detect.release()
                print("检测到物体，停止播放视频")
                self.start_detection()

    def toggle_video(self):
        self.file_path = '/home/pi/yolo11/LJClassification.mp4'
        self.cap = cv2.VideoCapture(self.file_path)

        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.update_frame_video)
        self.timer2.start(30)

    def update_frame_video(self):  # 更新视频帧
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
        else:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.file_path)
            ret, frame = self.cap.read()
            
    def update_frame(self):        # 更新检测帧
        # Read a frame from the video
        success, frame = self.cap_detect.read()

        if success:
            # Run YOLO inference on the frame
            results = self.model(frame)
            '''if not results[0].summary():
                self.replay_count += 1
                if self.replay_count > 180:  # 15秒后重新播放视频
                    self.toggle_video()
                    self.timer3.start(500)
                    self.cap_detect.release()
            else:
                self.replay_count = 0'''

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Convert the OpenCV image to a QImage
            height, width, channel = annotated_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            
            # Set the QPixmap to the QLabel 
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)

            self.detected.emit(results[0].summary())

    def start_detection(self):
        # Open the camera
        self.cap_detect = cv2.VideoCapture(0)

        # Create a timer to update the video frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_text_edit(self, summary):
        global flag
        global garbage_flag_count
        global index
        global same_name
        global same_count
        print(garbage_flag_count)
        #if summary: #debug
        if summary and self.is_detecting:   # 从串口获取数据——1表示执行机构空闲，0表示执行机构工作中
            for item in summary:
                if item["name"] == same_name:
                    same_count += 1
                else:
                    same_name = item["name"]
                    same_count = 0
                if item["name"] == "plastic" or item["name"] == "metal":
                    garbage_flag_count["可回收垃圾"] += 1
                elif item["name"] == "battery" or item["name"] == "drug":
                    garbage_flag_count["有害垃圾"] += 1
                elif item["name"] == "potato" or item["name"] == "carrot" or item["name"] == "daikon":
                    garbage_flag_count["厨余垃圾"] += 1
                else:
                    garbage_flag_count["其他垃圾"] += 1
                flag += 1
        #elif self.is_detecting:

        if flag > 40 or same_count > 15:
            index += 1
            max_category = max(garbage_flag_count, key=lambda k: garbage_flag_count[k])
            match max_category:
                case "可回收垃圾":
                    ser.write(str(1).encode())
                case "有害垃圾":
                    ser.write(str(2).encode())
                case "厨余垃圾":
                    ser.write(str(3).encode())
                case "其他垃圾":
                    ser.write(str(4).encode())   #debug
            text = f"{index}. {max_category} 1 OK!"    #debug
            self.text_edit.append(text)
            flag = 0
            same_count = 0
            garbage_flag_count = {"可回收垃圾": 0, "有害垃圾": 0, "厨余垃圾": 0, "其他垃圾": 0}
            garbage_count[max_category] += 1
            self.garbage_labels[max_category].setText(f"数量：{garbage_count[max_category]}")
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:  # 按下Esc键退出程序
            if self.timer.isActive():
                self.timer.stop()
            if self.cap:
                self.cap.release()
            sys.exit()
            QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
