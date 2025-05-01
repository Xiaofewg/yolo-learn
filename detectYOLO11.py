import cv2
from ultralytics import YOLO
import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Load the YOLO model
        self.model = YOLO("best_shensai.pt", task="detect")

        # Open the camera
        self.cap = cv2.VideoCapture(0)

        # Create a QLabel to display the video frames
        self.label = QLabel(self)

        # Create a layout and add the label to it
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Create a timer to update the video frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 milliseconds

    def update_frame(self):
        # Read a frame from the video
        success, frame = self.cap.read()

        if success:
            # Run YOLO inference on the frame
            results = self.model(frame)
            summary = results[0].summary()
            for item in summary:
               if item["name"] == 'person':
                   print(item["name"], item["confidence"])
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Convert the OpenCV image to a QImage
            height, width, channel = annotated_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_BGR888)

            # Convert the QImage to a QPixmap
            pixmap = QPixmap.fromImage(q_img)

            # Set the QPixmap to the QLabel
            self.label.setPixmap(pixmap)

    def closeEvent(self, event):
        # Release the video capture object when the window is closed
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
