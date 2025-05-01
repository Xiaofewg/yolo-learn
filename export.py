from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO(r"E:\yolov11\xiaosai.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
#ncnn_model = YOLO(r"E:\ultralytics\runs\train\exp\weights\last.pt")
