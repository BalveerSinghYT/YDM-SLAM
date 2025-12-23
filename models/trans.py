from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11s-seg.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("yolo11s-seg.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")
