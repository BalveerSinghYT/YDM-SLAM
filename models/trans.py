from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")

# Export the model to ONNX format
# opset=21: 兼容 ONNX Runtime 1.20.x (最高支持 opset 21)
# simplify=True: 简化模型结构
model.export(format="onnx", opset=21, simplify=True)

print("✅ ONNX export completed: best.onnx (opset=21)")

# 不运行推理测试（避免类别映射问题）
# 如需测试，请使用原始 .pt 模型：
# results = model("https://ultralytics.com/images/bus.jpg")
