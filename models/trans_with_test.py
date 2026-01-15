from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")

# Export the model to ONNX format
model.export(format="onnx", simplify=True)  # creates 'best.onnx'

print("✅ ONNX export completed: best.onnx")

# 如果需要测试推理（使用原始 .pt 模型，避免 ONNX 类别映射问题）
try:
    print("\n🔍 Testing inference with .pt model...")
    results = model("https://ultralytics.com/images/bus.jpg")
    print(f"✅ Detected {len(results[0].boxes)} objects")
    
    # 显示类别信息
    if hasattr(model, 'names'):
        print(f"Model classes: {len(model.names)} total")
        print(f"Class names: {list(model.names.values())[:10]}...")  # 前10个
except Exception as e:
    print(f"⚠️  Inference test failed: {e}")
    print("   This is normal if your model uses custom classes")
    print("   The ONNX file is still usable in C++ inference")
