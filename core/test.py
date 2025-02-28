import onnxruntime as ort
import numpy as np
from PIL import Image
import ctypes
import torchvision.transforms as transforms

# 加载模型
ctypes.cdll.LoadLibrary(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cudnn64_8.dll")
model_path = r"onnx_model/resnet50-v2-7.onnx"
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# 检查输入/输出
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Input name:", input_name)
print("Output name:", output_name)

# 图片预处理
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0).numpy()
    return image_tensor

# 加载图片
image_path = r"dog.jpg"  # 替换为你的图片路径
input_data = preprocess_image(image_path)

# 推理
output = session.run([output_name], {input_name: input_data})
output = np.array(output)

# 解析输出
predicted_class = np.argmax(output, axis=1)[0]
print(f"Predicted class ID: {predicted_class}")
