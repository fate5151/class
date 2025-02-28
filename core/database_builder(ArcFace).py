import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 初始化 ArcFace 模型
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 使用 GPU，ctx_id=-1 使用 CPU

# 数据库路径和输出文件
database_dir = r"data/data-collect/"  # 数据库文件夹，每个人一个子文件夹
output_file = "face_embeddings1.npy"
face_data = {}

# 函数：提取多张图像的平均特征向量
def get_average_embedding(images, app):
    embeddings = []
    for img in images:
        faces = app.get(img)
        if len(faces) > 0:
            embeddings.append(faces[0].embedding)
    if embeddings:
        return np.mean(embeddings, axis=0)
    return None

# 构建人脸数据库（提取每个人的特征向量）
print("构建数据库中...")
for person_name in os.listdir(database_dir):
    person_dir = os.path.join(database_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    images = []
    for file_name in os.listdir(person_dir):
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(person_dir, file_name)
            img = cv2.imread(img_path)
            images.append(img)

    # 获取多张图片的平均特征
    embedding = get_average_embedding(images, app)
    if embedding is not None:
        face_data[person_name] = embedding
        print(f"{person_name} 的特征提取成功")
    else:
        print(f"{person_name} 没有成功提取到特征")

# 保存特征向量到文件
np.save(output_file, face_data)
print("数据库构建完成并保存到文件")