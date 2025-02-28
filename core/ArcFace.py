import os
import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis

# 初始化 ArcFace 模型
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 使用 GPU，ctx_id=-1 使用 CPU

# 数据库路径和输出文件
database_dir = r"data\data-collect/"  # 数据库文件夹，每个人一个子文件夹
output_file = "face_embeddings4.npy"
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

# 在图像上绘制人脸框并标注名字
def draw_faces(img, faces, names):
    for face, name in zip(faces, names):
        bbox = face.bbox.astype(int)  # 获取人脸边界框
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            img,
            name,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

# 识别新图像中的人脸
def recognize_faces(image_path, app, face_data, threshold=1.0, save_path="output.jpg"):
    print(f"处理图像: {image_path}")
    img = cv2.imread(image_path)

    # 检测所有人脸
    faces = app.get(img)
    if len(faces) == 0:
        print("未检测到人脸")
        return

    recognized_names = []
    for face in faces:
        embedding = face.embedding

        # 在数据库中查找最相似的人
        best_name = "Unknown"
        best_distance = float("inf")
        for name, stored_embedding in face_data.items():
            distance = norm(embedding - stored_embedding)
            print(f"距离到 {name}: {distance:.4f}")
            if distance < best_distance:
                best_distance = distance
                best_name = name

        if best_distance < threshold:
            recognized_names.append(best_name)
            print(f"匹配成功: {best_name} (距离: {best_distance:.4f})")
        else:
            recognized_names.append("Unknown")
            print(f"未匹配到任何身份 (最近距离: {best_distance:.4f})")

    draw_faces(img, faces, recognized_names)
    cv2.imwrite(save_path, img)
    print(f"结果图像已保存到: {save_path}")


# 测试识别新图像
new_img_path = r"data\data\image2k.png"  # 替换为实际图像路径
output_img_path = "output9.jpg"  # 替换为输出图像路径
recognize_faces(new_img_path, app, face_data, threshold=21, save_path=output_img_path)
