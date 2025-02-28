import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis

# 初始化 ArcFace 模型
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 加载人脸数据库
face_data = np.load("face_embeddings1.npy", allow_pickle=True).item()

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

# 使用示例
if __name__ == "__main__":
    new_img_path = r"data/database/aiqinggongyu8.jpg"  # 替换为实际图像路径
    output_img_path = r"data/data-ouput/output10.jpg"  # 替换为输出图像路径
    recognize_faces(new_img_path, app, face_data, threshold=24, save_path=output_img_path)

cv2.imshow("img",cv2.imread(output_img_path))
cv2.waitKey(0)
cv2.destroyAllWindows()