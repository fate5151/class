import cv2
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

# 初始化 InsightFace
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 使用 GPU，ctx_id=-1 使用 CPU

# 读取图像
img = cv2.imread(r"data\database\aiqinggongyu7.jpg")

# 检测人脸
faces = app.get(img)
print(f"检测到 {len(faces)} 张人脸")

for i, face in enumerate(faces):
    bbox = face.bbox.astype(int)  # 获取人脸边界框
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # 提取特征向量（512维）
    embedding = face.embedding
    print(f"人脸 {i+1} 的特征向量: {embedding}")

    



# 将BGR图像转换为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 使用 matplotlib 显示图像
plt.imshow(img_rgb)
plt.axis('off')  # 隐藏坐标轴
plt.show()

# 保存结果
cv2.imwrite(r"data\data-ouput\output11.png", img)


# # 假设数据库特征向量存储在 `database` 中
# database = {
#     "person1": vec1,
#     "person2": vec2,
#     # ...
# }

# # 输入人脸的特征向量
# input_face_vec = faces[0].embedding

# # 比较并找到最匹配的身份
# best_match = None
# highest_similarity = 0
# for name, db_vec in database.items():
#     similarity = cosine_similarity(input_face_vec, db_vec)
#     if similarity > highest_similarity:
#         highest_similarity = similarity
#         best_match = name

# print(f"最匹配的身份: {best_match}，相似度: {highest_similarity}")
