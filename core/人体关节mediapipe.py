import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 检测姿态
    results = pose.process(rgb_frame)

    # 在图像上绘制关键点
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 显示图像
    cv2.imshow("Pose Estimation", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
