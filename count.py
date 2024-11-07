from ultralytics import YOLO
import cv2

video_path = "rtsp://localhost:8554/mystream"
model_path = "models/yolo11n.pt"
# 初始化统计变量
line_position_scale = 0.5  # 黄线的垂直位置（y坐标）
up_count = 0  # 从下往上穿越黄线的人数
down_count = 0  # 从上往下穿越黄线的人数
tracked_objects = {}  # 用于跟踪每个对象的上一个位置

model = YOLO(model_path)  # load a custom model



cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, 10)  # 设置为 10 FPS，减少每秒的帧数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# 循环处理视频帧
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("无法读取视频流")
        break

    # 在当前帧上绘制黄线
    frame_height, frame_width = frame.shape[:2]
    line_position = int(frame_height*line_position_scale) 
    cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 255, 255), 2)

    # 运行 YOLO 检测
    results = model.track(frame, persist=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            if box.id and model.names[class_id] == "car":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                center_y = (y1 + y2) // 2
                track_id = box.id.item()
                if track_id in tracked_objects:
                    previous_y = tracked_objects[track_id]
                    if previous_y < line_position and center_y >= line_position:
                        down_count += 1
                    elif previous_y > line_position and center_y <= line_position:
                        up_count += 1
                    # cv2.line(frame, (0, center_y), (frame_width, center_y), (0, 255, 0), 2)
                tracked_objects[track_id] = center_y
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[class_id]} {track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 在帧上显示人数统计
    cv2.putText(frame, f"Up Count: {up_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Down Count: {down_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示处理后的帧
    cv2.imshow("People Counting", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放视频流和销毁窗口
cap.release()
cv2.destroyAllWindows()