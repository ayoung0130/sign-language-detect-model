import cv2
import os
from dotenv import load_dotenv
import glob

# 수어 데이터셋 영상 크기와 비율을 조정하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

video_path = os.path.join(base_dir, "test_video/1")

output_folder = os.path.join(base_dir, "resize_test_video")
os.makedirs(output_folder, exist_ok=True)

# 너비, 높이 설정
# 원본 영상과 비율은 동일하게 유지. 1280x720 (1.78 : 1)
new_width = 455
new_height = 256

# 픽셀 크기 설정
distance_from_center = 200

center_x = new_width // 2
center_y = new_height // 2

left_x = center_x - distance_from_center
right_x = center_x + distance_from_center

# glob을 사용해 영상 파일 목록 가져오기 (.mp4 형식 기준)
video_files = glob.glob(f"{video_path}/*.mp4")

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)

    output_path = os.path.join(output_folder, f"{os.path.basename(video_file)}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # .avi 확장자로 설정
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (right_x - left_x, new_height))   # 30.0은 프레임

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 해상도 조절
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 픽셀 크기 조절
        cropped_frame = resized_frame[:, left_x:right_x]

        # 조절된 프레임을 새로운 동영상에 추가
        out.write(cropped_frame)

        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', cropped_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # 작업 완료 후 해제
    cap.release()
    out.release()

print("영상 크기 조절 및 저장 완료")
