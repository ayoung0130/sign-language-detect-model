import cv2
import numpy as np
import os
from landmark_processing import get_landmarks, get_landmarks_visibility
from dotenv import load_dotenv

# 테스트 영상을 넘파이 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

folder_path = os.path.join(base_dir, f"10_words")

# 데이터 저장 경로
save_path = os.path.join(base_dir, "test_npy/landmarks_visibility")

video_num = 0

for video_file in os.listdir(folder_path):
    video_num += 1
    # 동영상 불러오기
    video_path = os.path.join(folder_path, video_file)
    base_name = os.path.splitext(os.path.basename(video_path))[0]  # 확장자를 제거한 이름
    cap = cv2.VideoCapture(video_path)

    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 랜드마크, 프레임 가져오기
        d, frame = get_landmarks_visibility(frame)
    
        # 전체 데이터 배열에 추가
        if d is not None:
            data.append(d)

        # 화면에 표시
        cv2.imshow('test video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()  # 비디오 캡처 자원 해제

    # 넘파이 배열로 생성
    data = np.array(data)
    print("data shape: ", data.shape)

    # 넘파이 데이터 저장
    np.save(os.path.join(save_path, f'{base_name}'), data)

cv2.destroyAllWindows()