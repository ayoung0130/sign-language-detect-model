import cv2
import numpy as np
import os
from data_processing.landmark_processing import get_landmarks
from dotenv import load_dotenv

# 테스트 영상을 넘파이 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

folder_path = os.path.join(base_dir, f"test_video_10words/3")

# 데이터 저장 경로
save_path = os.path.join(base_dir, "test_npy")

# flip 여부를 결정하는 리스트
flip_options = [False, True]

for flip in flip_options:
    video_num = 0

    for video_file in os.listdir(folder_path):
        video_num += 1
        video_path = os.path.join(folder_path, video_file)  # 동영상 불러오기
        base_name = os.path.splitext(os.path.basename(video_path))[0]  # 확장자를 제거한 이름
        cap = cv2.VideoCapture(video_path)

        data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if flip :
                frame = cv2.flip(frame, 1)
            
            # 랜드마크, 프레임 가져오기
            d, frame = get_landmarks(frame)

            if d is not None:
                data.append(d)  # 전체 데이터 배열에 추가

            # 화면에 표시
            cv2.imshow('test video', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()  # 비디오 캡처 자원 해제

        # 넘파이 배열로 생성
        data = np.array(data)
        print("data[10]: ", data[10])
        print("data shape: ", data.shape)

        # 넘파이 데이터 저장
        if flip:
            np.save(os.path.join(save_path, f'flip_{base_name}'), data)
        else:
            np.save(os.path.join(save_path, f'{base_name}'), data)

cv2.destroyAllWindows()