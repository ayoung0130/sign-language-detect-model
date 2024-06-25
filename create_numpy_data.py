import cv2
import numpy as np
import os, time
from setting import actions
from landmark_processing import get_landmarks
from dotenv import load_dotenv

# 영상을 넘파이 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 오른쪽(47), 왼쪽(48)은 flip X
idx_list = [11, 13, 19, 42, 43, 49]

# 동영상 파일 설정
for idx in idx_list:

    action = actions[idx]
    folder_path = os.path.join(base_dir, f"video/resized_video_{idx}")

    # 데이터 저장 경로
    save_path = os.path.join(base_dir, "angle")
    flip_save_path = os.path.join(base_dir, "angle_flip")

    data = []
    flip_data = []
    video_num = 0

    for video_file in os.listdir(folder_path):
        video_num += 1
        # 동영상 불러오기
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 랜드마크, 프레임 가져오기
            d, original_frame = get_landmarks(frame, True)
            
            # 인덱스 추가
            d = np.append(d, idx)
        
            # 전체 데이터 배열에 추가
            data.append(d)

            # 좌우반전된 프레임 처리
            flipped_frame = cv2.flip(frame, 1)
            d_flipped, flipped_frame = get_landmarks(flipped_frame, True)
            d_flipped = np.append(d_flipped, idx)
            flip_data.append(d_flipped)

            # 화면에 표시
            cv2.imshow('Original', original_frame)
            cv2.imshow('Flipped', flipped_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # 넘파이 배열로 생성
    data = np.array(data)
    # flip_data = np.array(flip_data)
    print("data shape: ", action, data.shape)
    print(data[50])
    print("영상 개수: ", video_num)

    created_time = int(time.time())

    # 넘파이 데이터 저장
    np.save(os.path.join(save_path, f'{action}_{created_time}'), data)
    np.save(os.path.join(flip_save_path, f'flip_{action}_{created_time}'), flip_data)

    # 사용된 함수, 자원 해제
    cv2.destroyAllWindows()