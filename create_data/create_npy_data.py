import cv2
import numpy as np
import os, time
from setting import actions
from data_processing.landmark_processing import get_landmarks
from dotenv import load_dotenv

# 영상을 넘파이 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 데이터 저장 경로
save_path = os.path.join(base_dir, f"npy/20_29")
flip_save_path = os.path.join(base_dir, f"npy_flip/20_29")

# flip 여부를 결정하는 리스트
flip_options = [False, True]

for flip in flip_options:
    for idx in range(20, 30):
        action = actions[idx]
        folder_path = os.path.join(base_dir, f"video/resized_video_{idx}")
        video_num = 0
        data = []

        for video_file in os.listdir(folder_path):
            video_num += 1
            video_path = os.path.join(folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if flip:
                    frame = cv2.flip(frame, 1)

                # 랜드마크, 프레임 가져오기
                d, frame = get_landmarks(frame)

                if d is not None:
                    d = np.append(d, idx)
                    data.append(d)

                cv2.imshow('video', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        data = np.array(data)
        print(f"data[100]: {data[100]}")
        print(f"data shape: {action}, {data.shape}")
        
        # 데이터 저장
        created_time = int(time.time())

        if flip:
            np.save(os.path.join(flip_save_path, f'flip_{action}_{created_time}'), data)
        else :
            np.save(os.path.join(save_path, f'{action}_{created_time}'), data)

    print(f"영상 개수: {video_num}")

# 사용된 함수, 자원 해제
cv2.destroyAllWindows()