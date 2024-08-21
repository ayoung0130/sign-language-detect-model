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
save_path = os.path.join(base_dir, f"npy/0_9")
flip_save_path = os.path.join(base_dir, f"npy_flip/0_9")

def process_video(video_path, flip=False):
    cap = cv2.VideoCapture(video_path)
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        # 랜드마크, 프레임 가져오기
        d, processed_frame = get_landmarks(frame)

        if d is not None:
            data.append(d)

        cv2.imshow('video', processed_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    return np.array(data)

def save_data(action, data, flip=False):
    created_time = int(time.time())
    if flip:
        np.save(os.path.join(flip_save_path, f'flip_{action}_{created_time}'), data)
    else :
        np.save(os.path.join(save_path, f'{action}_{created_time}'), data)

def process_action_videos(action, folder_path, idx, flip=False):
    data = []
    video_num = 0

    for video_file in os.listdir(folder_path):
        video_num += 1
        video_path = os.path.join(folder_path, video_file)
        video_data = process_video(video_path, flip)
        if len(video_data) > 0:
            video_data = np.c_[video_data, np.full(video_data.shape[0], idx)]  # 인덱스 추가
            data.append(video_data)

    if data:
        data = np.concatenate(data)
        print(f"data[100]: {data[100]}")
        print(f"data shape: {action}, {data.shape}")
        print(f"영상 개수: {video_num}")
        save_data(action, data, flip)

# Original and Flip processing
for idx in range(10):
    action = actions[idx]
    folder_path = os.path.join(base_dir, f"video/resized_video_{idx}")

    # Process original videos
    process_action_videos(action, folder_path, idx, flip=False)

    # Process flipped videos
    process_action_videos(action, folder_path, idx, flip=True)

# 사용된 함수, 자원 해제
cv2.destroyAllWindows()