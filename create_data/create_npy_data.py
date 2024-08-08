import cv2
import numpy as np
import os, time
from setting import actions
from data_processing.landmark_processing import get_landmarks
from dotenv import load_dotenv

# 영상을 넘파이 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 동영상 파일 설정
for idx in range(0, 10):

    # 데이터 저장 경로
    save_path = os.path.join(base_dir, f"npy/landmarks_angle/{idx}")
    flip_save_path = os.path.join(base_dir, f"npy_flip/landmarks_angle/{idx}")

    action = actions[idx]
    folder_path = os.path.join(base_dir, f"video/resized_video_{idx}")

    video_num = 0

    for video_file in os.listdir(folder_path):
        video_num += 1
        # 동영상 불러오기
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        data = []
        flipped_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 랜드마크, 프레임 가져오기
            d, frame = get_landmarks(frame)

            if d is not None:
                # 인덱스 추가
                d = np.append(d, idx)
                
                # 전체 데이터 배열에 추가
                data.append(d)

            # 화면에 표시
            cv2.imshow('video', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()  # 비디오 캡처 자원 해제
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 랜드마크, 프레임 가져오기
            d_flipped, flipped_frame = get_landmarks(cv2.flip(frame, 1))

            if d_flipped is not None:
                # 인덱스 추가
                d_flipped = np.append(d_flipped, idx)
                
                # 전체 데이터 배열에 추가
                flipped_data.append(d_flipped)

            # 화면에 표시
            cv2.imshow('flipped_video', flipped_frame)
            if cv2.waitKey(1) == ord('q'):
                break
            
        cap.release()  # 비디오 캡처 자원 해제

        # 넘파이 배열로 생성
        data = np.array(data)
        flipped_data = np.array(flipped_data)

        print("data shape: ", action, video_num, data.shape)
        print("flipped_data shape: ", action, video_num, flipped_data.shape)

        # 넘파이 데이터 저장
        created_time = int(time.time())
        np.save(os.path.join(save_path, f'{action}_{video_num}_{created_time}'), data)
        np.save(os.path.join(flip_save_path, f'flip_{action}_{video_num}_{created_time}'), flipped_data)

    print(f"Index {idx} 영상 개수: ", video_num)

    # 사용된 함수, 자원 해제
    cv2.destroyAllWindows()