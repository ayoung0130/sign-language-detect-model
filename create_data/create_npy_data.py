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
save_path = os.path.join(base_dir, "npy/landmarks")
save_path_visibility = os.path.join(base_dir, "npy/landmarks_visibility")
save_path_angle = os.path.join(base_dir, "npy/landmarks_angle")
save_path_visibility_angle = os.path.join(base_dir, "npy/landmarks_visibility_angle")

flip_save_path = os.path.join(base_dir, "npy_flip/landmarks")
flip_save_path_visibility = os.path.join(base_dir, "npy_flip/landmarks_visibility")
flip_save_path_angle = os.path.join(base_dir, "npy_flip/landmarks_angle")
flip_save_path_visibility_angle = os.path.join(base_dir, "npy_flip/landmarks_visibility_angle")

# flip 여부
flip = True

# 동영상 파일 설정
for idx in range(0, 10):

    action = actions[idx]
    folder_path = os.path.join(base_dir, f"video/resized_video_{idx}")
    
    data = []
    data_visibility = []
    data_angle = []
    data_visibility_angle = []

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

            if flip :
                frame = cv2.flip(frame, 1)

            # output : tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]], Any]
            d, d_visibility, d_angle, d_visibility_angle, frame = get_landmarks(frame)

            if d is not None:
                # 인덱스 추가
                d = np.append(d, idx)
                d_visibility = np.append(d_visibility, idx)
                d_angle = np.append(d_angle, idx)
                d_visibility_angle = np.append(d_visibility_angle, idx)
                
                # 전체 데이터 배열에 추가
                data.append(d)
                data_visibility.append(d_visibility)
                data_angle.append(d_angle)
                data_visibility_angle.append(d_visibility_angle)

            # 화면에 표시
            cv2.imshow('video', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            
        cap.release()  # 비디오 캡처 자원 해제

    # 넘파이 배열로 생성
    data = np.array(data)
    data_visibility = np.array(data_visibility)
    data_angle = np.array(data_angle)
    data_visibility_angle = np.array(data_visibility_angle)

    print(data[100])
    print("data shape: ", action, data.shape)
    print("영상 개수: ", video_num)

    created_time = int(time.time())

    # 넘파이 데이터 저장
    if flip :
        np.save(os.path.join(flip_save_path, f'flip_{action}_{created_time}'), data)
        np.save(os.path.join(flip_save_path_visibility, f'flip_{action}_{created_time}'), data_visibility)
        np.save(os.path.join(flip_save_path_angle, f'flip_{action}_{created_time}'), data_angle)
        np.save(os.path.join(flip_save_path_visibility_angle, f'flip_{action}_{created_time}'), data_visibility_angle)
    else :
        np.save(os.path.join(save_path, f'{action}_{created_time}'), data)
        np.save(os.path.join(save_path_visibility, f'{action}_{created_time}'), data_visibility)
        np.save(os.path.join(save_path_angle, f'{action}_{created_time}'), data_angle)
        np.save(os.path.join(save_path_visibility_angle, f'{action}_{created_time}'), data_visibility_angle)

    # 사용된 함수, 자원 해제
    cv2.destroyAllWindows()