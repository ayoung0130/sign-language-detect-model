import cv2
import numpy as np
import os
from landmark_processing import get_test_landmarks_visibility
from dotenv import load_dotenv

# 테스트 영상을 넘파이 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

folder_path = os.path.join(base_dir, f"10_words_3")

# 데이터 저장 경로
save_path = os.path.join(base_dir, "test_npy/landmarks")
save_path_visibility = os.path.join(base_dir, "test_npy/landmarks_visibility")

# flip 여부
flip = False

video_num = 0

for video_file in os.listdir(folder_path):
    video_num += 1
    # 동영상 불러오기
    video_path = os.path.join(folder_path, video_file)
    base_name = os.path.splitext(os.path.basename(video_path))[0]  # 확장자를 제거한 이름
    cap = cv2.VideoCapture(video_path)

    data = []
    data_visibility = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if flip :
            frame = cv2.flip(frame, 1)
        
        # 랜드마크, 랜드마크 + 가시성정보, 프레임 가져오기
        # output : tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]], Any]
        d, d_visibility, frame = get_test_landmarks_visibility(frame)

        if d is not None and d_visibility is not None:
            # 전체 데이터 배열에 추가
            data.append(d)
            data_visibility.append(d_visibility)

        # 화면에 표시
        cv2.imshow('test video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()  # 비디오 캡처 자원 해제

    # 넘파이 배열로 생성
    data = np.array(data)
    data_visibility = np.array(data_visibility)
    print("data shape: ", data.shape)
    print("visibility data shape: ", data_visibility.shape)

    # 넘파이 데이터 저장
    np.save(os.path.join(save_path, f'flip_{base_name}'), data)
    np.save(os.path.join(save_path_visibility, f'flip_{base_name}'), data)

cv2.destroyAllWindows()