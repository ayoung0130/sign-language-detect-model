import cv2
import numpy as np
import os, time
from landmark_processing import get_landmarks
from dotenv import load_dotenv

# 테스트 영상을 넘파이 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

folder_path = os.path.join(base_dir, f"test_video/3_test_10_words")

# 데이터 저장 경로
save_path = os.path.join(base_dir, "test")
flip_save_path = os.path.join(base_dir, "test_flip")

video_num = 0

for video_file in os.listdir(folder_path):
    video_num += 1
    # 동영상 불러오기
    video_path = os.path.join(folder_path, video_file)
    base_name = os.path.splitext(os.path.basename(video_path))[0]  # 확장자를 제거한 이름
    cap = cv2.VideoCapture(video_path)

    data = []
    flip_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 랜드마크, 프레임 가져오기
        d, original_frame = get_landmarks(frame, True)
    
        # 전체 데이터 배열에 추가
        data.append(d)

        if "오른쪽" not in base_name and "왼쪽" not in base_name:
            # 좌우반전된 프레임 처리
            flipped_frame = cv2.flip(frame, 1)
            d_flipped, flipped_frame = get_landmarks(flipped_frame, True)
            flip_data.append(d_flipped)

            # 화면에 표시
            cv2.imshow('Flipped', flipped_frame)

        # 화면에 표시
        cv2.imshow('Original', original_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # 넘파이 배열로 생성
    data = np.array(data)
    flip_data = np.array(flip_data)
    print("data shape: ", data.shape)

    # 넘파이 데이터 저장
    np.save(os.path.join(save_path, f'{base_name}'), data)
    if flip_data.size > 0:  # flip_data가 비어있지 않은 경우에만 저장
        np.save(os.path.join(flip_save_path, f'flip_{base_name}'), flip_data)
    
print("테스트 영상 넘파이 변환 완료")
print("영상 개수: ", video_num)
# 사용된 함수, 자원 해제
cv2.destroyAllWindows()