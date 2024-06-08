import cv2
import numpy as np
import os, time
from setting import actions
from landmark_processing import get_landmarks

# 영상을 넘파이 배열로 변환하는 코드

# 동영상 파일 설정
# 인덱스 0(가렵다), 1(기절), 2(부러지다), 3(어제), 4(어지러움), 5(열나다), 6(오늘), 7(진통제), 8(창백하다), 9(토하다)
idx = 9
action = actions[idx]
folder_path = f"C:/Users/mshof/Desktop/video/resized_video_{idx}"

# 데이터 저장 경로
train_save_path = "C:/Users/mshof/Desktop/npy_angle_train"
val_save_path = "C:/Users/mshof/Desktop/npy_angle_val"
test_save_path = "C:/Users/mshof/Desktop/npy_angle_test"

data = []
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
        d, frame = get_landmarks(frame, True)

        # 인덱스 추가
        d = np.append(d, idx)
    
        # 전체 데이터 배열에 추가
        data.append(d)

        # 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    if video_num == 12:
        # 넘파이 배열로 생성(train)
        data = np.array(data)
        print("train data shape: ", action, data.shape)

        # 넘파이 데이터 저장(train)
        np.save(os.path.join(train_save_path, f'{action}_train'), data)

        data = []

    if video_num == 16:
        # 넘파이 배열로 생성(validation)
        data = np.array(data)
        print("val data shape: ", action, data.shape)

        # 넘파이 데이터 저장(validation)
        np.save(os.path.join(val_save_path, f'{action}_val'), data)

        data = []

# 넘파이 배열로 생성(test)
data = np.array(data)
print("test data shape: ", action, data.shape)

# 넘파이 데이터 저장(test)
np.save(os.path.join(test_save_path, f'{action}_test'), data)

# 사용된 함수, 자원 해제
cv2.destroyAllWindows()