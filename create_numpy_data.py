import cv2
import numpy as np
import os, time
from landmark_processing import get_landmarks

# 동영상 파일 설정
# 인덱스 0(가렵다), 1(기절), 2(부러지다), 3(어제), 4(어지러움), 5(열나다), 6(오늘), 7(진통제), 8(창백하다), 9(토하다)
action = "토하다"
idx = 9
folder_path = f"C:/Users/mshof/Desktop/video/resized_video_{idx}"

# 데이터 저장 경로
npy_save_path = "C:/Users/mshof/Desktop/npy_data/"

data = []

for video_file in os.listdir(folder_path):
    # 동영상 불러오기
    video_path = os.path.join(folder_path, video_file)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 랜드마크, 프레임 가져오기
        d, frame = get_landmarks(frame)

        # 인덱스 추가
        d = np.append(d, idx)
    
        # 전체 데이터 배열에 추가
        data.append(d)

        # 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break

# 넘파이 배열로 생성
data = np.array(data)
print("data shape: ", action, data.shape)
print("data\n", data[1000:1002])

created_time = int(time.time())

# 넘파이 데이터 저장
np.save(os.path.join(npy_save_path, f'{action}_{created_time}'), data)
print("npy data shape:", action, data.shape)

# 사용된 함수, 자원 해제
cv2.destroyAllWindows()