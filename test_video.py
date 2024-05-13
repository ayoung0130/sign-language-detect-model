import cv2, os, random
import numpy as np
from setting import actions
from keras.models import load_model
from landmark_processing import get_landmarks

# 모델 불러오기
model = load_model('models/model_slice.h5')

# 비디오 파일 설정
video_source = f"C:/Users/mshof/Desktop/video/test_video"

# 동영상 파일 목록 랜덤으로 섞기
video_files = os.listdir(video_source)
random.shuffle(video_files)
video_file_count = len(video_files)

seq_length = 30
correct_count = 0

for video_file in video_files:
    # 동영상 불러오기
    video_path = os.path.join(video_source, video_file)
    base_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)

    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 랜드마크, 프레임 가져오기
        d, frame = get_landmarks(frame, False)
        
        # 전체 데이터 배열에 추가
        data.append(d)

        # 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
    data = np.array(data)

    full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
    full_seq_data = np.array(full_seq_data)

    # 예측
    y_pred = model.predict(full_seq_data)
    print(y_pred)

    mean_pred = np.mean(np.array(y_pred), axis=0)
    print(mean_pred)

    max_pred = int(np.argmax(mean_pred))
    print(max_pred)

    conf = mean_pred[max_pred]

    action = actions[max_pred]

    print("예측결과: ", action)
    print(f"conf: {conf:.3f}")
    print("정답: ", base_name)

    if action in base_name:
        correct_count += 1

cap.release()
cv2.destroyAllWindows()

print("\n정답 수:", correct_count)
print(f"정답 확률: {(correct_count / video_file_count * 100):.2f}%")