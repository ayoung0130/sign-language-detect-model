import cv2, os, random
import numpy as np
from setting import actions, font
from keras.models import load_model
from PIL import ImageDraw, Image
from landmark_processing import get_landmarks

# 모델 불러오기
model = load_model('models/model.h5')

# 비디오 파일 설정
video_source = f"C:/Users/mshof/Desktop/video/test_video"

# 동영상 파일 목록 랜덤으로 섞기
video_files = os.listdir(video_source)
random.shuffle(video_files)

seq_length = 30

action = '수어 동작을 시작하세요'

for video_file in video_files:
    # 동영상 불러오기
    video_path = os.path.join(video_source, video_file)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 랜드마크, 프레임 가져오기
        d, frame = get_landmarks(frame)
        
        # 전체 데이터 배열에 추가
        data.append(d)

        # 프레임 길이가 시퀀스 길이보다 클 경우 예측 수행
        if frame_count > seq_length:
            data = np.array(data)

            full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
            full_seq_data = np.array(full_seq_data)

            # 예측
            y_pred = model.predict(full_seq_data)

            mean_pred = np.mean(np.array(y_pred), axis=0)
            # print("mean: ", mean_pred)

            max_pred = int(np.argmax(mean_pred))
            # print("max_pred_idx: ", max_pred)

            conf = mean_pred[max_pred]
            print(f"conf: {conf:.3f}")

            if conf > 0.5:
                action = actions[max_pred]
                print("예측결과: ", action)
            else:
                action = '수어 동작을 시작하세요'

            data = []
            frame_count = 1

        # 글자 표시
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((20,20), action, font=font, fill=(0,0,0))
        frame = np.array(img_pil)

        # 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()