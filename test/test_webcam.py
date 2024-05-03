import cv2
import numpy as np
from setting import actions, seq_length, font
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from get_landmarks import getLandmarks

# 모델 불러오기
model = load_model('models/model_xyz.h5')

# 웹캠 설정
cap = cv2.VideoCapture(0)

data = []
action = '수어 동작을 시작하세요'
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 랜드마크, 프레임 가져오기
    d, frame = getLandmarks(frame)
    
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