import cv2
import numpy as np
from setting import actions, font
from keras.models import load_model
from PIL import ImageDraw, Image
from landmark_processing import get_landmarks
from setting import hands

# 모델 불러오기
model = load_model('models/model_slice.h5')

# 웹캠 설정
cap = cv2.VideoCapture(0)

action = "수어 동작을 시작하세요"
frame_count = 0

data = []
seq_length = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results_hands = hands.process(frame)    # 손 랜드마크 검출

    # 랜드마크, 프레임 가져오기
    d, frame = get_landmarks(frame)

    # 글자 표시
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((20,20), action, font=font, fill=(0,0,0))
    frame = np.array(img_pil)

    # 화면에 표시
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    if results_hands.multi_hand_landmarks is not None:
        # 전체 데이터 배열에 추가
        data.append(d)

    elif results_hands.multi_hand_landmarks is None and len(data) > seq_length:
        data = np.array(data)

        full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
        full_seq_data = np.array(full_seq_data)

        # 예측
        y_pred = model.predict(full_seq_data)

        mean_pred = np.mean(np.array(y_pred), axis=0)

        max_pred = int(np.argmax(mean_pred))

        conf = mean_pred[max_pred]
        print(f"conf: {conf:.3f}")

        if conf > 0.5:
            action = actions[max_pred]
            print("예측결과: ", action)
        else:
            action = "정확도가 낮습니다. 동작을 다시 시작하세요"
            print("예측결과: ", actions[max_pred])
            print("정확도가 낮습니다")
            
        data = []

cap.release()
cv2.destroyAllWindows()