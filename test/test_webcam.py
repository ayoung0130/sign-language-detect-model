import cv2
import numpy as np
from setting import actions, font
from keras.models import load_model
from PIL import ImageDraw, Image
from data_processing.landmark_processing import get_landmarks
from setting import hands
from collections import Counter

# 웹캠으로 모델 예측을 수행하는 코드

# 모델 불러오기
model = load_model('models/model_0616.h5')

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
    d, frame = get_landmarks(frame, True)

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

        full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, 10)]
        full_seq_data = np.array(full_seq_data)

        # 예측
        y_pred = model.predict(full_seq_data)

        # 각 프레임의 가장 높은 확률을 가지는 클래스 선택
        predicted_classes = np.argmax(y_pred, axis=1)
        print(predicted_classes)

        # 다수결 투표 방식으로 최종 예측 결정
        vote_counts = Counter(predicted_classes)
        final_prediction, final_prediction_count = vote_counts.most_common(1)[0]

        action = actions[final_prediction]

        print("예측결과: ", action)
            
        data = []

cap.release()
cv2.destroyAllWindows()