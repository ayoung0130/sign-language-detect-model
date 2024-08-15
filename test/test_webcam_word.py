import cv2
import numpy as np
from setting import actions, seq_length, jumping_window, font
from keras.models import load_model
from PIL import ImageDraw, Image
from data_processing.landmark_processing import get_landmarks
from collections import Counter

# 웹캠으로 모델 예측을 수행하는 코드 (단어 단위)

# 모델 불러오기
model = load_model('models/model_10words_100.h5')

# 웹캠 설정
cap = cv2.VideoCapture(0)

action = "수어 동작을 시작하세요"
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 글자 표시
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((20,20), action, font=font, fill=(0,0,0))
    frame = np.array(img_pil)

    # 랜드마크, 프레임 가져오기
    d, frame = get_landmarks(frame)

    if d is not None:
        # 전체 데이터 배열에 추가
        data.append(d)
        print(d)

    elif len(data) > seq_length:
        data = np.array(data)

        full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, jumping_window)]
        full_seq_data = np.array(full_seq_data)

        # 예측
        y_pred = model.predict(full_seq_data)

        # 각 프레임의 가장 높은 확률을 가지는 클래스와 해당 확률 선택
        predicted_classes = []
        for pred in y_pred:
            max_prob = np.max(pred)
            if max_prob >= 0.90:
                predicted_class = np.argmax(pred)
                predicted_classes.append(predicted_class)

        print(predicted_classes)

        # 다수결 투표 방식으로 최종 예측 결정
        if predicted_classes:  # predicted_classes가 비어있지 않은 경우에만 처리
            vote_counts = Counter(predicted_classes)
            final_prediction, final_prediction_count = vote_counts.most_common(1)[0]
            action = actions[final_prediction]
            print("예측결과: ", action)

        else:
            action = "신뢰도가 낮습니다."

        data = []
    
    # 화면에 표시
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()