import cv2
import numpy as np
from setting import actions, seq_length, jumping_window, font
from keras.models import load_model
from PIL import ImageDraw, Image
from data_processing.landmark_processing import get_landmarks
from collections import Counter
from llm_tts import words_to_sentence, tts

# 웹캠으로 모델 예측을 수행하는 코드 (문장 단위)

# 모델 불러오기
model = load_model('models/model.h5')

# 웹캠 설정
cap = cv2.VideoCapture(0)

action = "수어 동작을 시작하세요"
data = []

# 단어 예측을 위한 리스트 초기화
predicted_words = []

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

        # 예측된 레이블 카운트
        label_counts = Counter(predicted_classes)

        # 다섯 번 이상 나온 레이블만 선택
        filtered_labels = [label for label, count in label_counts.items() if count >= 5]

        # 선택된 레이블을 단어로 변환
        for label in filtered_labels:
            predicted_words.append(actions[label])

        tts(words_to_sentence(predicted_words))

        # 데이터 초기화
        data = []
        
    # 화면에 표시
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()