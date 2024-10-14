import cv2
import numpy as np
import threading
from setting import actions, seq_length, jumping_window, font
from keras.models import load_model
from PIL import ImageDraw, Image
from data_processing.landmark_processing import get_landmarks
from llm_tts import words_to_sentence, tts

# 웹캠으로 모델 예측을 수행하는 코드 (문장 단위)

# 모델 불러오기
model = load_model('models/model_40words.keras')

# 웹캠 설정
cap = cv2.VideoCapture(0)

action = "수어 동작을 시작하세요"
data = []

# 단어 예측을 위한 리스트 초기화
predicted_classes = []
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

    # 손 검출시
    if d is not None:
        data.append(d)

    # 손이 화면에서 벗어나고 데이터 길이가 시퀀스 길이보다 길다면
    elif len(data) > seq_length:
        data = np.array(data)

        full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, 5)]
        full_seq_data = np.array(full_seq_data)

        # 예측
        y_pred = model.predict(full_seq_data)

        # 각 시퀀스의 가장 높은 확률을 가지는 클래스와 해당 확률 선택
        for pred in y_pred:
            predicted_class = np.argmax(pred)
            predicted_classes.append(predicted_class)

        # 선택된 레이블을 단어로 변환
        for label in predicted_classes:
            predicted_words.append(actions[label])

        print(predicted_words)

        # 문장 변환, 화면에 출력
        sentence = words_to_sentence(predicted_words)
        action = sentence

        # TTS 비동기 처리 시작 (별도의 스레드에서 실행)
        tts_thread = threading.Thread(target=tts, args=(sentence,))
        tts_thread.start()

        # 데이터 초기화
        data = []
        predicted_words = []
        predicted_classes = []

    # 손이 화면에서 벗어났지만 데이터 길이가 시퀀스 길이보다 짧다면
    elif len(data) > 0:
        action = "동작을 더 길게 수행해주세요"
        data = []
        
    # 화면에 표시
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()