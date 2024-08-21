import os
import numpy as np
from setting import actions, seq_length, jumping_window
from keras.models import load_model
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 텍스트 파일 경로
file_path = os.path.join(base_dir, 'data_android.txt')  # 파일이 위치한 경로

# 모델 불러오기
model = load_model('models/model.keras')

# 파일을 읽어서 NumPy 배열로 변환
data = []
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            # 쉼표로 구분된 float 값을 가져와서 리스트로 변환
            float_array = list(map(float, line.split(',')))
            data.append(float_array)

# NumPy 배열로 변환
data = np.array(data)
print(data.shape)

np.save(os.path.join(base_dir, 'data_android.npy'), data)

if len(data) > seq_length:
    full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, jumping_window)]
    full_seq_data = np.array(full_seq_data)
    print(full_seq_data.shape)

    # 예측
    y_pred = model.predict(full_seq_data)

    # 각 프레임의 가장 높은 확률을 가지는 클래스와 해당 확률 선택
    predicted_classes = []
    for pred in y_pred:
        predicted_class = np.argmax(pred)
        predicted_classes.append(predicted_class)

    print(predicted_classes)

    # 다수결 투표 방식으로 최종 예측 결정
    if predicted_classes:  # predicted_classes가 비어있지 않은 경우에만 처리
        vote_counts = Counter(predicted_classes)
        final_prediction, final_prediction_count = vote_counts.most_common(1)[0]
        action = actions[final_prediction]

        # 정답 출력/개수 계산
        print("예측결과: ", action)