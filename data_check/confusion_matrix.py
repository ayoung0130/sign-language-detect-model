import os, random
import numpy as np
from setting import actions, seq_length, jumping_window
from keras.models import load_model
from collections import Counter
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 모델 불러오기
model = load_model('models/model.h5')
 
# 넘파이 파일 설정
npy_data = os.path.join(base_dir, '20words')

# 동영상 파일 목록 불러오기
npy_files = os.listdir(npy_data)

y_true = []
y_pred = []

for npy_file in npy_files:
    # 동영상 불러오기
    file_path = os.path.join(npy_data, npy_file)
    base_name = os.path.basename(file_path)
    
    # 파일명에서 실제 클래스 이름 추출
    actual_action = None
    for action in actions:
        if action in base_name:
            actual_action = action
            break
    
    data = np.load(file_path)

    if len(data) > 30:
        full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, 10)]
        full_seq_data = np.array(full_seq_data)
        print(full_seq_data.shape)

        # 예측
        y_pred_raw = model.predict(full_seq_data)

        predicted_classes = []
        for pred in y_pred_raw:
            max_prob = np.max(pred)
            if max_prob >= 0.90:
                predicted_class = np.argmax(pred)
                predicted_classes.append(predicted_class)

        if predicted_classes:  # predicted_classes가 비어있지 않은 경우에만 처리
            vote_counts = Counter(predicted_classes)
            final_prediction, final_prediction_count = vote_counts.most_common(1)[0]

            # 실제 클래스와 예측 클래스 기록
            y_true.append(actions.index(actual_action))  # 실제 클래스는 파일명에서 추출한 동작 이름의 인덱스
            y_pred.append(final_prediction)  # 예측 클래스

# Confusion Matrix 생성
# 신뢰도가 낮아도 표시되도록
cm = confusion_matrix(y_true, y_pred)

# Confusion Matrix 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(len(actions)), yticklabels=range(len(actions)))
plt.xlabel('Predicted Label (Index)')
plt.ylabel('True Label (Index)')
plt.title('Confusion Matrix')
plt.show()

# 인덱스와 동작 이름 매핑 출력
print("Index to Action Mapping:")
for idx, action in enumerate(actions):
    print(f"{idx}: {action}")