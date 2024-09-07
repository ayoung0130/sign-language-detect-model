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
model = load_model('models/model.keras')
 
# 넘파이 파일 설정
npy_data = os.path.join(base_dir, 'test_30words')

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

    # 시퀀스 길이보다 데이터 길이가 작은 경우 패딩 적용
    if len(data) < seq_length:
        padding_length = seq_length - len(data)
        # 시퀀스의 부족한 부분을 0으로 채움
        data = np.pad(data, ((0, padding_length), (0, 0)), mode='constant')

    full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, jumping_window)]
    full_seq_data = np.array(full_seq_data)
    print(full_seq_data.shape)

    # 예측
    y_pred_raw = model.predict(full_seq_data)

    predicted_classes = []
    for pred in y_pred_raw:
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

# Confusion Matrix를 확률로 변환 (전체 예측 수로 나누기)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 각 행(실제 클래스)별로 비율을 계산

# 0%를 표시하지 않고 소수점 없이 처리하는 함수
def fmt_percent(value):
    if value == 0:
        return ""
    else:
        return f"{int(value * 100)}%"  # 소수점 없이 퍼센트로 변환

# Confusion Matrix 시각화 (확률로 표현)
plt.figure(figsize=(20, 18))

# annot 인수를 통해 직접 0% 제외 및 소수점 없는 퍼센트 표기 처리
annot = np.array([[fmt_percent(val) for val in row] for row in cm_normalized])

# heatmap 그리기
ax = sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues', cbar=True, xticklabels=range(len(actions)), yticklabels=range(len(actions)))

# 축과 제목 설정
plt.xlabel('Predicted Label (Index)')
plt.ylabel('True Label (Index)')
plt.title('Confusion Matrix (Probability %)')
# 컬러바 추가
cbar = ax.collections[0].colorbar
cbar.set_label('Probability (%)')
plt.show()

# 인덱스와 동작 이름 매핑 출력
print("Index to Action Mapping:")
for idx, action in enumerate(actions):
    print(f"{idx}: {action}")