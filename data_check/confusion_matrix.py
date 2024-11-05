import os, random
import numpy as np
from setting import actions, seq_length, jumping_window
from keras.models import load_model
from collections import Counter
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 모델 불러오기
model = load_model('models/model_88.33.keras')

# 폴더 목록
folders = ["0_9", "10_19", "20_29", "30_39"]  # "0_9", "10_19", "20_29", "30_39", "40_52"

y_true = []
y_pred = []

# 폴더 내의 모든 파일에 대해 예측 수행
for folder in folders:
    folder_path = os.path.join(base_dir, f"test_npy/{folder}")
    npy_files = os.listdir(folder_path)

    for npy_file in npy_files:
        # 동영상 불러오기
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        
        # 파일명에서 실제 클래스 이름 추출
        actual_action = None
        base_name_cleaned = os.path.splitext(base_name)[0]  # 파일명에서 확장자(.npy) 제거
        base_name_parts = base_name_cleaned.split('_')
        actual_action_candidate = base_name_parts[-1]  # 파일명에서 마지막 부분을 실제 클래스 이름으로 사용

        for action in actions:
            if action == actual_action_candidate:
                actual_action = action
        
        data = np.load(file_path)

        # 시퀀스 길이보다 데이터 길이가 작은 경우 패딩 적용
        if len(data) < seq_length:
            padding_length = seq_length - len(data)
            # 시퀀스의 부족한 부분을 0으로 채움
            data = np.pad(data, ((0, padding_length), (0, 0)), mode='constant')

        full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, jumping_window)]
        full_seq_data = np.array(full_seq_data)

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
cm = confusion_matrix(y_true, y_pred)

# Confusion Matrix를 확률로 변환 (전체 예측 수로 나누기)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 각 행(실제 클래스)별로 비율을 계산

# Confusion Matrix 시각화 (확률로 표현)
plt.figure(figsize=(20, 18))

# 0.0인 셀은 공백으로 표시하고, 나머지 셀은 소수점 둘째 자리까지만 표시하도록 수정
annot = np.where(cm_normalized != 0, np.round(cm_normalized, 2), '')

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