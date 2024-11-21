import os, random
import numpy as np
from setting import actions, seq_length, jumping_window
from keras.models import load_model
from collections import Counter
from dotenv import load_dotenv

# numpy 파일로 모델 예측을 수행하는 코드 

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 모델 불러오기
model = load_model('models/model.keras')

# 폴더 목록
folders = ["0_9", "10_19", "20_29", "30_39"]  # "0_9", "10_19", "20_29", "30_39", "40_52"

correct_count = 0
flip_correct_count = 0
npy_file_count = 0
flip_npy_file_count = 0

# 각 action별 정답 수를 저장할 딕셔너리 초기화
action_correct_counts = {action: 0 for action in actions}

# 폴더 내의 모든 파일에 대해 예측 수행
for folder in folders:
    folder_path = os.path.join(base_dir, f"test_npy/{folder}")
    npy_files = os.listdir(folder_path)
    random.shuffle(npy_files)
    npy_file_count += len(npy_files)
    flip_npy_file_count += len([file for file in npy_files if "flip" in file])

    for npy_file in npy_files:
        # 파일 불러오기
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)

        # 넘파이 데이터 로드
        data = np.load(file_path)

        # 시퀀스 길이보다 데이터 길이가 작은 경우 패딩 적용
        if len(data) < seq_length:
            padding_length = seq_length - len(data)
            # 시퀀스의 부족한 부분을 0으로 채움
            data = np.pad(data, ((0, padding_length), (0, 0)), mode='constant')

        # 분할하여 시퀀스 데이터로 변환
        full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, 5)]
        full_seq_data = np.array(full_seq_data)
        print(full_seq_data.shape)

        # 예측 수행
        y_pred = model.predict(full_seq_data)

        # 각 시퀀스의 가장 높은 확률을 가지는 클래스와 해당 확률 선택
        predicted_classes = [np.argmax(pred) for pred in y_pred]

        print(predicted_classes)

        # 다수결 투표 방식으로 최종 예측 결정
        if predicted_classes:  # predicted_classes가 비어있지 않은 경우에만 처리
            vote_counts = Counter(predicted_classes)
            final_prediction, final_prediction_count = vote_counts.most_common(1)[0]
            action = actions[final_prediction]

            # 정답 출력/개수 계산
            print("예측결과: ", action)
            print("정답: ", base_name)
            if action in base_name:
                correct_count += 1
                action_correct_counts[action] += 1
                if "flip" in base_name:
                    flip_correct_count += 1

        else:
            print("신뢰도가 낮습니다.")
            print("정답: ", base_name)

print("")
print("결과")

# 각 action별 정답 확률 출력
for action, correct in action_correct_counts.items():
    word_count = 6
    accuracy = (correct / word_count) * 100
    print(f"{action} --> {accuracy:.2f}% ({correct} / {word_count})")

# 총 정답 개수
print("\n총 정답 수:", correct_count, "/", npy_file_count)

# flip 영상이 있다면
if flip_npy_file_count > 0:
    print("좌우반전 정답 수:", flip_correct_count, "/", flip_npy_file_count)

print(f"정답 확률: {(correct_count / npy_file_count * 100):.2f}%")