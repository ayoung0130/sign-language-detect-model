import os, random
import numpy as np
from setting import actions, seq_length
from keras.models import load_model
from collections import Counter
from dotenv import load_dotenv

# numpy 파일로 모델 예측을 수행하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 모델 불러오기
model = load_model('models/model.h5')

# 넘파이 파일 설정
npy_data = os.path.join(base_dir, 'test_npy/landmarks')

# 동영상 파일 목록 랜덤으로 섞기
npy_files = os.listdir(npy_data)
random.shuffle(npy_files)
npy_file_count = len(npy_files)
flip_npy_file_count = len([file for file in npy_files if "flip" in file])

correct_count = 0
flip_correct_count = 0

# 각 action별 정답 수를 저장할 딕셔너리 초기화
action_correct_counts = {action: 0 for action in actions}

for npy_file in npy_files:
    # 동영상 불러오기
    file_path = os.path.join(npy_data, npy_file)
    base_name = os.path.basename(file_path)

    data = np.load(file_path)

    full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, 10)]
    full_seq_data = np.array(full_seq_data)
    print(full_seq_data.shape)

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

    # 예측값을 넘파이 파일로 저장
    # save_path = os.path.join(base_dir, f"pred/{base_name}_{action}.npy")
    # np.save(save_path, y_pred)

print("")
print("결과")

# 각 action별 정답 확률 출력
for action, correct in action_correct_counts.items():
    word_count = 6
    if "오른쪽" in action or "왼쪽" in action:
        word_count = 3
    accuracy = (correct / word_count) * 100
    print(f"{action} --> {accuracy:.2f}% ({correct} / {word_count})")

# 총 정답 개수
print("\n총 정답 수:", correct_count, "/", npy_file_count)

# flip 영상이 있다면
if flip_npy_file_count > 0:
    print("좌우반전 정답 수:", flip_correct_count, "/", flip_npy_file_count)

print(f"정답 확률: {(correct_count / npy_file_count * 100):.2f}%")