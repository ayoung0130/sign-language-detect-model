import cv2, os, random
import numpy as np
from setting import actions, seq_length, jumping_window
from keras.models import load_model
from data_processing.landmark_processing import get_landmarks
from collections import Counter
from dotenv import load_dotenv

# 촬영한 비디오로 모델 예측을 수행하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 모델 불러오기
model = load_model('models/model.h5')

# 비디오 파일 설정
video_source = os.path.join(base_dir, 'test_video_10_words')

# 동영상 파일 목록 랜덤으로 섞기
video_files = os.listdir(video_source)
random.shuffle(video_files)
video_file_count = len(video_files)
flip_video_file_count = len([file for file in video_files if "3" in file])

correct_count = 0
flip_correct_count = 0

# 각 action별 정답 수를 저장할 딕셔너리 초기화
action_correct_counts = {action: 0 for action in actions}

for video_file in video_files:
    # 동영상 불러오기
    video_path = os.path.join(video_source, video_file)
    base_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)

    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 랜드마크, 프레임 가져오기
        d, frame = get_landmarks(frame)

        if d is not None:
            # 전체 데이터 배열에 추가
            data.append(d)

        # 화면에 표시
        cv2.imshow('MediaPipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    data = np.array(data)

    full_seq_data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, jumping_window)]
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

    # # 예측값을 넘파이 파일로 저장
    # save_path = os.path.join(base_dir, f"pred/{base_name}_{action}.npy")
    # np.save(save_path, y_pred)

cv2.destroyAllWindows()

print("")
print("결과")

# 각 action별 정답 확률 출력
for action, correct in action_correct_counts.items():
    accuracy = (correct / 3 * 100)
    print(f"{action} --> {accuracy:.2f}% ({correct}/3)")

# 총 정답 개수
print("\n총 정답 수:", correct_count, "/", video_file_count)

# flip 영상이 있다면
if flip_video_file_count > 0:
    print("좌우반전 정답 수:", flip_correct_count, "/", flip_video_file_count)

print(f"정답 확률: {(correct_count / video_file_count * 100):.2f}%")