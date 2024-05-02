import cv2
import mediapipe as mp
import numpy as np
from setting import setVisibility
from keras.models import load_model
from collections import Counter

actions = ['가렵다', '기절', '부러지다', '어제', '어지러움']
seq_length = 30

# 모델 불러오기
model = load_model('models/model.h5')

# 미디어 파이프 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_landmark_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

mp_drawing = mp.solutions.drawing_utils

# 웹캠 또는 비디오 파일 설정
video_source = 0  # 웹캠을 사용하려면 0 또는 웹캠 장치 번호를 사용
cap = cv2.VideoCapture(video_source)

# 전체 데이터 저장할 배열 초기화
data = []

frame_len = 1
prev_action = None
consecutive_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame)    # 손 랜드마크 검출
    results_pose = pose.process(frame)      # 포즈 랜드마크 검출
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 관절 정보 저장할 넘파이 배열 초기화
    joint_left_hands = np.zeros((21, 4))
    joint_right_hands = np.zeros((21, 4))
    joint_pose = np.zeros((21, 4))
    joint = np.zeros((21, 12))

    # 손 검출시
    if results_hands.multi_hand_landmarks is not None:
        for res, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                
            # 손 -> 모든 관절에 대해 반복. 한 프레임에 왼손, 오른손 데이터가 0번부터 20번까지 들어감
            for j, lm in enumerate(res.landmark):
                if handedness.classification[0].label == 'Left':
                    joint_left_hands[j] = [lm.x, lm.y, lm.z, setVisibility(lm.x, lm.y)]
                else:
                    joint_right_hands[j] = [lm.x, lm.y, lm.z, setVisibility(lm.x, lm.y)]
                
            # 손 랜드마크 그리기
            if handedness.classification[0].label == 'Left':
                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)))   # green
            else:
                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0)))   # blue
    
    if results_pose.pose_landmarks is not None:
        # 전체 데이터(joint) 생성, 포즈 -> 지정한 관절에 대해서만 반복
        for j, i in enumerate(pose_landmark_indices):
            plm = results_pose.pose_landmarks.landmark[i]
            joint[j] = np.concatenate([joint_left_hands[j], joint_right_hands[j], [plm.x, plm.y, plm.z, plm.visibility]])

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    data.append(joint.flatten())

    # 화면에 표시
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
    if frame_len == 2 * seq_length:
        np_data = np.array(data)
        print(np_data.shape)

        # 시퀀스 데이터
        seq_data = np.array([np_data[seq:seq + seq_length] for seq in range(len(np_data) - seq_length)])

        # 모델로 예측 수행
        prediction = model.predict(seq_data)

        # 시퀀스별 예측값의 평균 계산
        avg_prediction = np.mean(prediction, axis=0)

        # 최종 예측값 출력
        pred = np.argmax(avg_prediction)
        print(actions[pred])
        action = actions[pred]
        cv2.putText(frame, f'{this_action.upper()}', org=(int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        data = []
        frame_len = 0

    frame_len += 1

# # 이전 동작과 현재 동작이 같은지 확인
# if action == prev_action:
#     consecutive_count += 1
# else:
#     consecutive_count = 1

# # 연속으로 같은 동작이 5프레임 이상 나타나면 출력
# if consecutive_count >= 5:
#     cv2.putText(frame, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# # 이전 동작 업데이트
# prev_action = action

cap.release()
cv2.destroyAllWindows()