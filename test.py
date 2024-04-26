import cv2
import mediapipe as mp
import numpy as np
from setting import setVisibility
from keras.models import load_model

actions = ['가렵다', '기절', '부러지다']
seq_length = 30

# 모델 불러오기
model = load_model('LSTM-Practice/models/model.h5')

# 미디어 파이프 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_landmark_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

mp_drawing = mp.solutions.drawing_utils

# 웹캠 또는 비디오 파일 설정
video_source = "resized_videos_2_20/9_부러지다(정).avi"  # 웹캠을 사용하려면 0 또는 웹캠 장치 번호를 사용
cap = cv2.VideoCapture(video_source)

# 전체 데이터 저장할 배열 초기화
data = []

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
                    joint_left_hands[j] = [lm.x, lm.y, lm.z, setVisibility(lm.x, lm.y, lm.z)]
                else:
                    joint_right_hands[j] = [lm.x, lm.y, lm.z, setVisibility(lm.x, lm.y, lm.z)]
                
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
            joint_pose[j] = [plm.x, plm.y, plm.z, plm.visibility]

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    data.append(joint.flatten())

    # 화면에 표시
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) == ord('q'):
        break

data = np.array(data)

# 시퀀스 데이터 생성
full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
full_seq_data = np.array(full_seq_data)
print(full_seq_data.shape)

# 모델로 예측 수행
prediction = model.predict(full_seq_data)
i_pred = int(np.argmax(prediction))
print(i_pred)
action = actions[i_pred]
print(action)

cap.release()
cv2.destroyAllWindows()