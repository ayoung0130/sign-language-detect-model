import cv2, os, random
import mediapipe as mp
import numpy as np
from setting import setVisibility, actions, seq_length, font
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from init_mediapipe import mp_hands, mp_pose, hands, pose, pose_landmark_indices, mp_drawing

# 모델 불러오기
model = load_model('models/model_xyz.h5')

# 웹캠 설정
cap = cv2.VideoCapture(0)

data = []
action = '수어 동작을 시작하세요'
frame_count = 1

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

    # 포즈 검출시
    if results_pose.pose_landmarks is not None:
        for j, i in enumerate(pose_landmark_indices):
            plm = results_pose.pose_landmarks.landmark[i]
            joint[j] = np.concatenate([joint_left_hands[j], joint_right_hands[j], [plm.x, plm.y, plm.z, plm.visibility]])
        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    d = np.array(joint.flatten())
    data.append(d)

    if frame_count > seq_length:
        data = np.array(data)

        full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
        full_seq_data = np.array(full_seq_data)

        # 예측
        y_pred = model.predict(full_seq_data)

        mean_pred = np.mean(np.array(y_pred), axis=0)
        # print("mean: ", mean_pred)

        max_pred = int(np.argmax(mean_pred))
        # print("max_pred_idx: ", max_pred)

        conf = mean_pred[max_pred]
        print(f"conf: {conf:.3f}")

        if conf > 0.5:
            action = actions[max_pred]
            print("예측결과: ", action)
        else:
            action = '수어 동작을 시작하세요'

        data = []
        frame_count = 1

    # 글자 표시
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((20,20), action, font=font, fill=(0,0,0))
    frame = np.array(img_pil)

    frame_count += 1

    # 화면에 표시
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# data = np.array(data)

# full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
# full_seq_data = np.array(full_seq_data)
# print("seq data shape:", full_seq_data.shape)

# # 예측
# y_pred = model.predict(full_seq_data)
# print(y_pred)

# mean_pred = np.mean(np.array(y_pred), axis=0)
# print("mean: ", mean_pred)

# max_pred = int(np.argmax(mean_pred))
# print("max_pred_idx: ", max_pred)

# conf = mean_pred[max_pred]
# print(f"conf: {conf:.3f}")

# action = actions[max_pred]
# print("예측결과: ", action)

cap.release()
cv2.destroyAllWindows()