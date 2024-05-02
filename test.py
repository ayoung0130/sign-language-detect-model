import cv2
import mediapipe as mp
import numpy as np
from setting import setVisibility
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# 인덱스 0(가렵다), 1(기절), 2(부러지다), 3(어제), 4(어지러움), 5(열나다), 6(오늘), 7(진통제), 8(창백하다), 9(토하다)
actions = ['가렵다', '기절', '부러지다', '어제', '어지러움', '열나다', '오늘', '진통제', '창백하다', '토하다']
seq_length = 30
font = ImageFont.truetype('fonts/MaruBuri-Bold.ttf', 30)

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
video_source = 0  # 웹캠 사용시 0, 비디오 파일 사용시 경로명 입력
cap = cv2.VideoCapture(video_source)

seq = []
action_seq = []

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
        # 전체 데이터(joint) 생성, 포즈 -> 지정한 관절에 대해서만 반복
        for j, i in enumerate(pose_landmark_indices):
            plm = results_pose.pose_landmarks.landmark[i]
            joint[j] = np.concatenate([joint_left_hands[j], joint_right_hands[j], [plm.x, plm.y, plm.z, plm.visibility]])

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    d = np.array(joint.flatten())

    seq.append(d)

    if len(seq) < seq_length:
        continue

    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
    
    y_pred = model.predict(input_data).squeeze()

    i_pred = int(np.argmax(y_pred))
    conf = y_pred[i_pred]

    if conf > 0.90 and results_hands.multi_hand_landmarks is not None:
        action = actions[i_pred]
        print(action)
        action_seq.append(action)

        if len(action_seq) < 5 :
            continue

        if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5]:
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((40,40), action, font=font, fill=(255,255,255))
            frame = np.array(img_pil)
    else:
        cv2.putText(frame, '-', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # 화면에 표시
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()