import numpy as np
import cv2
from setting import mp_hands, mp_pose, hands, pose, pose_landmark_indices, mp_drawing

# 미디어파이프 hands, pose 모델의 랜드마크를 처리하는 코드
# 랜드마크 추출, 각도값 계산
# Input: 동영상 파일 / Output: 좌표값, 가시성정보 numpy 배열

def get_landmarks(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame)    # 손 랜드마크 검출
    results_pose = pose.process(frame)      # 포즈 랜드마크 검출
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 관절 정보 저장할 넘파이 배열 초기화
    joint_left_hands = np.zeros((21, 3))
    joint_right_hands = np.zeros((21, 3))
    joint_pose = np.zeros((21, 3))

    # 손 검출시
    if results_hands.multi_hand_landmarks is not None:
        for res, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            # 손 -> 모든 관절에 대해 반복. 한 프레임에 왼손, 오른손 데이터가 0번부터 20번까지 들어감
            for j, lm in enumerate(res.landmark):
                if handedness.classification[0].label == 'Left':
                    joint_left_hands[j] = [lm.x, lm.y, lm.z]
                else:
                    joint_right_hands[j] = [lm.x, lm.y, lm.z]
            
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)) if handedness.classification[0].label == 'Left' else mp_drawing.DrawingSpec(color=(255, 0, 0)))

        # 포즈 검출시
        if results_pose.pose_landmarks is not None:
            # 포즈 -> 지정한 관절에 대해서만 반복
            for j, i in enumerate(pose_landmark_indices):
                plm = results_pose.pose_landmarks.landmark[i]
                joint_pose[j] = [plm.x, plm.y, plm.z]
            
            # 포즈 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 왼손 + 오른손 + 포즈
        joint = np.concatenate([joint_left_hands, joint_right_hands, joint_pose], axis=0)   # joint -> (63, 3)

        # joint (총 데이터 3*21*3 = 189)
        return joint.flatten(), frame
    
    # 손이 검출되지 않으면 None 반환
    return None, frame


def get_landmarks_visibility(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame)    # 손 랜드마크 검출
    results_pose = pose.process(frame)      # 포즈 랜드마크 검출
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 관절 정보 저장할 넘파이 배열 초기화
    joint_left_hands = np.zeros((21, 4))
    joint_right_hands = np.zeros((21, 4))
    joint_pose = np.zeros((21, 4))

    # 손 검출시
    if results_hands.multi_hand_landmarks is not None:
        for res, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            # 손 -> 모든 관절에 대해 반복. 한 프레임에 왼손, 오른손 데이터가 0번부터 20번까지 들어감
            for j, lm in enumerate(res.landmark):
                if handedness.classification[0].label == 'Left':
                    joint_left_hands[j] = [lm.x, lm.y, lm.z, set_visibility(lm.x, lm.y)]
                else:
                    joint_right_hands[j] = [lm.x, lm.y, lm.z, set_visibility(lm.x, lm.y)]
            
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)) if handedness.classification[0].label == 'Left' else mp_drawing.DrawingSpec(color=(255, 0, 0)))

        # 포즈 검출시
        if results_pose.pose_landmarks is not None:
            # 포즈 -> 지정한 관절에 대해서만 반복
            for j, i in enumerate(pose_landmark_indices):
                plm = results_pose.pose_landmarks.landmark[i]
                joint_pose[j] = [plm.x, plm.y, plm.z, plm.visibility]

            # 포즈 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 왼손 + 오른손 + 포즈
        joint = np.concatenate([joint_left_hands, joint_right_hands, joint_pose], axis=0)      # joint -> (63, 4)

        # joint (총 데이터 4*21*3 = 252)
        return joint.flatten(), frame
    
    # 손이 검출되지 않으면 None 반환
    return None, frame


def set_visibility(x, y, epsilon=1e-6):
    
    # 모든 좌표가 0인 경우
    if x <= epsilon and y <= epsilon:
        return 0
    
    # 한 좌표가 0인 경우
    if x <= epsilon or y <= epsilon:
        return 0.5
    
    # 전부 0이 아닌 경우
    return 1