import numpy as np
import cv2
from setting import mp_hands, mp_pose, hands, pose, pose_landmark_indices, mp_drawing

# Input: 동영상 파일 / Output: 좌표값, 가시성정보 numpy 배열
def get_landmarks(frame, angle):

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
                    joint_left_hands[j] = [lm.x, lm.y, lm.z, set_visibility(lm.x, lm.y)]
                else:
                    joint_right_hands[j] = [lm.x, lm.y, lm.z, set_visibility(lm.x, lm.y)]
            
            # 손 랜드마크 그리기
            if handedness.classification[0].label == 'Left':
                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)))   # green
            else:
                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0)))   # blue

    # 포즈 검출시
    if results_pose.pose_landmarks is not None:
        # 전체 데이터(joint) 생성 / 포즈 -> 지정한 관절에 대해서만 반복
        for j, i in enumerate(pose_landmark_indices):
            plm = results_pose.pose_landmarks.landmark[i]
            joint[j] = np.concatenate([joint_left_hands[j], joint_right_hands[j], [plm.x, plm.y, plm.z, plm.visibility]])
            joint_pose[j] = [plm.x, plm.y, plm.z, plm.visibility]

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if angle is False:
        # 좌표값 + 가시성 (총 데이터 4*21*3 = 252)
        return np.array(joint.flatten()), frame
    if angle is True:
        a = np.concatenate((angle_hands(joint_left_hands), angle_hands(joint_right_hands), angle_pose(joint_pose)))
        # 좌표값 + 가시성 + 각도 (총 데이터 4*21*3 + 45 = 297)
        return np.concatenate((joint.flatten(), a)), frame


def angle_hands(joint_hands):
    # 관절 간의 각도 계산
    v1 = joint_hands[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint  각 관절은 [x, y, z] 좌표로 표현되므로 :3
    v2 = joint_hands[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # [20, 3]. 20개 행과 3개 열

    # 정규화 전에 분모가 0인 경우에는 0으로 대체
    norm_v = np.linalg.norm(v, axis=1)
    v[norm_v == 0] = 0  # 분모가 0인 경우에는 해당 요소를 0으로 대체

    # 벡터 정규화
    v = v / norm_v[:, np.newaxis]

    # 각도 계산 (arccos를 이용하여 도트 곱의 역순 취함)
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

    # 라벨에 각도 정보 추가
    angle_label = np.array([angle], dtype=np.float32)
    
    return angle_label.flatten()


def angle_pose(joint_pose):
    v1 = joint_pose[[0, 2, 0, 1, 0, 0, 7, 8, 8, 8, 10, 12, 12, 12, 7, 7, 9, 11, 11, 11], :3]
    v2 = joint_pose[[2, 4, 1, 3, 5, 6, 8, 7, 10, 20, 12, 14, 16, 18, 9, 19, 11, 13, 15, 17], :3]
    # 0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    # 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    v = v2 - v1

    # 정규화 전에 분모가 0인 경우에는 0으로 대체
    norm_v = np.linalg.norm(v, axis=1)
    v[norm_v == 0] = 0  # 분모가 0인 경우에는 해당 요소를 0으로 대체

    v = v / norm_v[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],:], 
        v[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],:])) # [15,]

    angle_label = np.array([angle], dtype=np.float32)

    return angle_label.flatten()


def set_visibility(x, y, epsilon=1e-6):
    
    # 모든 좌표가 0인 경우
    if x <= epsilon and y <= epsilon:
        return 0
    
    # 한 좌표가 0
    elif x > epsilon and y <= epsilon:
        return 0.5
    elif x <= epsilon and y > epsilon:
        return 0.5
    
    # 전부 0이 아닌 경우
    else:
        return 1