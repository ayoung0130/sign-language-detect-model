import numpy as np
import cv2
from setting import mp_hands, mp_pose, hands, pose, pose_landmark_indices, mp_drawing

# 미디어파이프 hands, pose 모델의 랜드마크를 추출 및 처리하는 코드
# Input: 동영상 파일 프레임 / Output: 랜드마크(x, y, z), 프레임

def get_landmarks(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame)    # 손 랜드마크 검출
    results_pose = pose.process(frame)      # 포즈 랜드마크 검출
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 손과 포즈 동시 검출시
    if results_hands.multi_hand_landmarks is not None and results_pose.pose_landmarks is not None:
        
        # 관절 정보 저장할 넘파이 배열 초기화
        joint_left_hands = np.zeros((21, 3))
        joint_right_hands = np.zeros((21, 3))
        joint_pose = np.zeros((15, 3))

        for res, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            # 손 -> 모든 관절에 대해 반복. 한 프레임에 왼손, 오른손 데이터가 0번부터 20번까지 들어감
            for j, lm in enumerate(res.landmark):
                if handedness.classification[0].label == 'Left':
                    joint_left_hands[j] = [lm.x, lm.y, lm.z]
                else:
                    joint_right_hands[j] = [lm.x, lm.y, lm.z]

            # 손 랜드마크 그리기
            color = (0, 255, 0) if handedness.classification[0].label == 'Left' else (255, 0, 0)
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=color))

        # 포즈 -> 지정한 관절에 대해서만 반복
        for j, i in enumerate(pose_landmark_indices):
            plm = results_pose.pose_landmarks.landmark[i]
            joint_pose[j] = [plm.x, plm.y, plm.z]

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        joint = np.concatenate([joint_left_hands.flatten(), joint_right_hands.flatten(), joint_pose.flatten()])

        return joint, frame
    
    return None, frame

def angle_hands(joint_hands):
    # 관절 간의 각도 계산
    v1 = joint_hands[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint  각 관절은 [x, y, z] 좌표로 표현되므로 :3
    v2 = joint_hands[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # (20, 3). 20개 행과 3개 열

    # 벡터 크기 계산
    norm_v = np.linalg.norm(v, axis=1)  # (20,) sqrt(x^2+y^2+z^2) 즉 벡터의 크기(길이)

    # 벡터 크기가 0인 값을 위해 마스크 생성
    zero_norm_mask = norm_v == 0
    
    # 벡터 정규화   newaxis -> norm_v 배열을 (20,1)로 바꿔 v와 차원 맞춰줌
    v[~zero_norm_mask] = v[~zero_norm_mask] / norm_v[~zero_norm_mask][:, np.newaxis]
    v[zero_norm_mask] = 0  # 크기가 0인 벡터를 명시적으로 0으로 설정

    # 도트 곱 계산
    dot_product = np.clip(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]), -1.0, 1.0)
    
    # 각도 계산
    angle = np.arccos(dot_product) # (15,)

    return angle.flatten()

def angle_pose(joint_pose):
    v1 = joint_pose[[0,1, 1,3,5,5, 5,0,2, 2,4,6, 6, 6], :3] # 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    v2 = joint_pose[[1,3,13,5,7,9,11,2,4,14,6,8,10,12], :3] # 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14
    v = v2 - v1 # (14, 3)

    norm_v = np.linalg.norm(v, axis=1)

    zero_norm_mask = norm_v == 0

    v[~zero_norm_mask] = v[~zero_norm_mask] / norm_v[~zero_norm_mask][:, np.newaxis]
    v[zero_norm_mask] = 0

    dot_product = np.clip(np.einsum('nt,nt->n',
        v[[0,0,1,3,3,3,7,7, 8,10,10,10],:], 
        v[[1,2,3,4,5,6,8,9,10,11,12,13],:]), -1.0, 1.0)
    
    angle = np.arccos(dot_product) # (12,)

    return angle.flatten()