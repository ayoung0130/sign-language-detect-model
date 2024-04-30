import numpy as np


def angleHands(joint_hands):
    # 관절 간의 각도 계산
    v1 = joint_hands[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint  각 관절은 [x, y, z] 좌표로 표현되므로 :3
    v2 = joint_hands[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # [20, 3]. 20개 행과 3개 열

    # 벡터 정규화
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # 각도 계산 (arccos를 이용하여 도트 곱의 역순 취함)
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

    # 라벨에 각도 정보 추가
    angle_label = np.array([angle], dtype=np.float32)
    
    return angle_label.flatten()


def anglePose(joint_pose):
    v1 = joint_pose[[0, 2, 0, 1, 0, 0, 7, 8, 8, 8, 10, 12, 12, 12, 7, 7, 9, 11, 11, 11], :3]
    v2 = joint_pose[[2, 4, 1, 3, 5, 6, 8, 7, 10, 20, 12, 14, 16, 18, 9, 19, 11, 13, 15, 17], :3]
    # 0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    # 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

    v = v2 - v1

    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],:], 
        v[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],:])) # [15,]

    angle_label = np.array([angle], dtype=np.float32)

    return angle_label.flatten()


def setVisibility(x, y, epsilon=1e-6):
    
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