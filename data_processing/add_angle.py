import numpy as np
import os
from dotenv import load_dotenv

# 랜드마크 좌표값을 통해 각도값 계산, 데이터에 추가

def add_angle():

    load_dotenv()
    base_dir = os.getenv('BASE_DIR')

    folder_path = os.path.join(base_dir, 'npy/landmarks')
    save_path = os.path.join(base_dir, 'npy/landmarks_angle')

    for npy_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)

        data = np.load(file_path)
        print(f"{base_name} 프레임 수: {data.shape[0]}. shape: {data.shape}")

        # 열 개수 (visibility는 4)
        colm = 3

        all_frames = []

        # 매 프레임마다
        for frame in range(0, data.shape[0]):

            # 데이터의 특정 범위를 추출하여 손과 포즈 랜드마크 할당
            joint_left_hands = data[frame, 0:63].reshape(-1, colm)
            joint_right_hands = data[frame, 63:126].reshape(-1, colm)
            joint_pose = data[frame, 126:189].reshape(-1, colm)

            # 각도 값을 데이터에 삽입
            angles_combined = np.concatenate((angle_hands(joint_left_hands), angle_hands(joint_right_hands), angle_pose(joint_pose)), axis=0)
            
            # 기존 데이터와 각도를 결합
            combined_data = np.concatenate((data[frame, :-1], angles_combined, [data[frame, -1]]))
            all_frames.append(combined_data)

        # 리스트를 numpy 배열로 변환
        data_with_angles = np.array(all_frames)

        # 수정된 데이터를 저장
        np.save(os.path.join(save_path, base_name), data_with_angles)
        print(f"{base_name} 저장 완료. shape: {data_with_angles.shape}")
        print("")

def angle_hands(joint_hands):
    # 관절 간의 각도 계산
    v1 = joint_hands[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint  각 관절은 [x, y, z] 좌표로 표현되므로 :3
    v2 = joint_hands[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # [20, 3]. 20개 행과 3개 열

    # 벡터 정규화
    norm_v = np.linalg.norm(v, axis=1)

    if np.all(norm_v == 0):
        angle = np.zeros([15,])
    else: 
        v = v / norm_v[:, np.newaxis]
        # 각도 계산 (arccos를 이용하여 도트 곱의 역순 취함)
        dot_product = np.clip(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]), -1.0, 1.0)
        angle = np.arccos(dot_product) # [15,]

    return angle.flatten()

def angle_pose(joint_pose):
    v1 = joint_pose[[0, 2, 0, 1, 0, 0, 7, 8, 8, 8, 10, 12, 12, 12, 7, 7, 9, 11, 11, 11], :3]
    v2 = joint_pose[[2, 4, 1, 3, 5, 6, 8, 7, 10, 20, 12, 14, 16, 18, 9, 19, 11, 13, 15, 17], :3]
    # 0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    # 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    v = v2 - v1

    # 벡터 정규화
    norm_v = np.linalg.norm(v, axis=1)
    if np.all(norm_v == 0):
        angle = np.zeros([15,])
    else: 
        v = v / norm_v[:, np.newaxis]
        # 각도 계산 (arccos를 이용하여 도트 곱의 역순 취함)
        dot_product = np.clip(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]), -1.0, 1.0)
        angle = np.arccos(dot_product) # [15,]

    return angle.flatten()

add_angle()
print("각도 추가 완료")