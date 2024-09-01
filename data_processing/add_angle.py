import os
import numpy as np
from landmark_processing import angle_hands, angle_pose
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('BASE_DIR')

def add_angle():
    # 데이터 경로
    folder_path = os.path.join(base_dir, f"npy_flip/0_9")
    save_path = os.path.join(base_dir, f"npy_flip_angle/0_9")

    for file in os.listdir(folder_path):
        base_name = os.path.basename(file)

        data_path = os.path.join(folder_path, file)

        # numpy 파일 로드
        data = np.load(data_path)

        # 왼손(0번 인덱스부터 62번 인덱스까지): (n, 63) -> (n, 21, 3)
        left_hand_features = data[:, :63].reshape(-1, 21, 3)

        # 오른손(63번 인덱스부터 125번 인덱스까지): (n, 63) -> (n, 21, 3)
        right_hand_features = data[:, 63:126].reshape(-1, 21, 3)

        # 포즈(126번 인덱스부터 170번 인덱스까지): (n, 45) -> (n, 15, 3)
        pose_features = data[:, 126:171].reshape(-1, 15, 3)

        # 레이블(171번 인덱스)
        labels = data[:, 171]

        # 최종 결과를 담을 리스트 초기화
        final_data  = []

        for i in range(data.shape[0]):  # 프레임 수만큼 반복
            # 각 프레임에서 왼손, 오른손, 포즈의 각도 계산
            left_hand_angle = angle_hands(left_hand_features[i])
            right_hand_angle = angle_hands(right_hand_features[i])
            pose_angle = angle_pose(pose_features[i])
            
            # 각도 데이터를 기존 피처와 결합
            frame_data = np.hstack((
                left_hand_features[i].flatten(),  # 왼손 피처 (21, 3) -> (63,)
                right_hand_features[i].flatten(),  # 오른손 피처 (21, 3) -> (63,)
                pose_features[i].flatten(),  # 포즈 피처 (15, 3) -> (45,)
                left_hand_angle,  # 왼손 각도
                right_hand_angle,  # 오른손 각도
                pose_angle,  # 포즈 각도
                labels[i]  # 레이블
            ))

            # 결합된 프레임 데이터를 리스트에 추가
            final_data.append(frame_data)

        # 리스트를 numpy 배열로 변환
        final_data = np.array(final_data)

        # 변형된 데이터를 새로운 파일로 저장
        np.save(os.path.join(save_path, base_name), final_data)

        print(final_data.shape)
        print(final_data[100])

def add_angle_test():
    # 데이터 경로
    folder_path = os.path.join(base_dir, f"test_npy")
    save_path = os.path.join(base_dir, f"test_npy_angle")

    for file in os.listdir(folder_path):
        base_name = os.path.basename(file)

        data_path = os.path.join(folder_path, file)

        # numpy 파일 로드
        data = np.load(data_path)

        # 왼손(0번 인덱스부터 62번 인덱스까지): (n, 63) -> (n, 21, 3)
        left_hand_features = data[:, :63].reshape(-1, 21, 3)

        # 오른손(63번 인덱스부터 125번 인덱스까지): (n, 63) -> (n, 21, 3)
        right_hand_features = data[:, 63:126].reshape(-1, 21, 3)

        # 포즈(126번 인덱스부터 170번 인덱스까지): (n, 45) -> (n, 15, 3)
        pose_features = data[:, 126:171].reshape(-1, 15, 3)

        # 최종 결과를 담을 리스트 초기화
        final_data  = []

        for i in range(data.shape[0]):  # 프레임 수만큼 반복
            # 각 프레임에서 왼손, 오른손, 포즈의 각도 계산
            left_hand_angle = angle_hands(left_hand_features[i])
            right_hand_angle = angle_hands(right_hand_features[i])
            pose_angle = angle_pose(pose_features[i])
            
            # 각도 데이터를 기존 피처와 결합
            frame_data = np.hstack((
                left_hand_features[i].flatten(),  # 왼손 피처 (21, 3) -> (63,)
                right_hand_features[i].flatten(),  # 오른손 피처 (21, 3) -> (63,)
                pose_features[i].flatten(),  # 포즈 피처 (15, 3) -> (45,)
                left_hand_angle,  # 왼손 각도
                right_hand_angle,  # 오른손 각도
                pose_angle  # 포즈 각도
            ))

            # 결합된 프레임 데이터를 리스트에 추가
            final_data.append(frame_data)

        # 리스트를 numpy 배열로 변환
        final_data = np.array(final_data)

        # 변형된 데이터를 새로운 파일로 저장
        np.save(os.path.join(save_path, base_name), final_data)

        print(final_data.shape)
        print(final_data[5])

add_angle()
add_angle_test()