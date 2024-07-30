import numpy as np
import os
from dotenv import load_dotenv

# 랜드마크 좌표값을 x축 반전

load_dotenv()
base_dir = os.getenv('BASE_DIR')

folder_path = os.path.join(base_dir, '')
save_path = os.path.join(base_dir, '')
file_count = 0

def flip_data():

    for npy_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        data = np.load(file_path)
        file_count += 1
        
        # 왼손의 x좌표 인덱스
        left_hand_indices = list(range(0, 84, 4))
        # 오른손의 x좌표 인덱스
        right_hand_indices = list(range(84, 168, 4))
        # 포즈의 x좌표 인덱스
        pose_indices = list(range(168, 252, 4))
        
        # 왼손 x좌표를 1에서 빼기
        data[:, left_hand_indices] = 1 - data[:, left_hand_indices]
        
        # 오른손 x좌표를 1에서 빼기
        data[:, right_hand_indices] = 1 - data[:, right_hand_indices]
        
        # 왼손과 오른손 데이터 위치 바꾸기
        left_hand_data = data[:, left_hand_indices].copy()
        right_hand_data = data[:, right_hand_indices].copy()
        data[:, left_hand_indices] = right_hand_data
        data[:, right_hand_indices] = left_hand_data
        
        # 포즈 x좌표를 1에서 빼기
        data[:, pose_indices] = 1 - data[:, pose_indices]

        # 수정된 데이터를 저장
        np.save(os.path.join(save_path, f"flip_{base_name}"), data)

flip_data()
print(f"{file_count}개 파일 flip 완료")