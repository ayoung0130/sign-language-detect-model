import numpy as np
import os, time
from setting import seq_length, jumping_window
from dotenv import load_dotenv

# 넘파이 파일을 concatenate 후 시퀀스 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# "npy" "npy_flip" "npy_shift" "npy_flip_shift"
# "landmarks" "landmarks_angle" "landmarks_visibility" "landmarks_visibility_angle"
folder_name = "npy_flip_shift"

seq_save_path = os.path.join(base_dir, "seq_data/landmarks_angle")

full_seq_data = []
count = 0

for idx in range(0, 10):
    folder_path = os.path.join(base_dir, f"{folder_name}/landmarks_angle/{idx}")

    for npy_file in os.listdir(folder_path):
        # 파일 불러오기
        file_path = os.path.join(folder_path, npy_file)
        data = np.load(file_path)

        if len(data) > seq_length:

            # 시퀀스 데이터 생성
            data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, jumping_window)]
            data = np.array(data)

            # 시퀀스 데이터를 full_seq_data에 추가
            full_seq_data.append(data)
        
        count += 1

    print(f"Index {idx} 시퀀스 데이터 생성 및 추가 완료")

# full_seq_data 리스트에 있는 모든 시퀀스 데이터를 concatenate하여 하나의 배열로 만듦
full_seq_data = np.concatenate(full_seq_data)

created_time = int(time.time())

np.save(os.path.join(seq_save_path, f'seq_{folder_name}_{created_time}'), full_seq_data)
print("full seq data shape:",  full_seq_data.shape)
print(f"npy 파일 개수: {count}개")