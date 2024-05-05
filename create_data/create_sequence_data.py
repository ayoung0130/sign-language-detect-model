import numpy as np
import os, time

folder_path = f"C:/Users/mshof/Desktop/npy_data"
seq_save_path = "C:/Users/mshof/Desktop/seq_data/0505"

full_seq_data = []
seq_length = 60

for npy_file in os.listdir(folder_path):
    # 파일 불러오기
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)

    # 시퀀스 데이터 생성
    data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
    data = np.array(data)
    print(f"{os.path.basename(file_path)} seq data shape:", data.shape)

    # 시퀀스 데이터를 full_seq_data에 추가
    full_seq_data.append(data)

# full_seq_data 리스트에 있는 모든 시퀀스 데이터를 concatenate하여 하나의 배열로 만듦
full_seq_data = np.concatenate(full_seq_data, axis=0)

created_time = int(time.time())

np.save(os.path.join(seq_save_path, f'seq_{created_time}'), full_seq_data)
print("full seq data shape:",  full_seq_data.shape)