import numpy as np
import os, time

folder_path = f"C:/Users/mshof/Desktop/data"

seq_save_path = "C:/Users/mshof/Desktop/seq_data_0504/"

full_data = []

created_time = int(time.time())

seq_length = 60

for npy_file in os.listdir(folder_path):
    # 파일 불러오기
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)

    # 배열에 추가
    full_data.append(data)
    print(f"{os.path.basename(file_path)} 통합 완료")

full_data = np.concatenate(full_data, axis=0)
print("full data shape:",  full_data.shape)

# 시퀀스 데이터 저장
full_seq_data = [full_data[seq:seq + seq_length] for seq in range(len(full_data) - seq_length)]
full_seq_data = np.array(full_seq_data)
np.save(os.path.join(seq_save_path, f'seq_{created_time}'), full_seq_data)
print("seq data shape:",  full_seq_data.shape)