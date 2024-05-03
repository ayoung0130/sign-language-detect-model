import numpy as np
import os

folder_path = f"C:/Users/mshof/Desktop/seq_data_xyz"

# 각 파일의 시퀀스 데이터 길이 측정
max_seq_length = 0
for npy_file in os.listdir(folder_path):
    # 파일 불러오기
    file_path = os.path.join(folder_path, npy_file)

    # 각 파일의 시퀀스 데이터 길이 측정
    seq_data = np.load(file_path)
    seq_length = len(seq_data)
    print("조정 전:")
    print(f"{os.path.basename(file_path)}: {seq_length}")
    max_seq_length = max(max_seq_length, seq_length)

print("\nmax_len:", max_seq_length)


# 모든 파일의 시퀀스 데이터를 최대 길이에 맞춰 재조정
for npy_file in os.listdir(folder_path):
    # 파일 불러오기
    file_path = os.path.join(folder_path, npy_file)
    base_name = os.path.basename(file_path)

    seq_data = np.load(file_path)
    seq_length = len(seq_data)
    
    if seq_length < max_seq_length:
        # 부족한 부분은 0으로 패딩하여 최대 길이에 맞춤
        padding_length = max_seq_length - seq_length
        padding = np.zeros((padding_length, seq_data.shape[1], seq_data.shape[2]))
        seq_data = np.concatenate([seq_data, padding], axis=0)
    
    # 재조정된 시퀀스 데이터를 다시 npy 파일로 저장
    np.save(os.path.join(file_path, f'{base_name}_{max_seq_length}'), seq_data)


# 확인용
for npy_file in os.listdir(folder_path):
    # 파일 불러오기
    file_path = os.path.join(folder_path, npy_file)

    # 각 파일의 시퀀스 데이터 길이 측정
    seq_data = np.load(file_path)
    seq_length = len(seq_data)
    print("\n조정 후:")
    print(f"{os.path.basename(file_path)}: {seq_length}")