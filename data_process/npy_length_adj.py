import numpy as np
import os

folder_path = "C:/Users/mshof/Desktop/npy_data"
pad_save_path = "C:/Users/mshof/Desktop/pad_npy_data"
slice_save_path = "C:/Users/mshof/Desktop/slice_npy_data"

# 각 파일의  데이터 길이 측정
max_length = 0
min_length = 1000000

print("조정 전:")
for npy_file in os.listdir(folder_path):
    # 파일 불러오기
    file_path = os.path.join(folder_path, npy_file)

    # 각 파일의 시퀀스 데이터 길이 측정
    data = np.load(file_path)
    length = len(data)
    
    print(f"{os.path.basename(file_path)}: {length}")
    max_length = max(max_length, length)
    min_length = min(min_length, length)

print("\nmax_len:", max_length)
print("min_len:", min_length)
print("")

# 모든 파일의 시퀀스 데이터를 최대 길이에 맞춰 패딩
for npy_file in os.listdir(folder_path):
    # 파일 불러오기
    file_path = os.path.join(folder_path, npy_file)
    base_name = os.path.basename(file_path)

    data = np.load(file_path)
    length = len(data)
    
    if length < max_length:
        # 부족한 부분은 0으로 패딩하여 최대 길이에 맞춤
        padding_length = max_length - length
        padding = np.zeros((padding_length, data.shape[1]))

        # 인덱스 추출, 추가
        idx = data[0, -1]
        padding[:, -1] = idx

        data = np.concatenate([data, padding], axis=0)
    
    np.save(os.path.join(pad_save_path, "pad_" + base_name), data)
    print(f"{os.path.basename(file_path)}: {len(data)}")


# 모든 파일의 시퀀스 데이터를 최소 길이에 맞춰 자르기
# for npy_file in os.listdir(folder_path):
#     # 파일 불러오기
#     file_path = os.path.join(folder_path, npy_file)
#     base_name = os.path.basename(file_path)

#     data = np.load(file_path)
#     length = len(data)
    
#     if length > min_length:
#         slice_data = data[:min_length, :]

#         # 인덱스 추출, 추가
#         idx = data[0, -1]
#         slice_data[:, -1] = idx
    
#     np.save(os.path.join(slice_save_path, "slice_" + base_name), slice_data)
#     print(f"{os.path.basename(file_path)}: {len(slice_data)}")