import numpy as np

# 파일 경로
file_1_path = '1_가렵다_flip_test.npy'
file_2_path = '1_가렵다.npy'

# 넘파이 파일 로드
file_1 = np.load(file_1_path, allow_pickle=True)
file_2 = np.load(file_2_path, allow_pickle=True)

# 데이터 형태 비교
shape_1 = file_1.shape
shape_2 = file_2.shape

# 차이 계산
difference = np.sum(np.abs(file_1 - file_2))
max_difference = np.max(np.abs(file_1 - file_2))
mean_difference = np.mean(np.abs(file_1 - file_2))

print(f"Shape of file 1: {shape_1}")
print(f"Shape of file 2: {shape_2}")
print(f"Total difference: {difference}")
print(f"Max difference: {max_difference}")
print(f"Mean difference: {mean_difference}")
