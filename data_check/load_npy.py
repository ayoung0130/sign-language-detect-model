import os, json
import numpy as np
from dotenv import load_dotenv

# numpy 파일을 불러와 데이터를 체크하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')
 
# 넘파이 파일 불러오기
npy_file1 = np.load(os.path.join(base_dir, 'data_android.npy'))
npy_file2 = np.load(os.path.join(base_dir, '10_words_xy/1_가렵다.npy'))

#
### 비교
#

# 출력 및 비교 함수 정의
def print_and_compare(file1, file2, start_idx, end_idx, label):
    print(f"\nComparing {label} (indices {start_idx} to {end_idx}):")
    print(f"Frame 20 - {label}:")
    print("File 1:", file1[20, start_idx:end_idx])
    print("File 2:", file2[20, start_idx:end_idx])
    print("-" * 50)

# 왼손 비교 (인덱스 0번부터 41번 인덱스까지)
print_and_compare(npy_file1, npy_file2, 0, 42, "Left hand")

# 오른손 비교 (인덱스 42번부터 83번 인덱스까지)
print_and_compare(npy_file1, npy_file2, 42, 84, "Right hand")

# 포즈 비교 (인덱스 84번부터 125번 인덱스까지)
print_and_compare(npy_file1, npy_file2, 84, 126, "Pose")

# 왼손 각도 비교 (인덱스 126번부터 140번 인덱스까지)
print_and_compare(npy_file1, npy_file2, 126, 141, "Left hand angle")

# 오른손 각도 비교 (인덱스 141번부터 155번 인덱스까지)
print_and_compare(npy_file1, npy_file2, 141, 156, "Right hand angle")

# 포즈 각도 비교 (인덱스 156번부터 170번 인덱스까지)
print_and_compare(npy_file1, npy_file2, 156, 171, "Pose angle")



#
### 저장 및 출력
#

# # 넘파이 배열을 CSV 파일로 저장
# np.savetxt(os.path.join(base_dir, '1_가렵다_output.csv'), npy_file, delimiter=',')

# # 넘파이 배열을 텍스트 파일로 저장
# np.savetxt(os.path.join(base_dir, '2_토하다_output.txt'), npy_file, delimiter=',')

# # JSON 파일로 저장
# with open(os.path.join(base_dir, '1_가렵다_output.json'), 'w') as f:
#     json.dump(npy_file.tolist(), f)

# # 넘파이 배열 전체를 출력하도록 설정
# np.set_printoptions(threshold=np.inf)
# print(npy_file)