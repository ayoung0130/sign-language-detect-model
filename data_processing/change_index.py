import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('BASE_DIR')

def update_index_in_numpy(file_path, new_index, output_path):
    # 넘파이 파일 불러오기
    data = np.load(file_path)

    # 기존 인덱스 값 확인
    original_index = data[1, -1]
    
    # 인덱스 값 수정 (피처의 마지막 요소)
    data[:, -1] = new_index
    
    # 수정된 데이터 저장
    np.save(output_path, data)
    print(data[100])
    print(f"인덱스 {original_index} -> {new_index} 수정 완료")

file_path = os.path.join(base_dir, "npy_flip/must/flip_끝_1728529120.npy")
new_index = 38.0  # 수정할 인덱스 값
output_path = os.path.join(base_dir, "npy_flip/30_39/flip_끝_1728529120")  # 수정된 파일 저장 경로

update_index_in_numpy(file_path, new_index, output_path)