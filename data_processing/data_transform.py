import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 데이터 경로
folder_path = os.path.join(base_dir, f"npy_flip/0_9_0829")
save_path = os.path.join(base_dir, f"npy_flip/0_9")

for file in os.listdir(folder_path):
    base_name = os.path.basename(file)

    data_path = os.path.join(folder_path, file)

    # numpy 파일 로드
    data = np.load(data_path)

    # 첫 171개의 피처만 선택
    transformed_data = data[:, :171]

    # 변형된 데이터를 새로운 파일로 저장
    np.save(os.path.join(save_path, base_name), transformed_data)

    print(transformed_data.shape)