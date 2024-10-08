import numpy as np
import os
from dotenv import load_dotenv

# 랜드마크 좌표값을 이동

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# "npy" "npy_flip"
original = "npy_flip"
folder_path = os.path.join(base_dir, f'{original}/test2')              # 0_9 10_19 20_29 30_39 40_49 50_52
save_path = os.path.join(base_dir, f'{original}_shift/test2')

def shift_data():
    scales = [0.9, 1.1]

    for npy_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        
        for scale in scales:
            data = np.load(file_path)

            print(f"{base_name} {scale} shift 전: ", data[100, 0:5])

            # shift
            for x in range(0, 188, 3):
                data[:, x] = data[:, x] * scale # (x 좌표값) * (이동시킬 퍼센테이지)
            
            for y in range(1, 188, 3):
                data[:, y] = data[:, y] * scale # (y 좌표값) * (이동시킬 퍼센테이지)

            print(f"{base_name} {scale} shift 후: ", data[100, 0:5])
            print("")

            # 수정된 데이터를 저장
            np.save(os.path.join(save_path, f"{scale}_" + base_name), data)

shift_data()
print("파일 shift 완료")