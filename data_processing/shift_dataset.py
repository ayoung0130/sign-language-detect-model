import numpy as np
import os
from dotenv import load_dotenv

# 랜드마크 좌표값을 이동

load_dotenv()
base_dir = os.getenv('BASE_DIR')

folder_path = os.path.join(base_dir, 'npy/landmarks')
save_path = os.path.join(base_dir, 'npy_shift/landmarks')
file_count = 0

def shift_data():
    scales = [0.8, 1.2]

    for npy_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        data = np.load(file_path)
        file_count += 1
        
        for scale in scales:
            print("shift 전: ", data[100, 80:82])

            # 0번 인덱스부터 249번 인덱스까지
            # (x 좌표값) * (이동시킬 퍼센테이지)
            for x in range(0, 250, 4):
                data[:, x] = data[:, x] * scale
            
            # 1번 인덱스부터 250번 인덱스까지
            # (y 좌표값) * (이동시킬 퍼센테이지)
            for y in range(1, 251, 4):
                data[:, y] = data[:, y] * scale

            print("shift 후: ", data[100, 80:82])
            print("")

            # 수정된 데이터를 저장
            np.save(os.path.join(save_path, f"{scale}_" + base_name), data)

shift_data()
print(f"{file_count}개 파일 shift 완료")