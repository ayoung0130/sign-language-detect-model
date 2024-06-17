import numpy as np
import os
from dotenv import load_dotenv

# 데이터를 증강하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# 6/16 - shift
folder_path = os.path.join(base_dir, 'angle_flip')

save_path = os.path.join(base_dir, 'shift_flip')

def augment_data():
    scales = [0.8, 0.9, 1.1, 1.2]

    for npy_file in os.listdir(folder_path):
        # 파일 불러오기
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        data = np.load(file_path)

        #
        ### shift
        #
        for scale in scales:
            print("shift 전: ", data[120, 248:253])

            # 0번 인덱스부터 249번 인덱스까지
            # (x 좌표값) * (이동시킬 퍼센테이지)
            for x in range(0, 250, 4):
                data[:, x] = data[:, x] * scale
            
            # 1번 인덱스부터 250번 인덱스까지
            # (y 좌표값) * (이동시킬 퍼센테이지)
            for y in range(1, 251, 4):
                data[:, y] = data[:, y] * scale

            print("shift 후: ", data[120, 248:253])
            print("")


            # 수정된 데이터를 저장
            np.save(os.path.join(save_path, f"{scale}_" + base_name), data)

augment_data()