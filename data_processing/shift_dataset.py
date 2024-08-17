import numpy as np
import os
from dotenv import load_dotenv

# 랜드마크 좌표값을 이동

load_dotenv()
base_dir = os.getenv('BASE_DIR')

# "npy" "npy_flip"
original = "npy_flip"
folder_path = os.path.join(base_dir, f'npy/0_9_xy')              # 0_9 10_19 20_29 30_39 40_49 50_52
save_path = os.path.join(base_dir, f'{original}_shift/0_9_xy')

# 폴더 이름에 "visibility"가 포함되면 col을 4로, 그렇지 않으면 3으로 설정
col = 2 if "xy" in folder_path else 3

def shift_data():

    scales = [0.9, 1.1]

    for npy_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        
        for scale in scales:
            data = np.load(file_path)

            print(f"{base_name} {scale} shift 전: ", data[100, 0:5])

            if col == 3:
                # only landmarks
                # 0번 인덱스부터 186번 인덱스까지
                for x in range(0, 187, 3):
                    data[:, x] = data[:, x] * scale # (x 좌표값) * (이동시킬 퍼센테이지)
                
                # 1번 인덱스부터 187번 인덱스까지
                for y in range(1, 188, 3):
                    data[:, y] = data[:, y] * scale # (y 좌표값) * (이동시킬 퍼센테이지)

            elif col == 2:
                # landmarks (xy)
                # 0번 인덱스부터 124번 인덱스까지
                for x in range(0, 125, 2):
                    data[:, x] = data[:, x] * scale
                
                # 1번 인덱스부터 125번 인덱스까지
                for y in range(1, 126, 2):
                    data[:, y] = data[:, y] * scale

            print(f"{base_name} {scale} shift 후: ", data[100, 0:5])
            print("")

            # 수정된 데이터를 저장
            np.save(os.path.join(save_path, f"{scale}_" + base_name), data)

shift_data()
print("파일 shift 완료")