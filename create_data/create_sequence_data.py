import numpy as np
import os, time
from setting import seq_length, jumping_window
from dotenv import load_dotenv

# 넘파이 파일을 concatenate 후 시퀀스 배열로 변환하는 코드

load_dotenv()
base_dir = os.getenv('BASE_DIR')

folder_names = ["npy", "npy_flip", "npy_shift", "npy_flip_shift"]   # "npy", "npy_flip", "npy_shift", "npy_flip_shift" / "npy0", "npy0_flip", "npy0_shift", "npy0_flip_shift"
idx_list = ["0_9", "10_19", "20_29", "30_39"]  #"0_9", "10_19", "20_29", "30_39", "40_52"

seq_save_path = os.path.join(base_dir, "seq_data/40_words")   # seq_data / seq_data0

for folder_name in folder_names:

    full_seq_data = []
    count = 0

    for idx in idx_list:
        folder_path = os.path.join(base_dir, f"{folder_name}/{idx}")

        for npy_file in os.listdir(folder_path):
            # 파일 불러오기
            file_path = os.path.join(folder_path, npy_file)
            data = np.load(file_path).astype(np.float32)

            # 시퀀스 길이보다 데이터 길이가 작은 경우 패딩 적용
            if len(data) < seq_length:
                padding_length = seq_length - len(data)
                # 시퀀스의 부족한 부분을 0으로 채움
                data = np.pad(data, ((0, padding_length), (0, 0)), mode='constant')

            # 시퀀스 데이터 생성
            data = [data[seq:seq + seq_length] for seq in range(0, len(data) - seq_length + 1, jumping_window)]
            data = np.array(data)

            # 시퀀스 데이터를 full_seq_data에 추가
            full_seq_data.append(data)
            
            count += 1

    # full_seq_data 리스트에 있는 모든 시퀀스 데이터를 concatenate하여 하나의 배열로 만듦
    full_seq_data = np.concatenate(full_seq_data, axis=0)

    created_time = int(time.time())

    np.save(os.path.join(seq_save_path, f'seq_{folder_name}_{created_time}_{seq_length}_{jumping_window}'), full_seq_data)
    print("full seq data shape:",  full_seq_data.shape)
    print(f"npy 파일 개수: {count}개")