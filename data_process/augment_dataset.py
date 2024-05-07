import numpy as np
import os
# folder_path = "C:/Users/mshof/Desktop/pad_npy_data"
folder_path = "C:/Users/mshof/Desktop/slice_npy_data"
# folder_path = "C:/Users/mshof/Desktop/shift_npy_data"

# save_path = "C:/Users/mshof/Desktop/flip_pad_npy_data"
save_path = "C:/Users/mshof/Desktop/flip_slice_npy_data"
# save_path = "C:/Users/mshof/Desktop/flip_shift_npy_data"

def flip():
    for npy_file in os.listdir(folder_path):
        # 파일 불러오기
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        data = np.load(file_path)

        print("flip 전: ", data[100, 248:253])

        # 0번 인덱스부터 249번 인덱스까지
        # 1 - (x 좌표값)
        for x in range(0, 250, 4):
            nonzero_indices = np.nonzero(data[:, x])[0]  # 0이 아닌 값의 인덱스를 찾습니다.
            if len(nonzero_indices) > 0:  # 0이 아닌 값이 하나라도 있는 경우에만 연산을 수행합니다.
                data[:, x][nonzero_indices] = 1 - data[:, x][nonzero_indices]

        
        print("flip 후: ", data[100, 248:253])
        print("")

        # 수정된 데이터를 저장
        np.save(os.path.join(save_path, "flip_" + base_name), data)


def shift():
    for npy_file in os.listdir(folder_path):
        # 파일 불러오기
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        data = np.load(file_path)

        print("shift 전: ", data[500, :12])

        # 특정 값을 더해 이동
        # data[:, 0] += 
        # data[:, 1] += 
        # data[:, 2] += 
        
        print("shift 후: ", data[500, :12])
        print("")

        # 수정된 데이터를 저장
        np.save(os.path.join(save_path, "shift_" + base_name), data)


flip()
# shift()