import numpy as np
import os
folder_path = "C:/Users/_/Desktop/pad_npy_data"
# folder_path = "C:/Users/_/Desktop/slice_npy_data"
# folder_path = "C:/Users/mshof/Desktop/shift_npy_data"

save_path = "C:/Users/_/Desktop/flip_pad_npy_data"
# save_path = "C:/Users/_/Desktop/flip_slice_npy_data"
# save_path = "C:/Users/mshof/Desktop/flip_shift_npy_data"

def flip():
    for npy_file in os.listdir(folder_path):
        # 파일 불러오기
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        data = np.load(file_path)

        print("flip 전: ", data[500, 248:253])

        # 1 - (x 좌표값)
        for x in range(0, 249, 4):
            data[:, x] = 1 - data[:, x]

        
        print("flip 후: ", data[500, 248:253])
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