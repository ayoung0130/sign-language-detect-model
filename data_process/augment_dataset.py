import numpy as np
import os

# 데이터를 증강하는 코드
# 6/4 - flip, shift

folder_path = "C:/Users/mshof/Desktop/angle"

save_path = "C:/Users/mshof/Desktop/shift"

def augment_data():
    for npy_file in os.listdir(folder_path):
        # 파일 불러오기
        file_path = os.path.join(folder_path, npy_file)
        base_name = os.path.basename(file_path)
        data = np.load(file_path)

        #
        ### flip
        #
        # print("flip 전: ", data[100, 232:250])

        # 0번 인덱스부터 249번 인덱스까지
        # 1 - (x 좌표값)
        # for x in range(0, 250, 4):
        #     nonzero_indices = np.nonzero(data[:, x])[0]  # 0이 아닌 값의 인덱스를 찾기
        #     if len(nonzero_indices) > 0:  # 0이 아닌 값이 하나라도 있는 경우에만 연산을 수행
        #         data[:, x][nonzero_indices] = 1 - data[:, x][nonzero_indices]

        # print("flip 후: ", data[100, 232:250])
        # print("")


        #
        ### shift
        #
        print("shift 전: ", data[120, 248:253])

        # 0번 인덱스부터 249번 인덱스까지
        # (x 좌표값) * (이동시킬 퍼센테이지)
        for x in range(0, 250, 4):
            data[:, x] = data[:, x] * 1.1
        
        # 1번 인덱스부터 250번 인덱스까지
        # (y 좌표값) * (이동시킬 퍼센테이지)
        for y in range(1, 251, 4):
            data[:, y] = data[:, y] * 1.1

        print("shift 후: ", data[120, 248:253])
        print("")


        # 수정된 데이터를 저장
        np.save(os.path.join(save_path, "1.1_" + base_name), data)

augment_data()