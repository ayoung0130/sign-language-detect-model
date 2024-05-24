import numpy as np
import matplotlib.pyplot as plt
from setting import actions

# To-do: 그래프를 더 의미있는 자료로 나타낼 방안 논의 (한 프레임당의 정확도는 중요x?)

idx = 1
action = actions[idx]

# 불러올 파일 경로 설정
file_paths = [
    f'C:/Users/mshof/Desktop/pred/1_{action}.mp4_부러지다.npy'
]

# 0번 인덱스에 해당하는 값을 저장할 리스트
itch_data = []

# 각 파일에서 0번 인덱스 값을 추출하여 리스트에 추가
for file_path in file_paths:
    y_pred = np.load(file_path)
    itch_values = y_pred[:, idx]  # 해당 인덱스의 값 추출

    # 파일 이름에 따른 색상 지정
    if '1' in file_path:
        label = 'Person1'
        color = 'green'
    elif '2' in file_path:
        label = 'Person2'
        color = 'orange'
    elif '3' in file_path:
        label = 'Person3(flip)'
        color = 'blue'
    else:
        color = 'black'  # 기본 색상

    # 그래프 그리기
    plt.plot(itch_values, label=label, color=color, marker='o')

# 그래프 설정
plt.title(f'Predicted Values for Index {idx}')
plt.xlabel('Frame Index')
plt.ylabel('Predicted Value')
plt.legend()
plt.grid(True)
plt.show()