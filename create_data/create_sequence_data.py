import os, time
from setting import seq_length



# 시퀀스 데이터 저장
# full_seq_data = [data[seq:seq + seq_length] for seq in range(len(data) - seq_length)]
# full_seq_data = np.array(full_seq_data)
# np.save(os.path.join(seq_save_path, f'seq_{action}_{created_time}'), full_seq_data)
# print("seq data shape:", action, full_seq_data.shape)