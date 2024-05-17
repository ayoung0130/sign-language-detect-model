import mediapipe as mp
from PIL import ImageFont

# 미디어 파이프 hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 미디어 파이프 pose 모델 초기화 + 사용할 랜드마크 관절 번호만 추출
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_landmark_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# 랜드마크 그리기 유틸
mp_drawing = mp.solutions.drawing_utils

# 학습할 수어 단어
actions = ['가렵다', '기절', '부러지다', '어제', '어지러움', '열나다', '오늘', '진통제', '창백하다', '토하다']

# 시퀀스 길이
seq_length = 30

# 폰트 설정
font = ImageFont.truetype('fonts/NanumGothicBold.ttf', 25)