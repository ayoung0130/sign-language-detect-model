import mediapipe as mp
from PIL import ImageFont

# 모델 초기화, 유틸 선언, 단어 리스트, 시퀀스 길이, 폰트 등을 세팅하는 코드

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
actions = ['가렵다', '기절', '부러지다', '어제', '어지러움', '열나다', '오늘', '진통제', '창백하다', '토하다',
           '배', '다리', '어깨', '눈', '코', '목', '머리', '발', '아프다', '위', ] 
        #    '음식물', '깔리다', '인대', '뼈', '알려주세요', '내일', '피나다', '손', '월요일', '화요일', ]
        #    '수요일', '목요일', '금요일', '토요일', '일요일', '등', '무릎', '병원', '얼굴', '의사', 
        #    '귀', '두드러기생기다', '기침', '근육', '호흡곤란', '화상', '아래', '오른쪽', '왼쪽', '끝', ]
        #    '부터', '~적 없다', '~적 있다',]

# 윈도우 사이즈
seq_length = 15
jumping_window = 15

# 폰트 설정
font = ImageFont.truetype('fonts/NanumGothicBold.ttf', 25)